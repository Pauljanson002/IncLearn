import math

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ICIFAR100,get_train_transforms,get_test_transforms,get_train_dataset,get_test_dataset
from models import get_model
from torchvision import transforms
from augmentations import CIFAR10Policy
from torch import optim
from torch.nn import functional as F
from utils import save_checkpoint, ensure_dir, load_checkpoint, get_optimizer, get_scheduler
import copy

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def get_one_hot(target, num_class):
    one_hot = torch.zeros(target.shape[0], num_class).to(device)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
    return one_hot


def adjust_learning_rate(optimizer, epoch, learning_rate, final_epoch, warmup=0):
    lr = learning_rate
    if warmup > 0 and epoch < warmup:
        lr = lr / (warmup - epoch)
    else:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup) / (final_epoch - warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def attention_distillation_loss(attention_map1, attention_map2):
    """Calculates the attention distillation loss"""
    attn_difference = attention_map1 - attention_map2
    att_norm_each_head = torch.linalg.norm(attn_difference, dim=(2, 3), ord="fro")
    return F.normalize(att_norm_each_head, p=2., dim=1).mean(dim=1).mean()

class Trainer:
    def __init__(self, epochs, learning_rate, batch_size,optimizer,task_size, num_class, distill=True,dataset_name="cifar100"):
        super(Trainer, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = get_model('inc_cct_b')
        self.num_class = num_class
        self.train_transform =  get_train_transforms(dataset_name)
        self.test_transform = get_test_transforms(dataset_name)
        self.train_dataset = get_train_dataset(dataset_name,self.train_transform)
        self.test_dataset = get_test_dataset(dataset_name,self.test_transform)
        self.batch_size = batch_size
        self.task_size = task_size
        self.train_loader = None
        self.test_loader = None
        self.opt = None
        self.exemplar_set = []
        self.memory_size = 2000
        self.old_model = None
        self.distill = distill

    def beforeTrain(self):
        self.model.eval()
        classes = [self.num_class - self.task_size, self.num_class]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.num_class > self.task_size:
            self.old_model = copy.deepcopy(self.model)
            self.model.incremental_learning(self.num_class)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, self.exemplar_set)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=False,
                                 batch_size=self.batch_size)

        return train_loader, test_loader

    def train(self, resume=False, task_id=1):
        if resume:
            print("Loading from previous state dict")
            directory = './checkpoint'
            filename = directory + f'/task_id_{task_id}.pt'
            checkpt = load_checkpoint(filename)
            self.model.load_state_dict(checkpt['net'])
            return
        optimizer_fn = get_optimizer("radam")
        opt = optimizer_fn(self.model.parameters(), lr=self.learning_rate, weight_decay=3e-2)
        for epoch in range(1, self.epochs + 1):
            adjust_learning_rate(opt, epoch,self.learning_rate,self.epochs,5)
            total_loss = 0.
            total_images = 0
            self.model.train()

            for step, (indexs, images, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                                       desc='Training',position=0, leave=True):
                images, target = images.to(device), target.to(device)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                total_loss += loss_value.item()
                total_images += images.size(0)
            training_accuracy = self._test(self.train_loader)
            accuracy = self._test(self.test_loader)
            avg_loss = total_loss / total_images if total_images != 0 else 1000
            wandb.log({
                "epoch": epoch,
                "training_avg_loss": avg_loss,
                "test_accuracy": accuracy,
                "training_accuracy": training_accuracy,
                "learning_rate":opt.param_groups[0]['lr']
            })
            print('epoch:%d,accuracy:%.3f,training_accuracy:%.3f' % (epoch, accuracy, training_accuracy))
        self.opt = opt

    def _test(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        for step, (index, imgs, labels) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader),leave=True,position=0):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy

    def _compute_loss(self, index, imgs, target):
        output,attn = self.model(imgs,require_attention=True)
        output, target,attn = output.to(device), target.to(device),attn.to(device)
        distillation_loss = 0
        attention_loss = 0
        beta = 0.5
        alpha = (self.num_class - self.task_size) / self.num_class
        if self.old_model is not None:
            with torch.no_grad():
                output_hat,old_attn = self.old_model(imgs,require_attention=True)
                pi_hat = F.softmax(output_hat, dim=1)
            log_pi = F.log_softmax(output[:, :self.num_class - self.task_size], dim=1)
            distillation_loss = torch.mean(torch.sum(-pi_hat * log_pi, dim=1))
            selected_classes = (target<(self.num_class-self.task_size))
            attention_loss = attention_distillation_loss(attn[selected_classes],old_attn[selected_classes])
            wandb.log({
                "attention_loss":attention_loss,
                "num_class":self.num_class,
                "selected_classes":len(selected_classes)
            })
        classification_loss = F.cross_entropy(output, target.to(torch.long))
        return distillation_loss * alpha + classification_loss * (1 - alpha) + beta * attention_loss

    def afterTrain(self, task_id, no_save=False):
        self.model.eval()
        m = int(self.memory_size / self.num_class)
        self._reduce_exemplar_sets(m)
        for i in range(self.num_class - self.task_size, self.num_class):
            print('construct class %s examplar:' % (i), end='')
            images = self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images, m)
        self.num_class += self.task_size
        self.model.train()
        if no_save:
            return
        state = {
            'net': self.model.state_dict(),
            'task_id': task_id,
            'optim': self.opt.state_dict() if self.opt is not None else '',
        }
        directory = './checkpoint'
        filename = directory + f'/task_id_{task_id}.pt'
        ensure_dir(directory)
        save_checkpoint(state, filename)

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))

    def _construct_exemplar_set(self, images, m):
        exemplar = []

        for i in range(m):
            exemplar.append(images[i])

        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)

