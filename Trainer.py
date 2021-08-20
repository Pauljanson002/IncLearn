import math

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import ICIFAR100
from models import get_model
from torchvision import transforms
from augmentations import CIFAR10Policy
from torch import optim
from torch.nn import functional as F
from utils import save_checkpoint, ensure_dir, load_checkpoint

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


class Trainer:
    def __init__(self, epochs, learning_rate, batch_size, task_size, num_class):
        super(Trainer, self).__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = get_model('inc_vit_b')
        self.num_class = num_class
        self.train_transform = transforms.Compose([
            CIFAR10Policy(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_dataset = ICIFAR100('data', transform=self.train_transform, download=True)
        self.test_dataset = ICIFAR100('data', test_transform=self.test_transform, train=False, download=True)
        self.batch_size = batch_size
        self.task_size = task_size
        self.train_loader = None
        self.test_loader = None
        self.opt = None

    def beforeTrain(self):
        self.model.eval()
        classes = [self.num_class - self.task_size, self.num_class]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if self.num_class > self.task_size:
            self.model.incremental_learning(self.num_class)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.batch_size)

        return train_loader, test_loader

    def train(self,resume=False,task_id=1):
        if resume:
            print("Loading from previous state dict")
            directory = './checkpoint'
            filename = directory + f'/task_id_{task_id}.pt'
            self.model.load_state_dict(load_checkpoint(filename)['net'])
            return
        opt = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=3e-2)
        for epoch in range(1, self.epochs + 1):
            adjust_learning_rate(opt, epoch, self.learning_rate, self.epochs, 5)
            total_loss = 0.
            total_images = 0
            for step, (indexs, images, target) in tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                                                       desc='Training'):
                images, target = images.to(device), target.to(device)
                loss_value = self._compute_loss(indexs, images, target)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                total_loss += loss_value.item()
                total_images += images.size(0)
            accuracy = self._test(self.test_loader)
            avg_loss = total_loss / total_images if total_images != 0 else 1000
            wandb.log({
                "epoch": epoch,
                "training_avg_loss": avg_loss,
                "test_accuracy": accuracy
            })
            print('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        self.opt = opt

    def _test(self, test_loader):
        self.model.eval()
        correct, total = 0, 0
        for step, (index, imgs, labels) in tqdm(enumerate(test_loader), desc="Testing", total=len(test_loader)):
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
        output = self.model(imgs)
        target = get_one_hot(target, self.num_class)
        output, target = output.to(device), target.to(device)
        return F.binary_cross_entropy_with_logits(output, target)

    def afterTrain(self, task_id,no_save=False):
        self.model.eval()
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
