import torch
from models import get_model
from torch import optim
from augmentations import CIFAR10Policy
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm
from Trainer import adjust_learning_rate
from torch.nn import functional as F

inc_cct = get_model("inc_cct_b")
for i in range(2, 11):
    inc_cct.incremental_learning(i * 10)
state = torch.load("./checkpoint/task_id_10.pt")
inc_cct.load_state_dict(state["net"])
inc_cct.to('cuda')
opt = optim.AdamW(inc_cct.parameters(), 0.0005, weight_decay=3e-2)
opt.load_state_dict(state["optim"])

train_transform = transforms.Compose([
    CIFAR10Policy(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

train_dataset = CIFAR100("./data", True, transform=train_transform, download=False)
test_dataset = CIFAR100("./data", False, transform=test_transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

inc_cct.requires_grad_(False)
inc_cct.classification_head.linear.requires_grad_(True)

device = torch.device('cuda')
inc_cct.to(device)

for epoch in range(1, 2):
    adjust_learning_rate(opt, epoch, 0.0005, 100, 5)
    total_loss = 0.
    total_images = 0.
    # for step, (images, target) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
    #     images, target = images.to(device), target.to(device)
    #     loss_value = F.cross_entropy(inc_cct(images), target)
    #     opt.zero_grad()
    #     loss_value.backward()
    #     opt.step()
    #     total_loss += loss_value.item()
    #     total_images += images.size(0)
    correct, total = 0, 0
    for step, (images, target) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
        images, target = images.to(device), target.to(device)
        with torch.no_grad():
            outputs = inc_cct(images)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts == target).sum()
        total += len(target)
    accuracy = (100 * correct / total)
    print(f"Accuracy at {epoch} = {accuracy}")
