from .ICIFAR100 import ICIFAR100
from torchvision import transforms
from augmentations import CIFAR10Policy
from augmentations import ImageNetPolicy
from .ITinyImageNet import ITinyImageNet

def get_train_transforms(data_set_name: str):
    if data_set_name == "cifar100":
        return transforms.Compose([
            CIFAR10Policy(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    elif data_set_name == "tinyimagenet":
        return transforms.Compose([
            ImageNetPolicy(),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
        ])
    else:
        raise NotImplementedError("Not implemented")


def get_test_transforms(data_set_name: str):
    if data_set_name == "cifar100":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    elif data_set_name == "tinyimagenet":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
        ])
    else:
        raise NotImplementedError("Not implemented")


def get_train_dataset(data_set_name: str, train_transform):
    if data_set_name == "cifar100":
        return ICIFAR100('data', transform=train_transform, download=True)
    elif data_set_name == "tinyimagenet":
        return ITinyImageNet('data',transform=train_transform)
    else:
        raise NotImplementedError("Not implemented")


def get_test_dataset(data_set_name: str, test_transform):
    if data_set_name == "cifar100":
        return ICIFAR100('data', test_transform=test_transform, train=False, download=True)
    elif data_set_name == "tinyimagenet":
        return ITinyImageNet('data',train=False,transform=test_transform)
    else:
        raise NotImplementedError("Not implementd")
