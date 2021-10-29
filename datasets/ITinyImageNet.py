from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import numpy as np
import glob
from PIL import Image
class ITinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.transform = transform
        self.test_cls_dict = {}
        for i, line in enumerate(open('data/tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            if cls_id not in self.test_cls_dict.keys():
                self.test_cls_dict[cls_id] = [img]
            else:
                self.test_cls_dict[cls_id].append(img)
        self.id_dict_reverse = {}
        for i, line in enumerate(open('data/tiny-imagenet-200/wnids.txt', 'r')):
            self.id_dict_reverse[i] = line.replace('\n', '')

    def concatenate(self, data, labels):
        con_data = data[0]
        con_label = labels[0]
        for i in range(1, len(data)):
            con_data = np.concatenate((con_data, data[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTrainData(self, classes, exemplar_set):
        datas, labels = [], []
        if len(exemplar_set) != 0:
            datas = [exemplar for exemplar in exemplar_set]
            length = len(datas[0])
            labels = [np.full((length), label) for label in range(len(exemplar_set))]
        for label in range(classes[0], classes[1]):
            folder_name = self.id_dict_reverse[label]
            filenames = glob.glob(self.root + '/tiny-imagenet-200/train/' + folder_name + "/*/*")
            data = []
            for img_path in filenames:
                img = np.asarray(Image.open(img_path))
                if img.shape == (64, 64, 3):
                    data.append(img)
            data = np.stack(data)
            datas.append(data)
            labels.append(np.full(len(data), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        print("This size of the train set is " + str(self.TrainData.shape))
        print("The size of the train label is " + str(self.TrainLabels.shape))

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            class_id = self.id_dict_reverse[label]
            img_paths = self.test_cls_dict[class_id]
            data = []
            for img_path in img_paths:
                img = np.asarray(Image.open("data/tiny-imagenet-200/val/images/" + img_path))
                if img.shape == (64, 64, 3):
                    data.append(img)
            data = np.stack(data)
            datas.append(data)
            labels.append(np.full(len(data), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTrainItem(self, index):
        img, label = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        return index, img, label

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.transform:
            img = self.transform(img)

        return index, img, target

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)
        else:
            return None
    def get_image_class(self,label):
        folder_name = self.id_dict_reverse[label]
        filenames = glob.glob(self.root + '/tiny-imagenet-200/train/' + folder_name + "/*/*")
        data = []
        for img_path in filenames:
            img = np.asarray(Image.open(img_path))
            if img.shape == (64, 64, 3):
                data.append(img)
        data = np.stack(data)
        return data


