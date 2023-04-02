import os
import glob
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

# TODO: FIX paths


class CustomDataset(Dataset):
    def __init__(self, path, classes, img_dim=(32, 32), transform=None):
        self.transform = transform
        self.imgs_path = path
        file_list = glob.glob(os.path.join(self.imgs_path, "*"))

        self.data = []

        for class_path in file_list:
            class_path = os.path.join(class_path)
            class_path = os.path.join(class_path.replace('\\', '/'))
            class_name = class_path.split('/')[-1]
            #class_name = class_path.split("\\")[-1]
            #class_name = class_path.split("\\")[-1]
            x = os.path.join(class_path, "/*.jpg")
            p = glob.glob(os.path.join(class_path) + "/*.jpg")
            for img_path in p:
                # for img_path in glob.glob(class_path + "\\*.jpg"):
                img_path = img_path.replace('\\', '/')
                self.data.append([os.path.join(img_path), class_name])

        self.create_class_map(classes)
        self.img_dim = img_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)

        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.numpy()
        img_tensor = np.transpose(img_tensor, (1, 2, 0))

        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, class_id

    def create_class_map(self, classes):
        class_map = {}
        for idx, cls in enumerate(classes):
            class_map[cls] = idx

        self.class_map = class_map
