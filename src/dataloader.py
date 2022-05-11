"""
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to generate Dataset.
"""

import numpy as np
import cv2
import re
from torchvision import transforms
from torch.utils.data import Dataset
from preprocessing import gasuss_noise


class HairNetDataset(Dataset):
    def __init__(self, project_dir, train_flag=1, noise_flag=1):
        """
        param project_dir: the path of project, such as '/home/albertren/Workspace/HairNet/HairNet-ren'
        param train_flag: train_flag=1 -> generate training dataset, train_flag=0 -> generate testing dataset
        """
        self.project_dir = project_dir
        self.train_flag = train_flag
        self.noise_flag = noise_flag
        self.toTensor = transforms.ToTensor()
        train_path = self.project_dir + "/data/index/train.txt"
        test_path = self.project_dir + "/data/index/test.txt"
        self.data_path = self.project_dir + "/data/"
        self.convdata_path = self.project_dir + "/convdata/"
        # generate dataset
        if self.train_flag == 1:
            self.train_index = []
            self.train_index_path = train_path
            with open(self.train_index_path, "r") as f:
                lines = f.readlines()
                for x in lines:
                    self.train_index.append(x.strip().split(" "))
        if self.train_flag == 0:
            self.test_index = []
            self.test_index_path = test_path
            with open(self.test_index_path, "r") as f:
                lines = f.readlines()
                for x in lines:
                    self.test_index.append(x.strip().split(" "))

    def __getitem__(self, index):
        if self.train_flag == 1:
            current_index = self.train_index[index]
        else:
            current_index = self.test_index[index]

        current_convdata_index = re.search(
            "strands\d\d\d\d\d_\d\d\d\d\d_\d\d\d\d\d", str(current_index)
        ).group(0)
        current_convdata_path = (
            self.convdata_path + str(current_convdata_index) + ".convdata"
        )
        current_convdata = np.load(current_convdata_path).reshape(100, 4, 32, 32)
        # current_visweight = gen_vis_weight(
        #     self.data_path + str(current_index[0]) + ".vismap"
        # )
        current_visweight_path = self.data_path + str(current_index[0]) + ".vismap"
        current_visweight = np.load(current_visweight_path)
        current_img = cv2.imread(self.data_path + str(current_index[0]) + ".png")
        print(f'Path: {str(current_index[0]) + ".png"} | Convdata: {str(current_convdata_index) + ".convdata"}')

        if self.noise_flag == 1:
            current_img = gasuss_noise(current_img)

        current_img = self.toTensor(current_img)

        return current_img, current_convdata, current_visweight

    def __len__(self):
        if self.train_flag == 1:
            return len(self.train_index)
        else:
            return len(self.test_index)
