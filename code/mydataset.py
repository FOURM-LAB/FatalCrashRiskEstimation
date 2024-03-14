import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import utilities as UT


class MyDataset_MultiScale_PreTrain(Dataset):
    def __init__(self, 
                 data_path="../data", 
                 pos_folder_17="multi-level-overhead/Bing_Map_Positive_17", 
                 neg_folder_17="multi-level-overhead/Bing_Map_Negative_17",
                 pos_folder_18="CentralTX/Bing_Map_Tiles_Positive", 
                 neg_folder_18="CentralTX/Bing_Map_Tiles_Negative",
                 pos_folder_19="multi-level-overhead/Bing_Map_Positive_19", 
                 neg_folder_19="multi-level-overhead/Bing_Map_Negative_19",
                 metadata="../metadata/train.csv",
                 basic_transform=None,
                 aug_transform=None):
        
        self.data_path = data_path
        self.pos_folder_17 = pos_folder_17
        self.neg_folder_17 = neg_folder_17
        self.pos_folder_18 = pos_folder_18
        self.neg_folder_18 = neg_folder_18
        self.pos_folder_19 = pos_folder_19
        self.neg_folder_19 = neg_folder_19 
        
        self.metadata = UT.read_csv(metadata)
        
        self.basic_transform = basic_transform
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):        
        # get the file name (e.g., 29.79768281_-97.95545878_0_0.jpeg)
        filename = self.metadata[index][0]
        
        # get the class label (used to determime from which folder to load the image)
        label = filename.split("_")[-1][0]
        label = int(label)
        
        if(label>1):
            label = 1
        
        # get the image path
        if(label==0):
            img_path_17 = os.path.join(self.data_path, self.neg_folder_17, filename)
            img_path_18 = os.path.join(self.data_path, self.neg_folder_18, filename)
            img_path_19 = os.path.join(self.data_path, self.neg_folder_19, filename)
        else:
            img_path_17 = os.path.join(self.data_path, self.pos_folder_17, filename)
            img_path_18 = os.path.join(self.data_path, self.pos_folder_18, filename)
            img_path_19 = os.path.join(self.data_path, self.pos_folder_19, filename)
        
        # Random select two images to for the pair
        img_path_list = [img_path_17, img_path_18, img_path_19]
        np.random.shuffle(img_path_list) # random shuffle the list
        
        # read images 
        img_original_1 = Image.open(img_path_list[0])
        img_original_2 = Image.open(img_path_list[1])

        # Normalize
        img_1 = self.basic_transform(img_original_1)
        img_2 = self.basic_transform(img_original_2)
       
        # Apply the same Augmentation to all modalities
        # random RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(img_1, 
                                                             scale=(0.3, 1.0), 
                                                             ratio=(0.75, 1.3333))
        # apply crop (after this step, the image size will be smaller than the input size)
        img_output_1 = F.crop(img_original_1, i, j, h, w)
        img_output_2 = F.crop(img_original_2, i, j, h, w)

        # random horizontal flip 
        if torch.rand(1) < 0.5:
            img_output_1 = F.hflip(img_output_1)
            img_output_2 = F.hflip(img_output_2)

        # random verticalflip 
        if torch.rand(1) < 0.5:
            img_output_1 = F.vflip(img_output_1)
            img_output_2 = F.vflip(img_output_2)

        # random rotation
        angle = transforms.RandomRotation.get_params([-90, 90])
        img_output_1 = F.rotate(img_output_1, angle)
        img_output_2 = F.rotate(img_output_2, angle)        

        # Apply normalize
        img_output_1 = self.basic_transform(img_output_1)
        img_output_2 = self.basic_transform(img_output_2)
        
        output = [[img_1, img_2], img_output_1, img_output_2, img_path_list, label]

        return output   
    
    
class MyDataset_MultiScale_SameAug(Dataset):
    def __init__(self, 
                 data_path="../data", 
                 pos_folder_17="multi-level-overhead/Bing_Map_Positive_17", 
                 neg_folder_17="multi-level-overhead/Bing_Map_Negative_17",
                 pos_folder_18="CentralTX/Bing_Map_Tiles_Positive", 
                 neg_folder_18="CentralTX/Bing_Map_Tiles_Negative",
                 pos_folder_19="multi-level-overhead/Bing_Map_Positive_19", 
                 neg_folder_19="multi-level-overhead/Bing_Map_Negative_19",
                 metadata="../metadata/train.csv",
                 if_test=False,
                 basic_transform=None):
        
        self.data_path = data_path
        self.pos_folder_17 = pos_folder_17
        self.neg_folder_17 = neg_folder_17
        self.pos_folder_18 = pos_folder_18
        self.neg_folder_18 = neg_folder_18
        self.pos_folder_19 = pos_folder_19
        self.neg_folder_19 = neg_folder_19 
        
        self.metadata = UT.read_csv(metadata)

        self.if_test = if_test
        self.basic_transform = basic_transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):        
        # get the file name (e.g., 29.79768281_-97.95545878_0_0.jpeg)
        filename = self.metadata[index][0]
        
        # get the class label (used to determime from which folder to load the image)
        label = filename.split("_")[-1][0]
        label = int(label)
        
        if(label>1):
            label = 1
        
        # get the image path
        if(label==0):
            img_path_17 = os.path.join(self.data_path, self.neg_folder_17, filename)
            img_path_18 = os.path.join(self.data_path, self.neg_folder_18, filename)
            img_path_19 = os.path.join(self.data_path, self.neg_folder_19, filename)
        else:
            img_path_17 = os.path.join(self.data_path, self.pos_folder_17, filename)
            img_path_18 = os.path.join(self.data_path, self.pos_folder_18, filename)
            img_path_19 = os.path.join(self.data_path, self.pos_folder_19, filename)

        img_path_list = [img_path_17, img_path_18, img_path_19]
                            
        # read images 
        img_original_17 = Image.open(img_path_17)
        img_original_18 = Image.open(img_path_18)
        img_original_19 = Image.open(img_path_19)

        # Normalize
        img_17 = self.basic_transform(img_original_17)
        img_18 = self.basic_transform(img_original_18)
        img_19 = self.basic_transform(img_original_19)
               

        # Apply the same Augmentation to all modalities
        # get data augmentation parameters and apply the same parameter to both of the modalities
        if(not self.if_test):
            # random RandomResizedCrop
            i, j, h, w = transforms.RandomResizedCrop.get_params(img_original_17, 
                                                                 scale=(0.3, 1.0), 
                                                                 ratio=(0.75, 1.3333))
            # apply crop (after this step, the image size will be smaller than the input size)
            img_output_17 = F.crop(img_original_17, i, j, h, w)
            img_output_18 = F.crop(img_original_18, i, j, h, w)
            img_output_19 = F.crop(img_original_19, i, j, h, w)
            
            # random horizontal flip 
            if torch.rand(1) < 0.5:
                img_output_17 = F.hflip(img_output_17)
                img_output_18 = F.hflip(img_output_18)
                img_output_19 = F.hflip(img_output_19)
                
            # random verticalflip 
            if torch.rand(1) < 0.5:
                img_output_17 = F.vflip(img_output_17)
                img_output_18 = F.vflip(img_output_18)
                img_output_19 = F.vflip(img_output_19)   
                
            # random rotation
            angle = transforms.RandomRotation.get_params([-90, 90])
            img_output_17 = F.rotate(img_output_17, angle)
            img_output_18 = F.rotate(img_output_18, angle)
            img_output_19 = F.rotate(img_output_19, angle)

            # Apply normalize
            img_output_17 = self.basic_transform(img_output_17)
            img_output_18 = self.basic_transform(img_output_18)
            img_output_19 = self.basic_transform(img_output_19)
        
        if(self.if_test):
            output = [[img_17, img_18, img_19], img_path_list, label]
        else:      
            output = [[img_output_17, img_output_18, img_output_19], img_path_list, label]

        return output      