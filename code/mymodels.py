import torch 
import torch.nn as nn
from torchvision import models
import numpy as np

class ResNet50_with_Feature(nn.Module):
    def __init__(self, feature_shape=(2048,24,24), 
                 pretrained=True, requires_grad=True,
                 num_classes=2):
        super(ResNet50_with_Feature, self).__init__()
        self.ft_img = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.ft_img_modules = list(self.ft_img.children())[:-1]
        self.ft_img = nn.Sequential(*self.ft_img_modules)
        for p in self.ft_img.parameters():
            p.requires_grad = requires_grad

        # ConvLayer for image
        self.conv_img = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0], # input height
                out_channels=feature_shape[0], # n_filters
                kernel_size=1, # filter size
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )
        self.gap_img = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        x = self.ft_img(x)        
        x = self.conv_img(x)           
        x = self.gap_img(x)
        ft = x.view(x.size(0), -1)
        x = self.fc(ft)
        return x, ft
    

class ResNet50_MultiScale_MultiHead_V2(nn.Module):
    def __init__(self, feature_shape=(2048,24,24), modality=3,
                 requires_grad=True, num_classes=2, pretrain_weights=None):
        super(ResNet50_MultiScale_MultiHead_V2, self).__init__()
        
        self.feature_shape = feature_shape
        self.modality = modality
        
        # Load the feature extraction model
        self.ft_img = ResNet50_with_Feature()
        if(not (pretrain_weights is None)):
            self.ft_img.load_state_dict(pretrain_weights)        
        self.ft_img_modules = list(self.ft_img.children())[0][:-1]  
        self.ft_img = nn.Sequential(*self.ft_img_modules)
        for p in self.ft_img.parameters():
            p.requires_grad = requires_grad
            
        
        # Combine the features of multiple modalities together on the Channel dimension
        # then apply 1x1 Conv and GAP            
        # ConvLayer for image
        self.conv_img = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0]*modality, # input height
                out_channels=feature_shape[0]*modality, # n_filters
                kernel_size=1, # filter size
            ),
            nn.BatchNorm2d(feature_shape[0]*modality),
            nn.ReLU(),
        )
        
        self.gap_img = nn.AdaptiveAvgPool2d((1,1))        
        self.fc = nn.Linear(feature_shape[0]*modality, num_classes)


        # single modality 1
        self.conv_img_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0], # input height
                out_channels=feature_shape[0], # n_filters
                kernel_size=1, # filter size
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )        
        self.gap_img_1 = nn.AdaptiveAvgPool2d((1,1))        
        self.fc_1 = nn.Linear(feature_shape[0], num_classes)

        # single modality 2
        self.conv_img_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0], # input height
                out_channels=feature_shape[0], # n_filters
                kernel_size=1, # filter size
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )        
        self.gap_img_2 = nn.AdaptiveAvgPool2d((1,1))        
        self.fc_2 = nn.Linear(feature_shape[0], num_classes)

        # single modality 3
        self.conv_img_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=feature_shape[0], # input height
                out_channels=feature_shape[0], # n_filters
                kernel_size=1, # filter size
            ),
            nn.BatchNorm2d(feature_shape[0]),
            nn.ReLU(),
        )        
        self.gap_img_3 = nn.AdaptiveAvgPool2d((1,1))        
        self.fc_3 = nn.Linear(feature_shape[0], num_classes)
        

    def forward(self, x):
        # Pass the three images from the feature extractor
        ft1 = self.ft_img(x[0]) 
        ft2 = self.ft_img(x[1]) 
        ft3 = self.ft_img(x[2]) 
        
        # Concatenate the three images on the Modality dimension
        ft = torch.cat((ft1, ft2, ft3), dim=1)                                        
        ft = self.conv_img(ft)          
        ft = self.gap_img(ft)
        ft = ft.view(ft.size(0), -1)
        logit = self.fc(ft)

        # Single modality
        ft1 = self.conv_img_1(ft1) 
        ft2 = self.conv_img_2(ft2) 
        ft3 = self.conv_img_3(ft3) 

        ft1 = self.gap_img_1(ft1)
        ft2 = self.gap_img_2(ft2)
        ft3 = self.gap_img_3(ft3)

        ft1 = ft1.view(ft1.size(0), -1)
        ft2 = ft2.view(ft2.size(0), -1)
        ft3 = ft3.view(ft3.size(0), -1)

        logit1 = self.fc_1(ft1)
        logit2 = self.fc_2(ft2)
        logit3 = self.fc_3(ft3)

        return logit, logit1, logit2, logit3, ft, ft1, ft2, ft3        