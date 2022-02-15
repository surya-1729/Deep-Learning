import torch
import torch.nn as nn
import math
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F


##
#torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -F 
##

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        

        self.efficientnet =models.efficientnet_b0(pretrained=True)
        
        self.efficientnet.classifier[1] = nn.Linear(in_features=1280, out_features=8)
    def forward(self, x):
        
      
        x = self.efficientnet(x)
       
        
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), './checkpoints/model.pth')

