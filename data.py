from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import pathlib
from torchvision import transforms
import torch

class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
       
        self.path = path
        self.input_size = 224
        self.label_list = sorted(os.listdir(self.path))
        
        self.images_list = []
       
        for abspath,_,files in os.walk(self.path):
            for name in files :
                self.images_list.append(os.path.join(abspath, name))
                #print(pathlib.PurePath(abspath, name))
        if self.training == False:
            self.images_list =sorted(self.images_list, key=lambda i: int(i.split("/")[-1].split(".")[0]))
        #print(self.images_list)
    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model

        image = cv2.imread(self.images_list[index])
        #print(self.images_list[index])
       
        image = cv2.resize(image,(self.input_size,self.input_size))
        if self.training ==True :
            image = preprocess(image, self.input_size, True)
        else:
            image = preprocess(image, self.input_size, False)
        image_name = self.images_list[index].split("/")[-1].split(".")[0]
        image.reshape(3,-1).float()
        if self.training == False:
            return (image,int(image_name))
        
        label_name = self.images_list[index].split("/")[-2]
        label = self.label_list.index(label_name)
        
        return image,label
            
def preprocess(image, input_size, augmentation=True):
    if augmentation:
        crop_transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    else:
        crop_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            ])
    image = crop_transform(image)
    result = transforms.Compose([
        
        transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225])
    ])(image)
    
    return result      


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# a =ChristmasImages("dataset/data/train")
# print(get_mean_and_std(a))