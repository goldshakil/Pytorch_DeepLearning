import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import random
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# get the data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
class CustomDataset(Dataset):
    def __init__(self, x_path, y_path, transform=None):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform

        self.data = np.transpose(np.load(self.x_path), (0, 3, 1, 2))
        self.targets = torch.LongTensor(np.load(self.y_path))

        print(self.data.shape)
        print(self.targets.shape)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y


#Transformations: In the training phase more complicated transformations are needed (change if results are not consistent)
transform = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]) #normalize each channel =>image = (image - mean) / std

print('Loading Data ...')
testset = CustomDataset("./data/test/x.npy", "./data/test/y.npy",transform=transform)#change the path for your data since this is just fake

test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)#, num_workers=2)


"""### Basic Block Definition (BottleNeck Block is not needed in ResNet-26):"""

class BasicBlock(nn.Module):
  #Initialize your BasicBlock strucutre
  def __init__(self,in_channels,out_channels,stride=1):
    super().__init__()

    # Remember:
        #Conv2d Parameters: input channels,output channels, size_kernel
        #BatchNorm Goals: Normalize the outputs, parameters:(#output_channels/ node)

    #First Convolution: changes the dimensions of image according to the given parameters
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1) #change channels   #reduce size to half if stride is set to two


    #Second Convolution: doesn't change the dimensions at all (ignore it in calculation)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    #SE Block not used
    #self.SE_Block = SEBlock(out_channels)

    #This changes the input size to match the output size: #change channels   #reduce size to half if stride is set to two
    self.input_changer = nn.Sequential()
    if stride != 1  or in_channels != out_channels:
      self.input_changer = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride))

  def forward(self, x):
    #x is original data
    output=x
    #Pass through the first convolution
    output= self.conv1(F.relu(self.bn1(output)))  #conv -> normalize -> activate/threshold
    output=F.dropout(output,p=0.3,training=self.training)
    #Pass through the first convolution
    output= self.conv2(F.relu(self.bn2(output)))
    #Multiply by SE_Block Weights
    #output=self.SE_Block(output)
    #Add the input
    output+=self.input_changer(x)
    return output

"""### ResNet-26 Neural Network Definition:"""

class CustomRESNet(nn.Module):
  #initialize your network strucutre
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1) #3,32,32 -> 16,32,32


    # widening factor is 10
    #depth is 28 -> 4 blocks per convoultion

    #First Block: Change channels   #Other Blocks: Don't change anything  #Last Block: Change dimesnsions using stride 2
    self.conv2=nn.Sequential(BasicBlock(16,160,1),BasicBlock(160,160,1),BasicBlock(160,160,1),BasicBlock(160,160,1)) #16,32,32 -> 160,32,32 ->160,32,32

    self.conv3=nn.Sequential(BasicBlock(160,320,1),BasicBlock(320,320,1),BasicBlock(320,320,1),BasicBlock(320,320,2)) #160,32,32 ->320,32,32 ->320,16,16

    self.conv4=nn.Sequential(BasicBlock(320,640,1),BasicBlock(640,640,1),BasicBlock(640,640,1),BasicBlock(640,640,2)) #320,16,16 ->640,16,16 ->640,8,8

    # self.conv5=nn.Sequential(BasicBlock(256,512,1),BasicBlock(512,512,1))

    self.bn1 = nn.BatchNorm2d(640) #wait?

    self.pool = nn.AvgPool2d(8) #640,8,8 -> 640,1,1

    self.fc=nn.Linear(640, 91)

    #self.dropout = nn.Dropout(p=0.3)
    #Initialiliation of weights in the model
    for element in self.modules():
        if isinstance(element, nn.Conv2d):
          nn.init.kaiming_normal_(element.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(element, nn.BatchNorm2d):
          element.weight.data.fill_(1)
          element.bias.data.zero_()
        elif isinstance(element, nn.Linear):
          element.bias.data.zero_()

  def forward(self, x):
    x=self.conv1(x) #conv -> normalize -> activate/threshold

    #BasicBlocks
    x=self.conv2(x)
    x=self.conv3(x)
    x=self.conv4(x)
    # x=self.conv5(x)
    x=F.relu(self.bn1(x))

    x=self.pool(x)

    #flatten
    x=x.view(-1, 640)
    x=self.fc(x)

    #x=F.dropout(x,p=0.3,training=self.training)
    return x
"""### Driver Code:"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("This model is running on" , torch.cuda.get_device_name())

#Model
load_model=CustomRESNet().to(device)

load_model.load_state_dict(torch.load("./model_2017314461.pth"))

#training

load_model.eval()
load_model.training=False

correct =0
total=0
with torch.no_grad(): # no gradient
  for data in test_loader:
      X, y = data[0].to(device), data[1].to(device) # store the images in X and labels in y
      output = load_model(X) #send the 4 images
      #print(output)
      for k, i in enumerate(output): # the output is 4* 10 ARRAY
          if torch.argmax(i) == y[k]: # in every row find the highest prediction and comprae its index
              correct += 1
          total += 1

print("Test Accuracy: ", correct/total)
