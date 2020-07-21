# -*- coding: utf-8 -*-


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

"""### Upload Dataset:"""

# from google.colab import files
# uploaded = files.upload()
# !unzip data.zip

"""### Custom Dataset Class:"""

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

"""### Load Data and Apply Transformations:"""

#Transformations: In the training phase more complicated transformations are needed (change if results are not consistent)
transform = transforms.Compose([
                                      transforms.RandomCrop(32, padding=4), # remove this if results are less consistent
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                     ]) #normalize each channel =>image = (image - mean) / std

print('Loading Data ...')
dataset = CustomDataset("./data/train/x.npy", "./data/train/y.npy",transform=transform)

print('------------------------------')
print('Counting Classes ...')
classes=set(dataset.targets.tolist())
print(len(classes),'classes:',classes)

print('------------------------------')
print('Finding Distribution ...')

distribution= [0] * len(classes) #distribution is class count
for data in dataset: # data is a tuple if image and target
  distribution[data[1]]+=1

for i,value in enumerate(distribution):
  print('class',i,value)

"""### Stratified Splitting the Data

1.   80% Training
2.   20% Validation
"""

print('------------------------------')
print('Stratified Splitting Data to Validation and Train ...')
targets=dataset.targets.numpy() # all targets in the dataset
train_indices, valid_indices = train_test_split(np.arange(len(targets)) , test_size=0.2, stratify=targets) #split and stratify the data: keep the proportions

# Creating two new datasets to apply weights individually
train_set=torch.utils.data.Subset(dataset,train_indices)
valid_set=torch.utils.data.Subset(dataset,valid_indices)



# Preprocessing training set: find targets, distribution (class count), weight of each class, apply the weights to the samples
print("Train Size:",len(train_set))

train_targets=[]
for data in train_set: #every data is a tuple of an image/tensor and a target
  train_targets.append(data[1])
print("Targets:", train_targets)

train_distribution= [0] * len(classes) #class sample count
for data in train_set:
  train_distribution[data[1]]+=1
print("Distribution:",train_distribution)

train_class_weights = 1./torch.tensor(train_distribution, dtype=torch.float) # find the weight for each class
print("Class weights:",train_class_weights)

train_class_weights_all = train_class_weights[train_targets]
print("Weights applied:",train_class_weights_all)

print('------------------------------')

# Preprocessing validation set: find targets, distribution (class count), weight of each class, apply the weights to the samples
print("Valid Size:",len(valid_set))

valid_targets=[]
for data in valid_set: #every data is a tuple of an image/tensor and a target
  valid_targets.append(data[1])
print("Targets:", valid_targets)

valid_distribution= [0] * len(classes) #class sample count
for data in valid_set:
  valid_distribution[data[1]]+=1
print("Distribution:",valid_distribution)

valid_class_weights = 1./torch.tensor(valid_distribution, dtype=torch.float) # find the weight for each class
print("Class weights:",valid_class_weights)

valid_class_weights_all = valid_class_weights[valid_targets]
print("Weights applied:",valid_class_weights_all)

"""### Creating DataLoaders with Weighted Elements:
Weighted sampler is used for handling imbalanced classes
"""

train_weighted_sampler = WeightedRandomSampler(
    weights=train_class_weights_all,
    num_samples=len(train_class_weights_all),
    replacement=True
)

valid_weighted_sampler= WeightedRandomSampler(
    weights=valid_class_weights_all,
    num_samples=len(valid_class_weights_all),
    replacement=True
)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=128,sampler=train_weighted_sampler)#, num_workers=2)

validloader = torch.utils.data.DataLoader(valid_set, batch_size=128,sampler=valid_weighted_sampler)#, num_workers=2)

"""###This cell is only for checking the visuals for one picture (You can skip this):"""

# for data in trainloader:
#   print(data[0].shape)  # batch_size, # channels, #height, #width
#   break

# # show images
# plt.imshow(np.transpose(data[0][0], (1, 2, 0))) #replace 0 with 1 axis and 1 with 2 and 2 with 0  -> output: height,width ,channel
# plt.show
# print(data[1][0])

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

    self.fc=nn.Linear(640, len(classes))

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

"""### Cross Entropy Loss with Label Smoothoing:
Smoothing Factor=0.2
"""

# log softmax loss
class SoftMaxSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.2):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = len(classes)

    def forward(self, predictions, target):
        predictions = F.log_softmax(predictions,dim=-1) # apply log softmax

        with torch.no_grad():
            smoothed_labels = torch.zeros_like(predictions) # create an empty tensor filled with zero

            smoothed_labels.fill_(self.smoothing / (self.classes - 1)) # fill all elements with with the smoothing factor/classes -1

            smoothed_labels.scatter_(1, target.data.unsqueeze(1), self.confidence) # write confidence to the index of target

        return torch.mean(torch.sum(-smoothed_labels * predictions, dim=-1)) # predictions are like the actual log softmax probabilities (negative)* weight(labels) (negativee) -> positive

"""### Driver Code:"""

# Select GPU if available
total_epoch=150
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print("This model is running on" , torch.cuda.get_device_name())

#Model
net=CustomRESNet().to(device)


#Get adjustable parameters(weights) and optimize them
optimizer=optim.SGD(net.parameters(),lr=0.1,weight_decay=0.0005,momentum=0.9,nesterov=True) #weight decay is multiplied to weight to prevent them from growing too large
#optimizer=optim.Adam(net.parameters(),lr=0.1,weight_decay=0.0005) #weight decay is multiplied to weight to prevent them from growing too large

#Error Function
criterion = SoftMaxSmoothingLoss()

# Learning rate scheduler: adjusts learning rate as the epoch increases
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60,gamma=0.2) #Decays the learning rate by multiplyin by gamma every step_size epochs

#How many times we pass our full data (the same data)

torch.cuda.empty_cache()

"""### Training and Validation:"""

best_valid_acc=0

for cur_epoch in range(total_epoch):
  train_correct=0
  train_total=0
  train_loss=0 #loss per epoch

  valid_correct=0
  valid_total=0
  valid_loss=0 #loss per epoch

  net.train() #put the model in training mode
  net.training=True
  for data in trainloader:

    #every data consits of (batch_size) images
    X,y=data[0].to(device), data[1].to(device) #picture(X batch_size), label(X batch_size) -> #batch size comes first #note that the label here is a number which is index in labels list

    net.zero_grad()
    output = net(X)
    loss = criterion(output, y) #calculate the error/ loss for the that batch (data)

    loss.backward()  #computes dloss/dw for every parameter w  (loss for every parameter)
    optimizer.step() #update weights

    train_loss+=loss.item()

    #calculate how many right do you have in every training data until the end of all training datas
    #output is Batch_size*10 tensor
    for k, i in enumerate(output): # the output is batch_size* 10 tensor   # k is the index of the data # i the data itself
        if torch.argmax(i) == y[k]: # in every row find the highest prediction index and compare it to y[k]
                train_correct += 1
        train_total += 1

  exp_lr_scheduler.step() #learning rate adjustment

  net.eval() #put the model in evaluation mode
  net.training=False
  #validate for each epoch
  with torch.no_grad(): # no gradient
    for data in validloader:
      X, y = data[0].to(device), data[1].to(device) # store the images in X and labels in y
      output = net(X)
      loss = criterion(output, y)

      valid_loss += loss.item()

      for k, i in enumerate(output): # the output is batch_size* 10 ARRAY
          if torch.argmax(i) == y[k]: # in every row find the highest prediction and comprae its index
              valid_correct += 1
          valid_total += 1

  #if the model is better than the previous best store it
  if((valid_correct/valid_total)>best_valid_acc):
    best_valid_acc= (valid_correct/valid_total)
    torch.save(net.state_dict(), "./model_2017314461.pth") #save early stopping point

  #if((cur_epoch+1)%(total_epoch*0.1)==0):
  print(' Epoch {}/{}: Training Accuracy {} |  Training Loss {} || Validation Accuracy {} |  Validation Loss {}'.format(cur_epoch+1, total_epoch, train_correct/train_total,train_loss/len(trainloader),valid_correct/valid_total,valid_loss/len(validloader))) #accuray for each epoch
  print(' Best validation so far {}'.format(best_valid_acc))
  print('-------------------------------------------------------------------------------------------------------------------------------')

# # 150 epochs till now

# files.download('model_2017314461.pth')
