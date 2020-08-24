import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
class MNIST_Model(nn.Module):
    def __init__(self):
        super(MNIST_Model, self).__init__()  
        self.conv1 = nn.Conv2d(1, 32, 3,)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.max_pooling1=torch.nn.MaxPool2d(2,2)
        self.max_pooling2=torch.nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(4*4*64, 200) 
        self.fc2 = nn.Linear(200, 10) # 10分类
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pooling1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pooling2(x)
        
        x = x.permute((0, 2, 3, 1))
        x = x.contiguous().view(-1, 4 * 4 * 64)
        x = x.view(-1, 4 * 4 * 64)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.relu(x)
#         x = F.log_softmax(x,dim=1)
        return x

from torchvision import models
model = models.alexnet(pretrained=True)