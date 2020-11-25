import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 21

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.conv3 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 26 * 26, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size()[0], 16 * 26 * 26)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1_1 = nn.Conv2d(32, 64, 3)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout3 = nn.Dropout(0.4)
        self.conv4 = nn.Conv2d(32, 16, 3)
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        # x = self.dropout1(x)
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        # x = self.dropout2(x)
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        # x = self.dropout3(x)
        x = self.pool(x)
        x = self.relu(self.bn4(self.conv4(x)))
        # x = self.dropout4(x)
        x = self.pool(x)
        x = x.view(x.size()[0], 16 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# MultiScale VGG Network
class MSVGG16(nn.Module):
    # image size: n, 3, 227, 227
    def __init__(self):
        super(MSVGG16, self).__init__()
        d = 16
        self.conv1 = nn.Conv2d(3, d, 3, padding=1)
        self.conv1_1 = nn.Conv2d(d, d, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(d)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = nn.Conv2d(d, 2*d, 3, padding=1)
        self.conv2_1 = nn.Conv2d(2*d, 2*d, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2*d)
        self.drop2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(2*d, 4*d, 3, padding=1)
        self.conv3_1 = nn.Conv2d(4*d, 4*d, 3, padding=1)
        self.conv3_2 = nn.Conv2d(4*d, 4*d, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(4*d)
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(4*d, 2*d, 3, padding=1)
        self.conv4_1 = nn.Conv2d(2*d, 2*d, 3, padding=1)
        self.conv4_2 = nn.Conv2d(2*d, 2*d, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(2*d)
        self.drop4 = nn.Dropout(0.5)

        self.conv5= nn.Conv2d(2*d, d, 3, padding=1)
        self.conv5_1 = nn.Conv2d(d, d, 3, padding=1)
        self.conv5_2 = nn.Conv2d(d, d, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(d)
        self.drop5 = nn.Dropout(0.5)

        self.pool = nn.MaxPool2d(2, 2)
        self.pool_4 = nn.MaxPool2d(4, 4)
        self.pool_8 = nn.MaxPool2d(8, 8)

        # self.fc1 = nn.Linear(14*14*(16+32+64+128), 10000)
        self.fc1 = nn.Linear(7*7*d, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, NUM_CLASSES)

        self.relu = nn.ReLU()

    def forward(self, x):

        # x: 3 x 227 x 227
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv1 = self.relu(self.bn1(self.conv1_1(conv1)))
        conv1 = self.drop1(conv1)
        conv1 = self.pool(conv1)

        # conv1: d x 113 x 113
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv2 = self.relu(self.bn2(self.conv2_1(conv2)))
        conv2 = self.drop2(conv2)
        conv2 = self.pool(conv2)

        # conv2: 128 x 56 x 56
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv3 = self.relu(self.bn3(self.conv3_1(conv3)))
        conv3 = self.relu(self.bn3(self.conv3_2(conv3)))
        conv3 = self.drop3(conv3)
        conv3 = self.pool(conv3)

        # conv3: 256 x 28 x 28
        conv4 = self.relu(self.bn4(self.conv4(conv3)))
        conv4 = self.relu(self.bn4(self.conv4_1(conv4)))
        conv4 = self.relu(self.bn4(self.conv4_2(conv4)))
        conv4 = self.drop4(conv4)
        conv4 = self.pool(conv4)

        # conv4: 512 x 14 x 14
        conv5 = self.relu(self.bn5(self.conv5(conv4)))
        conv5 = self.relu(self.bn5(self.conv5_1(conv5)))
        conv5 = self.relu(self.bn5(self.conv5_2(conv5)))
        conv5 = self.drop5(conv5)
        conv5 = self.pool(conv5)

        # conv5: 512 x 7 x 7

        # MultiScale Feature from conv1, conv2, and conv3
        multi_scale1 = self.pool_8(conv1)   # 16 x 14 x 14
        multi_scale2 = self.pool_4(conv2)   # 32 x 14 x 14
        multi_scale3 = self.pool(conv3)     # 64 x 14 x 14
        #
        flat1 = multi_scale1.view(multi_scale1.size()[0], 16 * 14 * 14)
        flat2 = multi_scale2.view(multi_scale2.size()[0], 32 * 14 * 14)
        flat3 = multi_scale3.view(multi_scale3.size()[0], 64 * 14 * 14)
        flat4 = conv4.view(conv4.size()[0], 32 * 14 * 14)
        flat5 = conv5.view(conv5.size()[0], 16 * 7 * 7)
        multi_scale_all = torch.cat((flat1, flat2, flat3, flat4), dim = 1)

        fc1 = self.relu(self.fc1(multi_scale_all))
        # fc1 = self.relu(self.fc1(flat5))
        fc1 = self.dropout(fc1)
        fc2 = self.relu(self.fc2(fc1))
        fc2 = self.dropout(fc2)
        fc3 = self.fc3(fc2)

        return fc3

# Network based on ResNet
