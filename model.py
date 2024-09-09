import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

# model = MyNeuralNetwork().to(device)
# print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class my_yoloV1(nn.Module):
    def __init__(self, S, B, C):
        """
        + describe
            constructor of this class and create yoloV1 architecture
        + input
            S (int) : number of grid cell = (S * S)
            B (int) : number of bounding box per grid cell
            c (int) : number of class in dataset 
        """

        super(my_yoloV1, self).__init__()
        self.architechture = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Flatten(),

            nn.Linear(S * S * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096,  S * S *(B * 5 + C))
        )
    
    def forward(self, x):
        x = self.architechture(x)
        return x


if __name__ == "__main__":
    test = my_yoloV1(S = 7, B = 2, C = 20).to(device)
    # print(test)

    summary(test, (3, 448, 448))

