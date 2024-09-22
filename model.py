import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

class YoloV1_Model(nn.Module):
    def __init__(self, S, B, C):
        """
        - Describe
            constructor of this class and create yoloV1 architecture
        - Input
            + S (int) : number of grid cell = (S * S)
            + B (int) : number of bounding box per grid cell
            + c (int) : number of class in dataset 
        - Output
            None
        """

        super(YoloV1_Model, self).__init__()
        self.feature = nn.Sequential(
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
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*7*7, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, 1470)  # 1470 = 7*7*(2*5 + num_classes) where 2 is for bounding boxes
        )
    
    def forward(self, x):
        """
        - Describe
            forward propogation of model
        - Input
            + x (tensor) [N, 3, 448, 448] : picture size 448 * 448 in tensor ; N is batch_size
        - Output
            + x (tensor) [N, 1470] : predicted output in yolo label ; N is batch_size
        """

        x = self.feature(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":

    """
    input shape for model
        m_data = torch.randn(1, 3, 448, 448)
        
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    test = YoloV1_Model(S = 7, B = 2, C = 20).to(device)
    # print(test)

    summary(test, (3, 448, 448))

