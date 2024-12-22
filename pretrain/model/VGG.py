import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1=nn.Sequential(
            nn.Conv2d(in_channels=12,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2,2),
        )
        # self.fc=nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(100352,4096),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #     nn.Linear(4096,4096),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(0.5),
        #     nn.Linear(4096,5)
        # )
        self.upsample=nn.Upsample((128 ,128),mode='nearest')
        self.decoder=nn.Sequential(nn.Conv2d(512,2,3,1,1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(2))
        self.softmax=nn.Softmax2d()
    def forward(self,x):
        x1=self.stage1(x)
        # print("x1 size:{}".format(x1.shape))
        x2=self.stage2(x1)
        # print("x2 size:{}".format(x2.shape))
        x3=self.stage3(x2)
        # print("x3 size:{}".format(x3.shape))
        x4=self.stage4(x3)
        # x6=self.fc(x4)
        # print("x4 size:{}".format(x4.shape))
        x5=self.upsample(x4)
        # print("upsample size :{}".format(x5.shape))
        x6=self.decoder(x5)
        # out=self.softmax(x6)
        return x6


if __name__=="__main__":
    img=torch.rand((1,12,512,512))
    VGG=VGG16()
    PRE=VGG(img)
    print(PRE.shape)
    print(PRE[0][0][45][35])

