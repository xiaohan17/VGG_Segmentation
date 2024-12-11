import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from osgeo import gdal


#读取数据融合竞赛Track2的数据
class Datafusion(Dataset):
    def __init__(self,data_dir,transform=None,transform_label=None,is_train=True):
        self.data_dir=data_dir
        self.transform=transform
        self.is_train=is_train
        self.transform_label=transform_label
        if self.is_train:
            image_dir=os.path.join(data_dir,"train\images")
            label_dir=os.path.join(data_dir,"train\labels")
            image_names=os.listdir(image_dir)
            label_names=os.listdir(label_dir)
            self.image_paths=[os.path.join(image_dir,image_name) for image_name in image_names ]
            self.label_paths=[os.path.join(label_dir,label_name) for label_name in label_names ]
        else:
            print("还没写")


    def __getitem__(self, item):
        img_path=self.image_paths[item]
        img=gdal.Open(img_path).ReadAsArray()
        img=img.astype(np.float64)
        label=np.array(Image.open(self.label_paths[item]).convert("L")).astype(np.float32)
        if self.transform is not None:
            #由于读取的数据是C H W ,transform.ToTensor只能将H W C的形状正常转换
            img_tensor=self.transform(img).permute(1, 0, 2)
            if self.transform_label is not None:
                label_tensor=self.transform_label(label)
                return img_tensor,label_tensor
            else:
                print("小警告")
        else:
            print("请裁剪为相同大小，转为tensor")

    def __len__(self):
        return len(self.image_paths)


if __name__=="__main__":

    data_dir="../../Track2"

    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    transform_label=transforms.Compose([
        transforms.ToTensor()
    ])

    data=Datafusion(data_dir=data_dir,transform=transform,is_train=True,transform_label=transform_label)
    print(data[0])
    print(data[0][1].shape)
    print(data[0][0].shape)
    train_dataloader=DataLoader(data,batch_size=22,shuffle=True,drop_last=False,pin_memory=True)



