import logging
import argparse
import os.path
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from pretrain.datasets.Datafusion import Datafusion
from pretrain.model.VGG import VGG16
from osgeo import gdal


logging.basicConfig(
    filename='deep_learning_inference.log',
    format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s]',
    level=logging.INFO,
    filemode="w"
)


def get_args():
    parser=argparse.ArgumentParser(description="Deep Learning",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dd',"--data_dir",type=str,default="../../Track2",help="data root path")
    parser.add_argument('-mp', "--model_param", type=str, default="./train/Track2/10_epoch_model_weight.pth", help="modle param path")
    parser.add_argument('-bs', "--bach_size", type=int, default=2, help="bach size")
    parser.add_argument('-wd', "--work_dir", type=str, default="./train/Track2/infer", help="Inference work dir")

    return parser.parse_args()

def save_tif(work_dir,dataset,step):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        print("Error: Unable to get GTiff driver.")
        return None
    # print(dataset.shape)
    for i,data in enumerate(dataset):
        data_arry=np.array(data)
        result=np.zeros((data_arry.shape[1],data_arry.shape[2]),dtype=int)
        for j in range(data_arry.shape[1]):
            for k in range(data_arry.shape[2]):
                max_band=np.argmax(data_arry[:,j,k])
                result[j,k]=max_band
        output_path = os.path.join(work_dir, f"step_{step}_i_{i}.tif")
        img_width = dataset.shape[2]
        img_height = dataset.shape[3]
        datasetnew = driver.Create(output_path, img_width, img_height, 1, gdal.GDT_Float32)
        if datasetnew is None:
            print("Error: Unable to get GTiff datasetnew.")
            return
        # datasetnew.SetGeoTransform(None)
        # datasetnew.SetProjection('EPSG:4326')
        band = datasetnew.GetRasterBand(1)
        band.WriteArray(result)
        datasetnew.FlushCache()
        del datasetnew




def test(model,work_dir,dataloader):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

    #推理
    logging.info("infer mode")

    for step, [data,label] in enumerate(dataloader):
        model.eval()
        data = data.to(torch.float32).to(device)
        pre = model(data)

        pre_cpu=pre.to("cpu")
        save_tif(work_dir,pre_cpu.detach(),step)



if __name__=="__main__":

    classes=["no_water","water"]
    args=get_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
        print("succeed create dir")
    else:
        pass

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_label=transforms.Compose([
        transforms.ToTensor()
    ])


    data=Datafusion(args.data_dir,transform=transform,transform_label=transform_label)

    Inferdataloader=DataLoader(data,batch_size=args.bach_size,shuffle=False,drop_last=False,pin_memory=True)

    VGG16Model=VGG16()
    test(VGG16Model,work_dir=args.work_dir,dataloader=Inferdataloader)