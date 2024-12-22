import logging
import argparse
import os.path
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.optim import  SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from pretrain.matrix.matrix import compute_confusion_matrix
from pretrain.matrix.Index import condusion_matrix_Index

from pretrain.datasets.Datafusion import Datafusion
from pretrain.model.VGG import VGG16

logging.basicConfig(
    filename='deep_learning_infer.log',
    format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s]',
    level=logging.INFO,
    filemode="w"
)


def get_args():
    parser=argparse.ArgumentParser(description="Deep Learning",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dd',"--data_dir",type=str,default="../../Track2",help="data root path")
    parser.add_argument('-bs', "--bach_size", type=int, default=2, help="bach size")
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-8, help="model learning rate")
    parser.add_argument('-ep', "--epoches", type=int, default=10,  help="model train epoches")
    parser.add_argument('-init', "--is_init", action="store_true", help="model param init")
    parser.add_argument('-wd', "--work_dir", type=str, default="./train/Track2", help="Model work dir")
    parser.add_argument("--by_epoch", action="store_true",help="Whether to use epoch")

    return parser.parse_args()




def train(model,lr,work_dir,epoches,dataloader,valdataloader,interval,classes,by_epoch=True):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    #损失
    loss_fn=nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    #优化器
    optimer=SGD(params=model.parameters(),lr=lr,momentum=0.9 ,weight_decay=0)

    #学习率更新
    lr_schlar=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimer,T_max=20,eta_min=0.05)

    #summary展示
    summery=SummaryWriter(log_dir="{}/summary_workdir".format(work_dir))


    #训练
    if by_epoch:
        logging.info("epoch mode")
        for epoch in range(epoches):
            print("------------------第{}个epoch------------------".format(epoch+1))
            model.train()
            losses=0
            for step,[data,label] in enumerate(dataloader):
                # print(data.shape)
                data=data.to(torch.float32).to(device)
                label=label.to(device)
                # print("label shape{} ".format(label.shape))
                #预测
                pre=model(data)
                #计算损失
                loss=loss_fn(pre,label.squeeze(1).long())
                # print(loss)
                # 每次梯度归零
                optimer.zero_grad()
                #反向传播
                loss.backward()
                #优化器优化
                # print("{} optime: {}".format(step,optimer.param_groups[0]['lr']))
                optimer.step()
                # print("{} scheduler: {}".format(step,lr_schlar.get_last_lr()))
                lr_schlar.step()
                losses = loss.item()

            if by_epoch and ((epoch+1) % interval == 0):
                torch.save(model,work_dir+"/{}_epoch_model_weight.pth".format(epoch+1))
                val_model=torch.load(work_dir+"/{}_epoch_model_weight.pth".format(epoch+1))
                conf_matrix,val_loss = compute_confusion_matrix(val_model,valdataloader,device,len(classes),loss_fn)
                #计算指数
                precision_list,Recall_list,F1_score_list,Accurary= condusion_matrix_Index(conf_matrix,classes)
                # 计算精度平均值
                M_Precision = np.mean(precision_list)
                M_Recall = np.mean(Recall_list)
                M_F1_score = np.mean(F1_score_list)
                # print(conf_matrix.shape)
                logging.info("----epoch:[{}/{}]----Val_Accurary:{:.6f}----Val_M_loss:{:.6f}----Val_M_Precision:{:.6f}----Val_M_Recall:{:.6F}----Val_M_F1_score{:.6F}----".format(epoch+1,epoches,Accurary,val_loss,M_Precision,M_Recall,M_F1_score))
                summery.add_scalar('Val_M_loss',scalar_value=val_loss,global_step=epoch+1)
                summery.add_scalar("Val_Accurary",scalar_value=Accurary,global_step=epoch+1)
                summery.add_scalar("Val_M_Precision",scalar_value=M_Precision,global_step=epoch+1)
                summery.add_scalar("Val_M_Recall", scalar_value=M_Recall, global_step=epoch + 1)
                summery.add_scalar("Val_M_F1_score", scalar_value=M_F1_score, global_step=epoch + 1)
                summery.close()
        print("------------------计算完成------------------")
    else:
        logging.info("iter mode")
        logging.warning("coming soon")
        print("------------------计算完成------------------")



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

    dataloader=DataLoader(data,batch_size=args.bach_size,shuffle=True,drop_last=False,pin_memory=True)
    valdataloader=DataLoader(data,batch_size=args.bach_size,shuffle=True,drop_last=False,pin_memory=True)
    VGG16Model=VGG16()
    #初始化
    if args.is_init:
        for m in VGG16Model.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight,mode='fan_in')
        print("init success")
    else:
        print(VGG16Model.state_dict())
    train(VGG16Model, lr=args.learning_rate, work_dir=args.work_dir, epoches=args.epoches,
          dataloader=dataloader,valdataloader=valdataloader,interval=2,classes=classes,by_epoch=args.by_epoch)