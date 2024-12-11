import logging
import argparse
import torch.nn as nn
import torch.utils.data
from torch.optim import  SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from pretrain.datasets.Datafusion import Datafusion
from pretrain.model.VGG import VGG16

logging.basicConfig(
    filename='deep_learning.log',
    format='[%(asctime)s][%(filename)s][%(levelname)s][%(message)s]',
    level=logging.INFO,
    filemode="w"
)

def get_args():
    parser=argparse.ArgumentParser(description="Deep Learning",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dd',"--data_dir",type=str,default="../../Track2",help="data root path")
    parser.add_argument('-bs', "--bach_size", type=int, default=2, help="bach size")
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-8, help="model learning rate")
    parser.add_argument('-ep', "--epoches", type=int, default=2,  help="model train epoches")
    parser.add_argument('-init', "--is_init", action="store_false", help="model param init")
    parser.add_argument('-wd', "--work_dir", type=str, default="./train/Track2", help="Model work dir")

    return parser.parse_args()


def train(model,lr,work_dir,epoches,dataloader,interval,by_epoch=True):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    #损失
    loss_fn=nn.BCEWithLogitsLoss()
    loss_fn = loss_fn.to(device)
    #优化器
    optimer=SGD(params=model.parameters(),lr=lr,momentum=0.9 ,weight_decay=0)

    #学习率更新
    lr_schlar=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimer,T_max=20,eta_min=0.05)

    #summary展示
    summery=SummaryWriter(log_dir="{}/summary_workdir".format(work_dir))

    #训练
    for epoch in range(epoches):

        print("------------------第{}个epoch------------------".format(epoch))
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
            loss=loss_fn(pre,label)
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
            # loss_value_iter = loss.item()
            # print("{} loss : {}".format(step, loss_value_iter))
            if by_epoch and (epoch+1 % interval == 0):
                logging.info("------------------{} epoch------------------".format(epoch+1))
                losses=loss.item()
                logging.info("----loss:{}----".format(losses))
                torch.save(model.state_dict(),work_dir+" {} epoch_model_weight.pth".format(epoch+1))
                summery.add_scalar('loss',scalar_value=loss.item(),global_step=epoch+1)
                summery.close()
            elif by_epoch is False and (epoch%interval==0):
                logging.info("------------------{} iter------------------".format(step+1))
                logging.info("----loss:{}----".format(loss.item()))
                summery.add_scalar('loss',scalar_value=loss.item(),global_step=step+1)
                summery.close()
                torch.save(model.state_dict(), work_dir)
                torch.save(model.state_dict(), work_dir + "{}step_model_weight.pth".format(step+1))
        print("------------------计算完成------------------".format(epoch))

if __name__=="__main__":
    args=get_args()
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_label=transforms.Compose([
        transforms.ToTensor()
    ])

    data=Datafusion(args.data_dir,transform=transform,transform_label=transform_label)

    dataloader=DataLoader(data,batch_size=args.bach_size,shuffle=True,drop_last=False,pin_memory=True)
    VGG16Model=VGG16()
    #初始化
    # if args.is_init:
    #     for m in VGG16Model.modules():
    #         if isinstance(m,(nn.Conv2d,nn.Linear)):
    #             nn.init.kaiming_normal_(m.weight,mode='fan_in')
    #     train(VGG16Model,lr=args.learning_rate,work_dir=args.work_dir,epoches=args.epoches,dataloader=dataloader,interval=2)
    # else:
    #     train(VGG16Model,lr=args.learning_rate,work_dir=args.work_dir,epoches=args.epoches,dataloader=dataloader,interval=2)
    train(VGG16Model, lr=args.learning_rate, work_dir=args.work_dir, epoches=args.epoches, dataloader=dataloader,interval=2)









