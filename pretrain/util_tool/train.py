import torch.nn as nn
import torch.utils.data
from torch.optim import  SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision
from pretrain.model import  VIT,VGG
from pretrain.util_tool.DDP import optim


def train(model,epochs,lr,train_data,interval:int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.__version__)
    Model=model.to(device)
    loss_fn=nn.CrossEntropyLoss()
    loss_fn=loss_fn.to(device)
    optimer=SGD(Model.parameters(),lr=lr)
    lr_schduler=torch.optim.lr_scheduler.StepLR(optimizer=optimer,step_size=30,gamma=0.1)
    work_dir="./VIT"
    writer=SummaryWriter("{}/logs_batch_norm".format(work_dir))
    for epoch in range(epochs):
        losses = 0
        print("---------epoch {}---------".format(epoch+1))
        for step,[data,targe] in enumerate(train_data):

            data=data.to(device)
            targe=targe.to(device)
            out=Model(data)
            loss=loss_fn(out,targe)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            print(loss.item())
            losses = losses + loss.item()
        lr_schduler.step()
        if(epochs%interval==0):
            print("The epoch:{},loss_epoch:{}".format(epoch,losses))
            # writer.add_scalars("VIT_LOSS_train",losses,epoch+1)
            # writer.add_scalar("VIT_lr",lr_schduler.get_lr(),epoch+1)
            torch.save(Model.state_dict(), "{}/epoch_{}_VIT_norm.pth".format(work_dir, epoch+1))
            writer.close()


if __name__=="__main__":
    train_path = '../../datasets/train'
    val_path = '../../datasets/val'
    data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    bs=1
    train_data = torchvision.datasets.ImageFolder(root=train_path, transform=data_transform)
    val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=data_transform)
    print(train_data[0])
    train_data_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=bs,shuffle=True,num_workers=2)
    val_dataset_loader=torch.utils.data.DataLoader(dataset=val_dataset,batch_size=bs,shuffle=True,num_workers=2)
    # model_vit=VIT(image_size=224,
    #               patch_size=32,
    #               num_class=5,
    #               dim=1024,
    #               depth=6,
    #               heads=16,
    #               mlp_dim=2048,
    #               emb_Dropout=0.1,
    #               trans_num=3)
    # epochs=30
    # lr=0.01
    # for m in model_vit.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.kaiming_normal_(m.weight,mode='fan_in')
    # print(model_vit.state_dict())
    # interval=5
    # train(model=model_vit,epochs=epochs,lr=lr,train_data=train_data_loader,interval=interval)