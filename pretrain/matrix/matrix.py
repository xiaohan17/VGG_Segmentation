import numpy as np
from sklearn.metrics import confusion_matrix
import torch


# model 是你训练好的模型
# device 是你的设备，比如 'cuda' 或 'cpu'
def compute_confusion_matrix(model, data_loader, device, num_classes,loss_fn):
    # 初始化混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()  # 进入评估模式
    losses=0
    total_samples = 0
    with torch.no_grad():  # 禁止梯度计算，节省内存
        for images, labels in data_loader:
            images, labels = images.to(torch.float32).to(device), labels.to(device)
            outputs = model(images)  # 得到模型输出
            loss=loss_fn(outputs, labels.squeeze(1).long())
            _, preds = torch.max(outputs, dim=1)  # 获取预测的类别

            # Flatten tensors to 1D arrays for confusion matrix calculation
            preds = preds.view(-1)
            labels = labels.view(-1)
            losses+=loss.item()
            # 计算混淆矩阵
            cm = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels=np.arange(num_classes))
            confusion += cm  # 累加每次的混淆矩阵
            total_samples += images.size(0)
    return confusion,losses/total_samples
