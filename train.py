import torch
from torchvision.models import densenet169 #这里可以换模型
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import numpy as np
import os
from load_datasets import DataLoader
from network import Network

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止BUG
# 定义参数
n_class = 3  # 类别数
batch_size = 4  # 训练的批次，如果报显存错误，调小即可
lr = 0.0001  # 学习率
epochs = 200  # 训练总次数
# 加载数据集
dataset_name = 'apple2orange'  # 数据集名字
dir_path = r'.\datasets\%s' % dataset_name  # 训练集图片
dataloader = DataLoader(dataset_name=dataset_name, norm_range=(0, 1), img_res=(512, 512))
x_train, y_train, x_test, y_test = dataloader.load_datasets(dir_path)
print('训练集%s类' % np.unique(y_train), x_train.shape, y_train.shape, x_train.min(), x_train.max())
print('测试集%s类' % np.unique(y_test), x_test.shape, y_test.shape, x_test.min(), x_test.max())

x_train = np.transpose(x_train, axes=(0, 3, 1, 2))
x_test = np.transpose(x_test, axes=(0, 3, 1, 2))
b, c, h, w = x_train.shape
torch_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),
                                                     torch.from_numpy(y_train).long())
torch_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
train_loader = torch.utils.data.DataLoader(dataset=torch_train_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=torch_test_dataset, batch_size=batch_size,
                                          shuffle=False)

a = next(iter(train_loader))
print(a[0].shape, a[1].shape)
model = Network(n_class=n_class).cuda()

for name, param in model.named_parameters():  # 这里的parma和上面的一样的，这里同样也可以调节让model的参数不训练
    if param.requires_grad:  # 打印出可训练的参数
        print('可训练的参数:', name)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 定义优化器
# 训练
try:  # 尝试载入模型，可断点续训
    model.load_state_dict(torch.load('model.h5'))
    print('断点续训')
except:
    pass
history = {'epoch': [], 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
for epoch in range(epochs):
    # 训练
    model.train()
    train_correct = 0
    train_total = 0
    train_total_loss = 0.
    for i, (x, y) in enumerate(train_loader):
        x = x.cuda()
        y = y.cuda()
        _, y_pred = model(x)
        train_loss = nn.CrossEntropyLoss()(y_pred, y)  # y不需要onehot编码！！
        # print('loss:', loss, loss.shape)
        # train_loss = train_loss.mean()
        train_total_loss += train_loss
        train_total += y.size(0)
        _, y_pred = torch.max(y_pred, dim=1)
        train_correct += (y_pred == y).sum().item()
        # 梯度归0反向传播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # print('loss:', loss.item())
    # 测试
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        test_total_loss = 0.
        for x, y in test_loader:
            x = x.cuda()
            y = y.cuda()
            _, y_pred = model(x)
            test_loss = nn.CrossEntropyLoss()(y_pred, y)
            test_total_loss += test_loss
            _, y_pred = torch.max(y_pred, dim=1)
            test_total += y.size(0)
            test_correct += (y_pred == y).sum().item()
            # print(torch.equal(pred, labels))  # 只能比较整个tensor是不是一样的，不能逐个比较元素

    loss = train_total_loss / (i + 1)
    val_loss = test_total_loss / (i + 1)
    accuracy = train_correct / train_total
    val_accuracy = test_correct / test_total
    history['loss'].append(np.array(loss.cpu().detach()))
    history['val_loss'].append(np.array(val_loss.cpu().detach()))
    history['accuracy'].append(np.array(accuracy))
    history['val_accuracy'].append(np.array(val_accuracy))
    history['epoch'].append(epoch)
    print('epochs:%s/%s:' % (epoch + 1, epochs),
          'loss:%.6f' % history['loss'][epoch], 'accuracy:%.6f' % history['accuracy'][epoch],
          'val_loss:%.6f' % history['val_loss'][epoch], 'val_accuracy:%.6f' % history['val_accuracy'][epoch])
    # 保存模型
    torch.save(model.state_dict(), 'model.h5')
