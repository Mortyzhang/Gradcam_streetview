import torch
from torchvision.models import densenet169 #这里可以换模型
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import numpy as np
import os
from network import Network
from load_datasets import DataLoader
from torchvision.transforms import Resize

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止BUG
# 定义参数
n_class = 3  # 类别数
sample_batch = 900  # 随机生成几张图片的类激活图
h, w = 128, 128  # 训练时候图片送入神经网络的高和宽
# 加载数据集
dataset_name = 'apple2orange'  # 数据集名字
dir_path = r'.\datasets\%s' % dataset_name  # 训练集图片
dataloader = DataLoader(dataset_name=dataset_name, norm_range=(0, 1), img_res=None)
x_train, y_train, x_test, y_test = dataloader.load_datasets(dir_path)
print('训练集%s类' % np.unique(y_train), x_train.shape, y_train.shape, x_train.min(), x_train.max())
print('测试集%s类' % np.unique(y_test), x_test.shape, y_test.shape, x_test.min(), x_test.max())

ids = np.random.choice(len(x_test), size=sample_batch)
print('ids:', ids)

y_test_batch = y_test[ids]
x_test_batch = x_test[ids]
original_x_test_batch = (x_test_batch * 255.).astype('uint8')
x_test_batch = torch.from_numpy(x_test_batch.transpose([0, 3, 1, 2]))  # (b,c,h,w)
#print(x_test_batch.shape)
x_test_batch = Resize(size=(h, w))(x_test_batch)
print(x_test_batch.shape, y_test_batch.shape)

# 加载预训练模型
model = Network(n_class=n_class)
try:  # 尝试载入模型，可断点续训
    model.load_state_dict(torch.load('model.h5'))
    print('载入模型')
except:
    print('模型未载入！！')
model.eval()

dst_list = []  # 输出的一个batch的类激活图
for i, (image, original_x_test) in enumerate(zip(x_test_batch, original_x_test_batch)):
    image = image.unsqueeze(0)  # (1,c,h,w)
    out, logits = model(image)  # (b,512,h,w)  (b,n_class)
    # out.requires_grad=True
    y_pred = torch.softmax(logits, dim=-1)  # (b,n_class)
    cate_num = torch.argmax(y_pred, dim=-1)[0]  # (b,) (1,)
    cate_prob = y_pred[:, cate_num.detach().cpu().numpy()][0].mean(axis=-1)

    print('第%s张图片' % (i + 1), '类别：', cate_num.numpy(), '概率：', cate_prob.detach().cpu().numpy())
    grads = torch.autograd.grad(cate_prob, [out])[0]  # (b,512,h,w) 类别对特征层的梯度
    grads = torch.mean(grads, dim=(-1, -2), keepdim=True)  # (b,512,1,1)  获得此类别的权重
    # 以下都是绘制类激活图
    heatmap = torch.mul(out, grads).permute(0, 2, 3, 1).detach().cpu().numpy()  # (b,h,w,512) 权重与特征层相乘
    heatmap = np.mean(heatmap, axis=-1)  # (b,h,w) 对通道取平均
    # 显示heatmap
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heat_max = np.max(heatmap)
    heatmap = heatmap / heat_max
    heatmap = np.clip(heatmap, 0, 1) * 255.
    heatmap = heatmap.astype('uint8')
    heatmap = cv.resize(heatmap, dsize=(original_x_test.shape[1], original_x_test.shape[0]),
                        interpolation=cv.INTER_CUBIC)
    heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
    dst = cv.addWeighted(original_x_test, 0.6, heatmap, 0.5, 0)
    cv.putText(dst, text=r'confidence:%s' % (np.round(cate_prob.detach().cpu().numpy(), 2)),
               org=(0, dst.shape[0] // 10), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.55, color=(0, 0, 255))
    # cv.putText(dst, text=r'class:%s  confidence:%s' % (cate_num.numpy(), np.round(cate_prob.detach().cpu().numpy(), 2)),
    #            org=(0, dst.shape[0] // 10), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.55, color=(0, 0, 255))
    cv.imwrite('.\show\grad-cam_batch%d.png' % (i + 1), dst)
print('已保存%s张结果到当前目录' % sample_batch)
