import torch
from torchvision.models import resnet101 #这里可以换模型
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import numpy as np
import os
from network import Network

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 防止BUG

n_class = 3  # 类别数
# 加载图片
original_img = cv.imread(r'test/604_Paris_2.3187402_48.8239733.jpg')  # 这里改成你图片的路径
image = original_img.astype('float32') / 255.  # 预处理，归一化为0-1
image = torch.from_numpy(np.expand_dims(image.transpose([2, 0, 1]), axis=0))  # (b,c,h,w)
print(image.shape)
# 加载预训练模型
model = Network(n_class=n_class)
try:  # 尝试载入模型，可断点续训
    model.load_state_dict(torch.load('model.h5'))
    print('载入模型')
except:
    print('模型未载入！！')
model.eval()
out, logits = model(image)  # (b,512,h,w)  (b,n_class)
# out.requires_grad=True
y_pred = torch.softmax(logits, dim=-1)  # (b,n_class)
cate_num = torch.argmax(y_pred, dim=-1)[0]  # (b,) (1,)
cate_prob = y_pred[:, cate_num.detach().cpu().numpy()][0].mean(axis=-1)

print('类别：', cate_num.numpy(), '概率：', cate_prob.detach().cpu().numpy())
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

heatmap = cv.resize(heatmap, dsize=(original_img.shape[1], original_img.shape[0]), interpolation=cv.INTER_CUBIC)
heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_JET)
dst = cv.addWeighted(original_img, 0.7, heatmap, 0.5, 0)
cv.putText(dst, text=r'class:%s  acc:%s' % (cate_num.numpy(), np.round(cate_prob.detach().cpu().numpy(), 2)),
           org=(0, dst.shape[0] // 10), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=(0, 0, 255))

cv.imwrite('grad-camt.png', dst) #保存在当前目录，和对应文件名
print('已保存结果到当前目录')
