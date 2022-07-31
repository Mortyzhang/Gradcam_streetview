import sys, os
import cv2 as cv
import numpy as np
import torch
print(torch.__version__)

class DataLoader():
    def __init__(self, dataset_name, norm_range=(0, 1), img_res=(256, 256)):
        '''
        :param dataset_name: datasets文件夹里边的数据集文件夹名字
        :param norm_range: 输入0-255的图片，归一化为每个像素值在norm_range区间内(含边界)的图片
        :param img_res: 图片形状变为(h,w)
        '''
        self.dataset_name = dataset_name
        self.norm_range = norm_range
        self.img_res = img_res

    def img_normalize(self, image, norm_range=(0, 1)):
        '''
        输入0-255的图片，归一化为每个像素值在norm_range区间内(含边界)的图片
        '''
        image = np.array(image).astype('float32')
        image = (norm_range[1] - norm_range[0]) * image / np.max(image) + norm_range[0]
        return image

    def load_image(self, image_dir_path, class_name):  # image_dir_path为放图片的文件夹，class_name类别文件夹的名字
        ImgTypeList = ['jpg', 'JPG', 'bmp', 'png', 'jpeg', 'rgb', 'tif']
        x_list = []
        y_list = []
        image_names = os.listdir(image_dir_path)
        for image_name in image_names:
            if image_name.split('.')[-1] in ImgTypeList:  # 说明是图片，往下操作
                image = cv.imread(os.path.join(image_dir_path, image_name), flags=-1)  # flags=-1是以完整的形式读入图片
                image = self.img_normalize(image, self.norm_range)
                if self.img_res is not None:
                    image = cv.resize(image,
                                      dsize=(self.img_res[1], self.img_res[0]))  # 如果图片大小与需要裁剪的大小不一样，裁剪图片为指定大小
                if len(image.shape) == 2:  # 说明是灰图
                    image = np.expand_dims(image, axis=-1)
                x_list.append(image)
                # ----------这里可改成你想要的标签，也可以按照此格式继续增加标签-----------
                if class_name == 'H':  # 图片文件夹名字为'A'的，作为第0类
                    y_list.append(0)
                elif class_name == 'M':  # 图片文件夹名字为'B'的，作为第1类
                    y_list.append(1)
                elif class_name == 'L':  # 图片文件夹名字为'B'的，作为第1类
                    y_list.append(2)
                # ----------这里可改成你想要的标签，也可以按照此格式继续增加标签-----------
        x = np.array(x_list)
        y = np.array(y_list)
        return x, y

    def load_datasets(self, dir_path):
        file_names = os.listdir(dir_path)
        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []
        for i, file_name in enumerate(file_names):
            print(file_name)
            file_name_path = os.path.join(dir_path, file_name)
            class_names = os.listdir(file_name_path)
            for class_name in class_names:
                image_dir_path = os.path.join(file_name_path, class_name)
                print(image_dir_path)
                if file_name == 'train':  # 训练集文件夹名字，可能需要改成你的
                    x_train, y_train = self.load_image(image_dir_path, class_name)
                    x_train_list.append(x_train), y_train_list.append(y_train)
                elif file_name == 'test':  # 测试集文件夹名字，可能需要改成你的
                    x_test, y_test = self.load_image(image_dir_path, class_name)
                    x_test_list.append(x_test), y_test_list.append(y_test)
        return np.concatenate(x_train_list, axis=0), np.concatenate(y_train_list, axis=0), \
               np.concatenate(x_test_list, axis=0), np.concatenate(y_test_list, axis=0)



def main():
    dataset_name = 'apple2orange'  # 数据集名字
    dir_path = r'.\datasets\%s' % dataset_name  # 训练集图片
    dataloader = DataLoader(dataset_name=dataset_name, norm_range=(-1, 1), img_res=(512, 512))
    x_train, y_train, x_test, y_test = dataloader.load_datasets(dir_path)
    print('训练集%s类' % np.unique(y_train), x_train.shape, y_train.shape, x_train.min(), x_train.max())
    print('测试集%s类' % np.unique(y_test), x_test.shape, y_test.shape, x_test.min(), x_test.max())


if __name__ == '__main__':
    main()
