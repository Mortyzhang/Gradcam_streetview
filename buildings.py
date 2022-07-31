#!--*-- coding:utf-8 --*--

# Deeplab Demo

import os
import tarfile
import numpy as np
from PIL import Image
import tensorflow as tf
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] ="0"



class DeepLabModel(object):

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf.Graph()
        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)


    def run(self, image):

        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(image)]})
        seg_map = batch_seg_map[0]
        return seg_map

def convert(orl,mask):
    rows,cols = mask.shape
    img = np.ones((rows,cols,3),dtype="int")
    for i in range(rows):
        for j in range(cols):
            if mask[i][j] == 2:  # 这里控制露出的 class 8是绿化，2是building
                img[i][j][0] = orl[i][j][0]
                img[i][j][1] = orl[i][j][1]
                img[i][j][2] = orl[i][j][2]


    return img

def panduan(root):
    return os.path.exists(root)

if __name__ == '__main__':
    in_dir = r"E:\Streetview\nanjing\south"   #街景照片原始数据存储位置
    out_dir = r"E:\Streetview\nanjing\south_building"    #街景语义分割结果数据保存位置
    model_file = r"E:\街景爬取学习\课程资料\deeplabv3_cityscapes_train_2018_02_06.tar.gz"   #大模型，已训练完成模型位置
    # model_file = r"F:\The image of the city\deeplab\deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz"  # 小模型，已训练完成模型位置
    # model_file = r"D:\街景爬取学习\pspnet50"   # new model

    t = int(time.time())
    MODEL = DeepLabModel(model_file)
    for imgfile in os.listdir(in_dir):
        try:
            if panduan(os.path.join(out_dir,imgfile)):
                print("hehe")
                continue
            st = int(time.time())
            orignal_im = Image.open(os.path.join(in_dir,imgfile))
            seg_map = MODEL.run(orignal_im)
            img = convert(np.array(orignal_im,dtype="float"),seg_map)
            new_im = Image.fromarray(img.astype('uint8'))
            new_im.save(os.path.join(out_dir,imgfile))
            print(imgfile,"spend time",int(time.time())-st,"s")
        except:
            print("no direction")
    print(int(time.time())-t)

