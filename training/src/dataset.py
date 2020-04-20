# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-

import tensorflow as tf

from dataset_augment import pose_random_scale, pose_rotation, pose_flip, pose_resize_shortestedge_random, \
    pose_crop_random, pose_to_img
from dataset_prepare import CocoMetadata
from os.path import join
from pycocotools.coco import COCO
from multiprocessing import Process,Queue
import random
import numpy as np

BASE = "/root/hdd"
BASE_PATH = ""
#TRAIN_JSON = "ai_challenger_train.json"
#VALID_JSON = "ai_challenger_valid.json"
TRAIN_JSON = "train.json"
VALID_JSON = "test.json"


TRAIN_ANNO = None
VALID_ANNO = None
CONFIG = None

train_sequence_data = None
valid_sequence_data = None


def set_config(config):
    global CONFIG, BASE, BASE_PATH
    CONFIG = config
    BASE = CONFIG['imgpath']
    BASE_PATH = CONFIG['datapath']


def _parse_function(imgId, is_train, ann=None):
    """
    :param imgId:
    :return:
    """

    global TRAIN_ANNO
    global VALID_ANNO

    if ann is not None:
        if is_train == True:
            TRAIN_ANNO = ann
        else:
            VALID_ANNO = ann
    else:
        if is_train == True:
            anno = TRAIN_ANNO
        else:
            anno = VALID_ANNO

    img_meta = anno.loadImgs([imgId])[0]
    anno_ids = anno.getAnnIds(imgIds=imgId)
    img_anno = anno.loadAnns(anno_ids)
    idx = img_meta['id']
    img_path = join(BASE, img_meta['file_name'])

    img_meta_data = CocoMetadata(idx, img_path, img_meta, img_anno, sigma=6.0)
    img_meta_data = pose_random_scale(img_meta_data)
    img_meta_data = pose_rotation(img_meta_data)
    img_meta_data = pose_flip(img_meta_data)
    img_meta_data = pose_resize_shortestedge_random(img_meta_data)
    img_meta_data = pose_crop_random(img_meta_data)
    return pose_to_img(img_meta_data)


def _set_shapes(img, heatmap):
    img.set_shape([CONFIG['input_height'], CONFIG['input_width'], 3])
    heatmap.set_shape(
        [CONFIG['input_height'] / CONFIG['scale'], CONFIG['input_width'] / CONFIG['scale'], CONFIG['n_kpoints']])
    return img, heatmap

class SequenceData():
    def __init__(self, anno, batch_size=32, epoch=10, buffer_size=30, is_train=True):
        self.anno = anno
        self.batch_size = batch_size
        self.epoch = epoch
        self.is_train = is_train
        self.datas = anno.getImgIds()
        self.L = len(self.datas) 
        self.index = []
        for epoch_id in range(epoch):
            epoch_index = random.sample(range(self.L), self.L)
            self.index.extend(epoch_index)
        self.L = len(self.index) 
        self.queue = Queue(maxsize=buffer_size)
        
        self.Process_num=CONFIG['multiprocessing_num']
        for i in range(self.Process_num):
            print(i,'start')
            ii = int(self.__len__()/self.Process_num)
            t = Process(target=self.f,args=(i*ii,(i+1)*ii))
            t.start()
    def __len__(self):
        return self.L - self.batch_size
    def __getitem__(self, idx):
        batch_indexs = self.index[idx:(idx+self.batch_size)]
        batch_datas = [self.datas[k] for k in batch_indexs]
        imgs,heats = self.data_generation(batch_datas)
        return imgs,heats
    
    def f(self,i_l,i_h):
        for i in range(i_l,i_h):
            t = self.__getitem__(i)
            self.queue.put(t)

    def gen(self):
        while 1:
            t = self.queue.get()
            yield t[0],t[1]

    def data_generation(self, batch_datas):
        #数据预处理操作
        imgs = []
        heats = []
        for d in batch_datas:
            img, heat = _parse_function(d, self.is_train)
            #img, heat = _set_shapes(tf.convert_to_tensor(img), tf.convert_to_tensor(heat))
            imgs.append(img)
            heats.append(heat)

        return imgs, heats

def _get_dataset_pipeline(anno, batch_size, epoch, buffer_size, is_train=True):
    global train_sequence_data, valid_sequence_data

    if is_train and train_sequence_data is None:
        train_sequence_data = SequenceData(anno, batch_size, epoch, buffer_size, is_train)
        dataset = tf.data.Dataset().batch(1).from_generator(train_sequence_data.gen,
                                           output_types= (tf.float32,tf.float32))

    if not is_train and valid_sequence_data is None:
        valid_sequence_data = SequenceData(anno, batch_size, epoch, buffer_size, is_train)
        dataset = tf.data.Dataset().batch(1).from_generator(valid_sequence_data.gen,
                                           output_types= (tf.float32,tf.float32))

    return dataset

def get_train_dataset_pipeline(batch_size=32, epoch=10, buffer_size=1):
    global TRAIN_ANNO

    anno_path = join(BASE_PATH, TRAIN_JSON)
    print("preparing annotation from:", anno_path)
    TRAIN_ANNO = COCO(
        anno_path
    )
    return _get_dataset_pipeline(TRAIN_ANNO, batch_size, epoch, buffer_size, True)

def get_valid_dataset_pipeline(batch_size=32, epoch=10, buffer_size=1):
    global VALID_ANNO

    anno_path = join(BASE_PATH, VALID_JSON)
    print("preparing annotation from:", anno_path)
    VALID_ANNO = COCO(
        anno_path
    )
    return _get_dataset_pipeline(VALID_ANNO, batch_size, epoch, buffer_size, False)
