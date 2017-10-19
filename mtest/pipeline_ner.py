#!/usr/bin python
# -*- coding:utf-8 -*-

"""
    说明：
        1：pip中已经安装了anago这个包，但是这里的测试代码是依托repositoriesgit库中的这个anago项目
        2：目的是跑通网络，查看比较
"""

import os
from anago import config
from anago.data import reader,preprocess

if __name__ == '__main__':

    # 设置参数
    DATA_ROOT = 'data/conll2003/en/ner'
    SAVE_ROOT = './models'  # trained model
    LOG_ROOT = './logs'  # checkpoint, tensorboard
    embedding_path = './data/glove.6B/glove.6B.100d.txt'
    model_config = config.ModelConfig()
    training_config = config.TrainingConfig()

    # 加载数据
    train_path = os.path.join(DATA_ROOT, 'train.txt')
    valid_path = os.path.join(DATA_ROOT, 'valid.txt')
    test_path = os.path.join(DATA_ROOT, 'test.txt')
    x_train, y_train = reader.load_data_and_labels(train_path)
    x_valid, y_valid = reader.load_data_and_labels(valid_path)
    x_test, y_test = reader.load_data_and_labels(test_path)
    print(x_train.shape,y_train.shape)
    print(x_valid.shape,y_valid.shape)
    print(x_test.shape,y_test.shape)

