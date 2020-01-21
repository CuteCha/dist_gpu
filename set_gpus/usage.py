# -*- coding:utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sklearn
import sys
import time
import tensorflow as tf

from tensorflow import keras


def build_model0():
    model = keras.Sequential()
    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加全连接层
    # 先flatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='selu'))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",  # 损失函数
                  optimizer="sgd",  # 优化器名
                  metrics=["accuracy"])

    model.summary()

    return model


def build_model1():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = keras.Sequential()
        # 添加两个卷积层一个pooling层
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu', input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'))
        model.add(keras.layers.MaxPool2D(pool_size=2))

        # 添加两个卷积层一个pooling层
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
        model.add(keras.layers.MaxPool2D(pool_size=2))

        # 添加两个卷积层一个pooling层
        model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
        model.add(keras.layers.MaxPool2D(pool_size=2))

        # 添加全连接层
        # 先flatten
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='selu'))
        model.add(keras.layers.Dense(10, activation="softmax"))

        model.compile(loss="sparse_categorical_crossentropy",  # 损失函数
                      optimizer="sgd",  # 优化器名
                      metrics=["accuracy"])

        model.summary()

    return model


def build_model2():
    model = keras.Sequential()
    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加全连接层
    # 先flatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='selu'))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",  # 损失函数
                  optimizer="sgd",  # 优化器名
                  metrics=["accuracy"])

    model.summary()

    estimator = keras.estimator.model_to_estimator(model)

    return estimator


def build_model2():
    model = keras.Sequential()
    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu', input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加两个卷积层一个pooling层
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(keras.layers.MaxPool2D(pool_size=2))

    # 添加全连接层
    # 先flatten
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='selu'))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",  # 损失函数
                  optimizer="sgd",  # 优化器名
                  metrics=["accuracy"])

    model.summary()

    strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(train_distribute=strategy)
    estimator = keras.estimator.model_to_estimator(model, config=config)

    return estimator


def set_gpu():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    tf.config.experimental.set_visible_devices(devices=gpus[-4:], device_type='GPU')
    logical_gpus = tf.config.experimental.list_logical_devices(device_type='GPU')

    tf.config.set_soft_device_placement(True)

    return len(logical_gpus)


def run():
    set_gpu()


if __name__ == '__main__':
    run()
