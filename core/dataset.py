import json

import tensorflow as tf
import pathlib
import random
import numpy as np

from core.utils import text2vec, preprocess_img

with open("config.json", 'r') as f:
    config = json.load(f)


class ImageDataSet:
    def __init__(self):
        self.max_captcha = config['max_length']
        self.char_set = config['char_set']
        self.char_set_len = len(self.char_set)
        self.train_batch_size = config['train_batch_size']
        self.valid_batch_size = config['test_batch_size']

        data_root = pathlib.Path(config['train_img_dir'])
        image_paths = list(data_root.glob("*"))
        image_paths = [str(path) for path in image_paths]
        random.shuffle(image_paths)

        # image_paths = image_paths[:int(0.07 * len(image_paths))]

        self.train_image_paths = image_paths[:int(0.9 * len(image_paths))]
        self.train_image_count = len(self.train_image_paths)

        self.valid_image_paths = image_paths[int(0.9 * len(image_paths)):]
        self.valid_image_count = len(self.valid_image_paths)

    def build(self):
        train_image_ds = tf.data.Dataset.from_tensor_slices(self.train_image_paths)
        train_image_ds = train_image_ds.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_all_images_labels = [text2vec(pathlib.Path(path).name.split("_")[0]) for path in self.train_image_paths]
        train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_all_images_labels, tf.int32))

        train_set = tf.data.Dataset.zip((train_image_ds, train_label_ds))
        train_set = train_set.shuffle(1000).batch(self.train_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        valid_image_ds = tf.data.Dataset.from_tensor_slices(self.valid_image_paths)
        valid_image_ds = valid_image_ds.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        valid_all_images_labels = [text2vec(pathlib.Path(path).name.split("_")[0]) for path in self.valid_image_paths]
        valid_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(valid_all_images_labels, tf.int32))

        valid_set = tf.data.Dataset.zip((valid_image_ds, valid_label_ds))
        valid_set = valid_set.shuffle(1000).batch(self.valid_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE).repeat()
        return train_set, valid_set

    def preprocess_img_and_label(self, path, label):
        return self.preprocess_img(path), label
    #
    # def preprocess_img(self, path):
    #     # 图片预处理，归一化、灰度化
    #     image = tf.io.read_file(path)
    #     image = tf.image.decode_png(image, channels=3)
    #     image = tf.image.resize(image, [config['image_height'], config['image_width']])
    #     image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    #     if len(image.shape) > 2:
    #         r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    #         image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    #     image = tf.expand_dims(image, axis=2)
    #     return image
    #
    # def text2vec(self, text):
    #     # 标签处理
    #     if len(text) > self.max_captcha:
    #         print(text)
    #         raise ValueError('{}'.format(self.max_captcha))
    #
    #     vector = np.zeros(self.max_captcha * self.char_set_len)
    #
    #     for i, ch in enumerate(text):
    #         idx = i * self.char_set_len + self.char_set.index(ch)
    #         vector[idx] = 1
    #     return vector.reshape([self.max_captcha, self.char_set_len])

