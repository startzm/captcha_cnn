import json

import tensorflow as tf
import numpy as np

with open("config.json", 'r') as f:
    config = json.load(f)

max_captcha = config['max_length']
char_set = config['char_set']
char_set_len = len(char_set)


class CharAcc(tf.keras.metrics.Metric):
    # 计算字符准确率
    def __init__(self, name="charAcc", **kwargs):
        super(CharAcc, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred):
        y_pred = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.int32)
        y_true = tf.cast(tf.argmax(y_true, axis=2), dtype=tf.int32)
        self.total.assign_add(tf.reshape(y_true, shape=[-1]).shape[0])
        self.count.assign_add(tf.reduce_sum(tf.cast(tf.equal(y_pred, y_true), dtype=tf.int32)))

    def result(self):
        return self.count / self.total


class ImgAcc(tf.keras.metrics.Metric):
    # 计算图片准确率
    def __init__(self, name="imgAcc", **kwargs):
        super(ImgAcc, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred):
        y_pred = tf.cast(tf.argmax(y_pred, axis=2), dtype=tf.int32)
        y_true = tf.cast(tf.argmax(y_true, axis=2), dtype=tf.int32)
        self.total.assign_add(y_true.shape[0])
        self.count.assign_add(tf.reduce_sum(tf.cast(tf.reduce_all(tf.equal(y_pred, y_true), axis=1), dtype=tf.int32)))

    def result(self):
        return self.count / self.total


def vec2text(pred):
    pred = np.array(tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32)).tolist()
    label = ''
    for temp in pred:
        for char in temp:
            label += char_set[char]
    return label


def text2vec(text):
    if len(text) > max_captcha:
        print(text)
        raise ValueError('{}'.format(max_captcha))
    vector = np.zeros(max_captcha * char_set_len)
    for i, ch in enumerate(text):
        idx = i * char_set_len + char_set.index(ch)
        vector[idx] = 1
    return vector.reshape([max_captcha, char_set_len])


def preprocess_img(path):
    # 图片预处理
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [config['image_height'], config['image_width']])
    image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    if len(image.shape) > 2:
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        image = 0.2989 * r + 0.5870 * g + 0.1140 * b
    image = tf.expand_dims(image, axis=2)
    return image