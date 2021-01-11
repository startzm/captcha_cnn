import json
import os

import tensorflow as tf

from core.model import CNN
from core.utils import vec2text

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 默认使用第0个GPU设备
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Predict:
    # 传入图片预测结果
    def __init__(self):
        self.model_save_dir = config['model_save_dir'] + '/model_weight'
        self.model = CNN(config['max_length'], len(config['char_set']))
        try:
            tf.print("尝试加载模型文件..")
            self.model.load_weights(self.model_save_dir)
            tf.print("加载模型成功")
        except:
            tf.print("未读取到模型文件..")

    def pred_from_path(self, path):
        # 以路径形式传入图片识别
        image = tf.io.read_file(path)
        image = self.preprocess_img(image)
        pred = self.model(image)
        label = vec2text(pred)
        return label

    def pred_from_bytes(self, image):
        # 以二进制流形式传入图片识别
        image = tf.convert_to_tensor(image)
        image = self.preprocess_img(image)
        pred = self.model(image)
        label = vec2text(pred)
        return label

    def preprocess_img(self, image):
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [config['image_height'], config['image_width']])
        image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
        if len(image.shape) > 2:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        image = tf.expand_dims(image, axis=0)
        image = tf.expand_dims(image, axis=3)
        return image


if __name__ == '__main__':
    # 加载配置文件
    with open("config.json", 'r') as f:
        config = json.load(f)
        f.close()

    # 以路径形式传入图片预测
    # label = Predict().pred_from_path("E:\\cnn_captcha-master\sample\\train\\xhuw_1605837031398.png")
    # print("预测结果为：", label)

    # 以二进制流形式传入图片预测
    image = open("captcha.png", 'rb').read()
    label = Predict().pred_from_bytes(image)
    print("预测结果为：", label)