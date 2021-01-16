import datetime
import json
import os

import tensorflow as tf
from tensorflow.keras import metrics
import numpy as np

from core.dataset import ImageDataSet
from core.model import CNN
from core.utils import CharAcc, ImgAcc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Train:
    def __init__(self):
        # 初始化训练集、验证集
        self.train_set, self.valid_set = ImageDataSet().build()

        # 初始化模型
        self.model_save_dir = config['model_save_dir'] + '/model_weight'
        self.model = CNN(config['max_length'], len(config['char_set']))
        try:
            tf.print("尝试加载模型文件..")
            self.model.load_weights(self.model_save_dir)
            tf.print("加载模型成功")
        except:
            tf.print("未读取到模型文件..")
        # 初始化优化器
        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.train_loss = metrics.Mean(name="train_loss")
        self.train_char_metric = CharAcc(name="train_char_accuracy")
        self.train_img_metric = ImgAcc(name="train_img_accuracy")

        self.valid_loss = metrics.Mean(name='valid_loss')
        self.valid_char_metric = CharAcc(name="valid_char_metric")
        self.valid_img_metric = ImgAcc(name="valid_img_metric")

        self.val_step = iter(self.valid_set)

        self.loss_list = [0]
        self.less_loss = 999
        self.wait = 0
        # 初始化tensorboard
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(os.path.join("logs/"+stamp))

    def train_model(self, epoches):
        step = 0
        for epoch in range(1, epoches + 1):
            for x, y in self.train_set:
                with tf.GradientTape() as tape:
                    logits = self.model(x)
                    loss = tf.losses.categorical_crossentropy(y, logits)
                    loss = tf.reduce_mean(loss)
                # 每个step结束后原地更新梯度
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                if step % config["cycle_save"] == 0:
                    # 每训练达指定次数 自动保存模型
                    self.model.save_weights(self.model_save_dir)
                    tf.print("保存模型成功")
                    # 降低学习率
                    # self.optimizer.learning_rate = self.optimizer.learning_rate / 2

                if step % 10 == 0:
                    # 每训练10次 计算准确率并输入
                    self.train_char_metric.update_state(y, logits)
                    self.train_img_metric.update_state(y, logits)

                    self.valid_step(next(self.val_step))

                    train_logs = '训练次数:{} \n训练集: Loss:{} 字符准确率: {}, 图片准确率: {}\n' \
                                 '验证集: Loss:{}, 字符准确率:{}, 图片准确率:{}\n'
                    tf.print("=========================================================")
                    tf.print(tf.strings.format(train_logs,
                                               (step, float(loss), self.train_char_metric.result(),
                                                self.train_img_metric.result(), self.valid_loss.result(),
                                                self.valid_char_metric.result(), self.valid_img_metric.result())))
                    self.write_summary(step, loss)

                    if float(self.train_char_metric.result()) >= float(config['acc_stop']):
                        tf.print("已经达到指定准确率，退出训练")
                        self.model.save_weights(self.model_save_dir)
                        tf.print("保存模型成功")
                        break
                    if self.early_stop(float(loss)):
                        tf.print("训练集loss已经不再提升，退出训练")
                        self.model.save_weights(self.model_save_dir)
                        tf.print("保存模型成功")
                        break

                    self.valid_loss.reset_states()
                    self.train_char_metric.reset_states()
                    self.valid_char_metric.reset_states()
                    self.train_img_metric.reset_states()
                    self.valid_img_metric.reset_states()

                step += 1
            else:
                continue
            break

    @tf.function
    def valid_step(self, data):
        features, labels = data
        predictions = self.model(features, training=False)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(labels, predictions))

        self.valid_char_metric.update_state(labels, predictions)
        self.valid_img_metric.update_state(labels, predictions)
        self.valid_loss.update_state(loss)

    def early_stop(self, loss):
        # 提前停止训练
        if len(self.loss_list) > 10:
            self.loss_list[1:].append(loss - self.loss_list[-1])
        else:
            self.loss_list.append(loss - self.loss_list[-1])
        # 10轮以上loss更新均值低于loss_stop停止训练
        if abs(np.mean(self.loss_list)) < config['loss_stop']:
            return True
        # 10轮以上loss值不再降低则停止训练
        if self.less_loss != min(self.loss_list):
            # 临时保存最佳权重
            self.less_loss = loss
            self.best_weight = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > 10:
                self.model.set_weights(self.best_weight)
                return True
        return False

    def write_summary(self, step, train_loss):
        # 写入tensorboard
        with self.summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=step)
            tf.summary.scalar('train_char_accuracy', self.train_char_metric.result(), step=step)
            tf.summary.scalar('train_img_accuracy', self.train_img_metric.result(), step=step)

            tf.summary.scalar('valid_loss', self.valid_loss.result(), step=step)
            tf.summary.scalar('valid_char_accuracy', self.valid_char_metric.result(), step=step)
            tf.summary.scalar('valid_img_accuracy', self.valid_img_metric.result(), step=step)


if __name__ == '__main__':
    # 默认使用第0个GPU设备
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 加载配置文件
    with open("config.json", 'r') as f:
        config = json.load(f)

    # 训练
    Train().train_model(config['max_epochs'])

