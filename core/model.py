import tensorflow as tf
from tensorflow.keras import Model, layers


class CNN(Model):
    def __init__(self, max_captcha, char_set_len, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.conv2 = layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.conv3 = layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.conv4 = layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        self.bn4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation=tf.nn.relu)
        self.drop1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(max_captcha*char_set_len, activation=None)
        self.drop2 = layers.Dropout(0.5)
        self.reshape = layers.Reshape([max_captcha, char_set_len])
        self.out = layers.Softmax()

    @tf.function
    def call(self, inputs, training=None, mask=None):
        model = self.conv1(inputs)
        model = self.bn1(model)
        model = self.pool1(model)

        model = self.conv2(model)
        model = self.bn2(model)
        model = self.pool2(model)

        model = self.conv3(model)
        model = self.bn3(model)
        model = self.pool3(model)

        model = self.conv4(model)
        model = self.bn4(model)
        model = self.pool4(model)

        model = self.flatten(model)
        model = self.dense1(model)
        model = self.drop1(model)
        model = self.dense2(model)
        model = self.drop2(model)
        model = self.reshape(model)
        model = self.out(model)

        return model
