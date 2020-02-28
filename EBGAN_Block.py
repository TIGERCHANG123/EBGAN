import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class generator_Input(tf.keras.Model):
  def __init__(self, shape):
    super(generator_Input, self).__init__()
    self.dense = layers.Dense(shape[0] * shape[1] * shape[2], use_bias=False)
    self.reshape = layers.Reshape(shape)
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.relu = tf.keras.layers.ReLU()
  def call(self, x):
    x = self.dense(x)
    x = self.reshape(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class generator_Middle(tf.keras.Model):
  def __init__(self, filters, strides):
      super(generator_Middle, self).__init__()
      self.conv = layers.Conv2DTranspose(filters, (5, 5), strides=strides, padding='same', use_bias=False)
      self.bn = layers.BatchNormalization(momentum=0.9)
      self.relu = tf.keras.layers.ReLU()
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class generator_Output(tf.keras.Model):
  def __init__(self, image_depth, strides):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2DTranspose(image_depth, (5, 5), strides=strides, padding='same', use_bias=False)
    self.actv = layers.Activation(activation='tanh')
  def call(self, x):
    x = self.conv(x)
    x = self.actv(x)
    return x

class Encoder(tf.keras.Model):
  def __init__(self, filters, strides):
      super(Encoder, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same")
      self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
      self.Relu = tf.keras.layers.ReLU()
      self.dropout = tf.keras.layers.Dropout(0.3)

  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.Relu(x)
      x = self.dropout(x)
      return x

class Encoder_embedding(tf.keras.Model):
    def __init__(self, hidden_unit, shape):
        super(Encoder_embedding, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.encoder_dense = tf.keras.layers.Dense(128)
        self.decoder_dense = tf.keras.layers.Dense(shape[0] * shape[1] * shape[2])
        self.reshape = tf.keras.layers.Reshape(shape)

    def call(self, x):
        x = self.flatten(x)
        embedding = self.encoder_dense(x)
        x = self.decoder_dense(embedding)
        x = self.reshape(x)
        return x, embedding

class Decoder(tf.keras.Model):
    def __init__(self, filters, strides):
        super(Decoder, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(filters, kernel_size=5, strides=strides, padding="same")
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.Relu = tf.keras.layers.ReLU()
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.Relu(x)
        x = self.dropout(x)
        return x


