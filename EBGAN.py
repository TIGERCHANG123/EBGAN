from EBGAN_Block import *

class generator(tf.keras.Model):
  def __init__(self):
    super(generator, self).__init__()
    self.input_layer = generator_Input(shape=[7, 7, 256])

    self.middle_layer_list = [
      # generator_Middle(filters=512, strides=2),
      # generator_Middle(filters=256, strides=1),
      generator_Middle(filters=128, strides=2),
      generator_Middle(filters=64, strides=2)
    ]

    self.output_layer = generator_Output(image_depth=1, strides=1)
  def call(self, x):
    x = self.input_layer(x)
    # x = self.middle_layer1(x)
    for i in range(len(self.middle_layer_list)):
      x = self.middle_layer_list[i](x)
    x = self.output_layer(x)
    return x

class discriminator(tf.keras.Model):
  def __init__(self):
    super(discriminator, self).__init__()
    self.encoder_layer_list = [
      Encoder(filters=64, strides=1),
      Encoder(filters=128, strides=2),
      Encoder(filters=256, strides=2),
    ]
    self.flatten = tf.keras.layers.Flatten()
    self.encoder_dense = tf.keras.layers.Dense(128)
    self.decoder_dense = tf.keras.layers.Dense(128)
    self.reshape = tf.keras.layers.Reshape([7, 7, 256])
    self.decoder_layer_list = [
      Decoder(filters=256, strides=2),
      Decoder(filters=128, strides=2),
      Decoder(filters=3, strides=1),
    ]

  def call(self, x):
    for i in range(len(self.encoder_layer_list)):
      x = self.encoder_layer_list[i](x)
    x = self.flatten(x)
    embedding = self.encoder_dense(x)
    x = self.decoder_dense(embedding)
    for i in range(len(self.decoder_layer_list)):
      x = self.decoder_layer_list[i](x)
    return x, embedding

def get_gan():
  Generator = generator()
  Discriminator = discriminator()
  gen_name = 'EBGAN'
  return Generator, Discriminator, gen_name


