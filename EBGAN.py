from EBGAN_Block import *

class generator(tf.keras.Model):
  def __init__(self):
    super(generator, self).__init__()
    self.input_layer = generator_Input(shape=[4, 4, 512])

    self.middle_layer_list = [
      # generator_Middle(filters=512, strides=2),
      generator_Middle(filters=256, strides=2),
      generator_Middle(filters=128, strides=2),
      generator_Middle(filters=64, strides=2)
    ]

    self.output_layer = generator_Output(image_depth=3, strides=1)
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
      Encoder(filters=64, strides=2),
      Encoder(filters=128, strides=2),
      Encoder(filters=256, strides=2),
    ]
    self.encoder_embedding = Encoder_embedding(128, [4, 4, 256])

    self.decoder_layer_list = [
      Decoder(filters=256, strides=2),
      Decoder(filters=128, strides=2),
      Decoder(filters=64, strides=2),
    ]
    self.output_layer = tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same")

  def call(self, x):
    for i in range(len(self.encoder_layer_list)):
      x = self.encoder_layer_list[i](x)
    x, embedding = self.encoder_embedding(x)
    for i in range(len(self.decoder_layer_list)):
      x = self.decoder_layer_list[i](x)
    x = self.output_layer(x)
    return x, embedding

def get_gan():
  Generator = generator()
  Discriminator = discriminator()
  gen_name = 'EBGAN'
  return Generator, Discriminator, gen_name


