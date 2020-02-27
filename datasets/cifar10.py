from __future__ import print_function
import tensorflow_datasets as tfds
import tensorflow as tf

class mnist_dataset():
    def __init__(self, root, noise_dim):
        file_path = root + '/datasets/tensorflow_datasets'
        mnist, meta = tfds.load('cifar10', data_dir=file_path, download=False, as_supervised=True, with_info=True)
        print(meta)
        self.train_dataset=mnist['train']
        self.noise_dim = noise_dim
        self.name = 'cifar10'
        return
    def parse(self, x, y):
        x=tf.cast(x, tf.float32)
        # x = tf.expand_dims(x, -1)
        x=x/255*2 - 1
        y = tf.cast(y, tf.int64)
        return x, y
    def get_train_dataset(self):
        train_dataset = self.train_dataset.map(self.parse).shuffle(60000).batch(128)
        return train_dataset
