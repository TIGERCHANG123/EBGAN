# -*- coding:utf-8 -*-
import os
import getopt
import sys
import tensorflow as tf
from EBGAN import get_gan
from show_pic import draw
from Train import train_one_epoch
from datasets.cifar10 import mnist_dataset
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

ubuntu_root='/home/tigerc'
windows_root='D:/Automatic/SRTP/GAN'
root = '/content/drive/My Drive'
# root = ubuntu_root
temp_root = root+'/temp'
dataset_root = '/content'


def main(continue_train, train_time, train_epoch):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    noise_dim = 100
    batch_size = 128

    generator_model, discriminator_model, model_name = get_gan()
    dataset = mnist_dataset(dataset_root, batch_size)
    model_dataset = model_name + '-' + dataset.name

    train_dataset = dataset.get_train_dataset()
    pic = draw(10, temp_root, model_dataset, train_time=train_time)
    generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.5)

    print('continue train: {}\n train time: {}\n total epoch: {}\n, model: {}\n dataset: {}\n'
          .format(continue_train, train_time, train_epoch, model_name, dataset.name))
    checkpoint_path = temp_root + '/temp_model_save/' + model_dataset
    ckpt = tf.train.Checkpoint(genetator_optimizers=generator_optimizer, discriminator_optimizer=discriminator_optimizer ,
                               generator=generator_model, discriminator=discriminator_model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    if ckpt_manager.latest_checkpoint and continue_train:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    gen_loss = tf.keras.metrics.Mean(name='gen_loss')
    disc_loss = tf.keras.metrics.Mean(name='disc_loss')

    train = train_one_epoch(model=[generator_model, discriminator_model], train_dataset=train_dataset,
              optimizers=[generator_optimizer, discriminator_optimizer], metrics=[gen_loss, disc_loss], noise_dim=noise_dim, margin=20)

    for epoch in range(train_epoch):
        train.train(epoch=epoch, pic=pic)
        pic.show()
        if (epoch + 1) % 5 == 0:
            ckpt_manager.save()
        pic.save_created_pic(generator_model, 8, noise_dim, epoch)
    pic.show_created_pic(generator_model, 8, noise_dim)
    return

if __name__ == '__main__':
    continue_train = False
    train_time  = 0
    epoch = 500
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:v:")
        for op, value in opts:
            if op == "-continue":
                continue_train = True
            elif op == "-time":
                train_time = value
            elif op == "-epoch":
                epoch = value
    except:
        print('wrong input!')

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    main(continue_train=continue_train, train_time=train_time, train_epoch = epoch)