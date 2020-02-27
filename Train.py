import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class train_one_epoch():
    def __init__(self, model, train_dataset, optimizers, metrics, noise_dim, margin):
        self.generator, self.discriminator = model
        self.generator_optimizer, self.discriminator_optimizer = optimizers
        self.gen_loss, self.disc_loss = metrics
        self.train_dataset = train_dataset
        self.noise_dim = noise_dim
        self.margin = margin

        self.fake_loss = 0
        self.real_loss = 0
        self.pt_loss = 0

    def pullaway_loss(self, embeddings, batch_size):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings)))
        normalized_embeddings = embeddings / norm
        similarity = tf.matmul(normalized_embeddings, normalized_embeddings, transpose_b=True)
        pt_loss = (tf.reduce_sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return pt_loss

    def get_loss(self, input, output):
        loss = tf.sqrt(tf.reduce_mean(tf.square(input-output)))
        return loss

    def train_step(self, noise, images):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output, _ = self.discriminator(images, training=True)
            fake_output, embedding = self.discriminator(generated_images, training=True)

            self.real_loss = self.get_loss(images, real_output)
            self.fake_loss = self.get_loss(generated_images, fake_output)
            self.pt_loss = self.pullaway_loss(embedding, images.shape[0])

            zero = tf.zeros_like(self.margin - self.fake_loss)
            disc_loss = self.real_loss + tf.maximum(zero, self.margin - self.fake_loss)#avoid cllapsing
            gen_loss = self.fake_loss + 0.1 * self.pt_loss
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.gen_loss(gen_loss)
        self.disc_loss(disc_loss)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, epoch,  pic):
        self.gen_loss.reset_states()
        self.disc_loss.reset_states()

        for (batch, (images, labels)) in enumerate(self.train_dataset):
            noise = tf.random.normal([images.shape[0], self.noise_dim])
            self.train_step(noise, images)
            pic.add([self.gen_loss.result().numpy(), self.disc_loss.result().numpy()])
            pic.save()
            if batch % 100 == 0:
                print('epoch: {}, gen loss: {}, disc loss: {}, real loss: {}, fake loss: {}, pt loss: {}'
                      .format(epoch, self.gen_loss.result(), self.disc_loss.result(), self.real_loss, self.fake_loss, self.pt_loss))