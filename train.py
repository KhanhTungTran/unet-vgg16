import tensorflow as tf

import os
import time

from IPython import display
from tensorflow.keras import backend as K

import gc

from utils import *
from unet import *

# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Hyperparameters:
BUFFER_SIZE = 400
BATCH_SIZE = 1
EPOCHS = 50

INPUT_PATH = 'data2/train/imgs/'
MASK_PATH = 'data2/train/masks/'
TEST_INPUT_PATH = 'data2/test/imgs/'
TEST_MASK_PATH = 'data2/test/masks/'

# Load data:
train_dataset = tf.data.Dataset.list_files(INPUT_PATH+'*.jpg')
train_dataset = train_dataset.map(lambda x: tf.py_function(load_image_train, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(TEST_INPUT_PATH+'*.jpg')
test_dataset = test_dataset.map(lambda x: tf.py_function(load_image_test, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Create U-net model:
generator = Generator()
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Define VGG-16 model for Perceptual loss:
loss_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256,256,3))
loss_model.trainable=False
for layer in loss_model.layers:
    layer.trainable=False

loss_model = tf.keras.Model(loss_model.inputs, loss_model.layers[5].output)

def generator_loss(gen_output, target):
    channels = 128
    size = 128
    perceptual_loss = K.mean(K.sqrt(K.sum(K.square(loss_model(gen_output)-loss_model(target)))))/(channels*size*size)
    
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    total_gen_loss = LAMBDA*perceptual_loss + l1_loss
    
    return total_gen_loss, perceptual_loss, l1_loss

# Checkpoint for saving model:
checkpoint_dir = './training_checkpoints_update_unet'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 generator=generator)

import datetime
log_dir = 'logs/'

summary_writer = tf.summary.create_file_writer(
    log_dir + 'fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        gen_total_loss, gen_perceptual_loss, gen_l1_loss = generator_loss(gen_output, target)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_perceptual_loss', gen_perceptual_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)

def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()

        display.clear_output(wait=True)

        # for example_input, example_target in test_dataset.take(1):
        #     generate_images(generator, example_input, example_target)
        print('Epoch: ', epoch)

        # Train
        for n, (input_image, target) in train_ds.enumerate():
            # print('.', end='')
            if (n+1)%100==0:
                gc.collect()
                print(str(int(n)+1))
            train_step(input_image, target, epoch)
        print()

        # saving (checkpoint) the model every epochs
        checkpoint.save(file_prefix=checkpoint_prefix)
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

fit(train_dataset, EPOCHS, test_dataset)

for inp, tar in train_dataset.take(5):
    generate_images(generator, inp, tar)