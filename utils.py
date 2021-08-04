import tensorflow as tf
from matplotlib import pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256
# SCALE = 12

def load(image_file):
    input_image = tf.io.read_file(image_file)
    # input_image = tf.image.decode_jpeg(input_image)
    input_image = tf.io.decode_png(input_image, channels=3)
    input_image = tf.cast(input_image, tf.float32)

    real_image = tf.io.read_file(image_file.replace('imgs', 'masks'))
    # real_image = tf.image.decode_jpeg(real_image)
    real_image = tf.io.decode_png(real_image, channels=3)
    real_image = tf.cast(real_image, tf.float32)
    # real_image = input_image

    return input_image, real_image 

def resize(input_image, real_image, height, width):
    # scl_height = int(IMG_HEIGHT/SCALE)
    # scl_width = int(IMG_WIDTH/SCALE)
    # input_image = tf.image.resize(input_image, [scl_height, scl_width],
    #                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # real_image = tf.image.resize(real_image, [scl_height, scl_width],
    #                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

# normalizing the images to [-1, 1]
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image

def load_image_train(image_file):
    image_file = bytes.decode(image_file.numpy())
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    
    return input_image, real_image

def load_image_test(image_file):
    image_file = bytes.decode(image_file.numpy())
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image

def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()
