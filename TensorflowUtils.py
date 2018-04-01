# Utils used with tensorflow implemetation
import tensorflow as tf
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io
import numpy as np

RESIZED_IMAGE_HEIGHT = 1944
RESIZED_IMAGE_WIDTH = 2592

def argmax_threshold(input, threshold):
    threshold = tf.constant(threshold)
    input = tf.nn.softmax(input)
    feature_0 = input[:, :, :, 1]
    feature_0 = tf.expand_dims(feature_0,3)
    result = tf.cast(tf.greater(feature_0, threshold), tf.float32)
    return result

def argmax_pre_threshold(input):
    input = tf.nn.softmax(input)
    feature_0 = input[:, :, :, 1]
    feature_0 = tf.expand_dims(feature_0,3)
    feature_0 = tf.cast(feature_0, tf.float32)
    return feature_0

def argmax_pre_threshold_resize(input):
    input = tf.nn.softmax(input)
    feature_0 = input[:, :, :, 1]
    feature_0 = tf.expand_dims(feature_0,3)
    feature_0 = tf.cast(feature_0, tf.float32)
    resize_feature_0 = tf.image.resize_images(feature_0, [RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH])
    return feature_0, resize_feature_0

def apply_threshold(input, threshold):
    threshold = tf.constant(threshold)
    result = tf.cast(tf.greater(input, threshold), tf.float32)
    return result

def generate_image_mask_list(mode):
    images_name = []
    masks_name = []

    if mode == "Train":
        log_path = "./Eye_picture/Training/log_training.txt"
        image_path = "./Eye_picture/Training/image/"
        mask_path = "./Eye_picture/Training/mask_2class/"
    elif mode == "Test":
        log_path = "./Eye_picture/Test/log_test.txt"
        image_path = "./Eye_picture/Test/image/"
        mask_path = "./Eye_picture/Test/mask_2class/"
    else:
        print("Mode parameter is wrong !")
        log_path=""
        image_path=""
        mask_path=""

    file = open(log_path, 'r')
    lines = file.readlines()

    for line in lines:
        print(line.strip('\n'))
        image_name, mask_name = line.split(' ')
        mask_name = mask_name.strip('\n')
        images_name.append(image_path + image_name)
        masks_name.append(mask_path + mask_name)

    image_list = tf.convert_to_tensor(images_name, dtype=tf.string)
    mask_list = tf.convert_to_tensor(masks_name, dtype=tf.string)

    return image_list, mask_list

def generate_image_mask_list_practical():
    images_name = []

    log_path = "./Practical_Eye_image/log.txt"
    image_path = "./Practical_Eye_image/"

    file = open(log_path, 'r')
    lines = file.readlines()

    for line in lines:
        print(line.strip('\n'))
        image_name = line.strip('\n')
        images_name.append(image_path + image_name)

    image_list = tf.convert_to_tensor(images_name, dtype=tf.string)

    return image_list


def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def save_image(image, save_dir, name, mean=None, category = False):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    image = np.array(image, dtype = np.uint8)
    if mean:
        image = unprocess_image(image, mean)
    if category == True:
        image = image * 255
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var

def weight_variable(shape, name=None, init=None):
    # print(shape)
    if init == "Truncated_Normal":
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
    if init == "Xavier":
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride = 2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x):
    return tf.layers.batch_normalization(x)


def process_image(image, mean_pixel):
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    return image + mean_pixel


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))
