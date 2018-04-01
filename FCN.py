import tensorflow as tf
import numpy as np
import TensorflowUtils as utils
import os
import argparse
import h5py as h5

parser = argparse.ArgumentParser()

parser.add_argument('--MODEL_NAME')
parser.add_argument('--GPU')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--WEIGHT_INIT', default="Xavier")  # Truncated_Normal  or  Xavier
parser.add_argument('--TRAINING_BATCH_SIZE', type=int, default=4)
parser.add_argument('--LEARNING_RATE', type=float, default=1e-4)
parser.add_argument('--mode', default='Train')
parser.add_argument('--debug', default=True)

args = parser.parse_args()

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

sess = tf.InteractiveSession()

debug = args.debug
mode = args.mode

image_list, mask_list = utils.generate_image_mask_list(mode)

weight_init = args.WEIGHT_INIT
model_name = args.MODEL_NAME
threshold = args.threshold

logs_dir = './' + model_name + "/logs/"
model_dir = "./Model/"
saved_dir = './' + model_name + "/saved_model/"
test_saved_dir = './' + model_name + "/test_trainingPeriod/"

training_batch_size = args.TRAINING_BATCH_SIZE
start_learning_rate = args.LEARNING_RATE
test_batch_size = 1
test_batch_num = 20
epoch = 300

len_image = sess.run(tf.shape(image_list))[0]
iteration_epoch = int(len_image / training_batch_size) + 1
MAX_ITERATION = int(iteration_epoch * epoch) + 1

NUM_OF_CLASSESS = 2
IMAGE_SIZE_L = 800
IMAGE_SIZE_W = 800

pre_threshold = False
pre_threshold_resize = False

pre_threshold_dir = test_saved_dir + 'h5_pre_threshold/'
pre_threshold_resize_dir = test_saved_dir + 'h5_pre_threshold_resize/'

# print(sess.run(image_list))
# print(sess.run(mask_list))

print("MODE:" + mode)
print("DEBUG:" + str(debug))
print("WEIGHT_INIT:" + weight_init)
print("MODEL_NAME:" + model_name)
print("LOGS_DIR:" + logs_dir)
print("MODEL_DIR:" + model_dir)
print("SAVED_DIR:" + saved_dir)
print("TEST_SAVED_DIR:" + test_saved_dir)
print("TRAINING_BATCH_SIZE:" + str(training_batch_size))
print("START_LEARNING_RATE:" + str(start_learning_rate))
print("TEST_BATCH_SIZE:" + str(test_batch_size))
print("TEST_BATCH_NUM:" + str(test_batch_num))
print("MAX_OF_CLASS:" + str(NUM_OF_CLASSESS))
print("IMAGE_SIZE:" + str(IMAGE_SIZE_L) + '*' + str(IMAGE_SIZE_W))
print("GPU:" + args.GPU)
print("Threshold:" + str(threshold))

with tf.name_scope('Learning_Rate'):
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, iteration_epoch, 0.98,
                                               staircase=True, name='learning_rate')
    tf.summary.scalar(" ", learning_rate)

if not os.path.isdir('./' + model_name):
    os.makedirs('./' + model_name)

filename_queue = tf.train.slice_input_producer([image_list, mask_list])

image_content = tf.read_file(filename_queue[0])
mask_content = tf.read_file(filename_queue[1])

img = tf.image.decode_png(image_content, channels=3)
img = tf.cast(img, tf.float32)
img = tf.reshape(img, [IMAGE_SIZE_W, IMAGE_SIZE_L, 3])

mask = tf.image.decode_png(mask_content, channels=1)
mask = tf.cast(mask, tf.int32)
mask = tf.reshape(mask, [IMAGE_SIZE_W, IMAGE_SIZE_L, 1])

print("Data sets processe finished !")

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv4_3"]

        W6 = utils.weight_variable([1, 1, 512, 1024], name="W6", init=weight_init)
        b6 = utils.bias_variable([1024], name="b6")

        conv6 = utils.conv2d_basic(conv_final_layer, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 1024, NUM_OF_CLASSESS], name="W7", init=weight_init)
        b7 = utils.bias_variable([NUM_OF_CLASSESS], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        
        # now to upscale to actual image size
        deconv_shape1 = image_net["pool2"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1", init=weight_init)
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv7, W_t1, b_t1, output_shape=tf.shape(image_net["pool2"]))
        fuse_1 = tf.add(conv_t1, image_net["pool2"], name="fuse_1")

        deconv_shape2 = image_net["pool1"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2", init=weight_init)
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool1"]))
        fuse_2 = tf.add(conv_t2, image_net["pool1"], name="fuse_1")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([4, 4, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3", init=weight_init)
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=2)

    return conv_t3


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")

    if mode == "Train":
        batch_size = training_batch_size
    elif mode == "Test":
        batch_size = test_batch_size

    image_batch, mask_batch, image_name= tf.train.batch([img, mask, filename_queue[0]], batch_size)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    logits = inference(image_batch, keep_probability)

    pred_annotation = utils.argmax_threshold(logits, threshold)
    pred_annotation = tf.reshape(pred_annotation, [batch_size, IMAGE_SIZE_W, IMAGE_SIZE_L, 1])
    logits = tf.reshape(logits, [batch_size, IMAGE_SIZE_W, IMAGE_SIZE_L, 2])

    if mode == 'Train':
        tf.summary.image("input_image", image_batch, max_outputs=10)
        tf.summary.image("ground_truth", tf.cast(mask_batch, tf.uint8) * 255, max_outputs=10)
        tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8) * 255, max_outputs=10)

        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=tf.squeeze(mask_batch,
                                                                                                squeeze_dims=[3]),
                                                                              name="entropy")))
        tf.summary.scalar("entropy", loss)

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    if mode == "Train":
        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()
    else:
        summary_op = None

    saver = tf.train.Saver()
    if mode == "Train":
        print("Setting up Saver...")
        summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    else:
        summary_writer = None

    sess.run(tf.global_variables_initializer())

    if mode == "Test":
        ckpt = tf.train.get_checkpoint_state(saved_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored...")

    if mode == "Train":
        for itr in range(MAX_ITERATION):
            feed_dict = {keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str, lr = sess.run([loss, summary_op, learning_rate], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g, learning_rate: %g" % (itr, train_loss, lr))
                summary_writer.add_summary(summary_str, itr)

                saver.save(sess, saved_dir + 'model.ckpt', global_step=global_step)

    elif mode == "Test":
        if not os.path.isdir(test_saved_dir):
            os.makedirs(test_saved_dir)

        for counter in range(test_batch_num):
            pred, image_test, mask_test, img_name = sess.run([pred_annotation, image_batch, mask_batch, image_name],
                                                   feed_dict={keep_probability: 1.0})

            img_name = img_name[0].decode()
            img_name = img_name.split('/')[-1].strip('.png')
            print(img_name)

            mask_test = np.squeeze(mask_test, axis=3)
            pred = np.squeeze(pred, axis=3)

            image_test = np.reshape(image_test, [IMAGE_SIZE_W, IMAGE_SIZE_L, 3])
            pred = np.reshape(pred, [IMAGE_SIZE_W, IMAGE_SIZE_L])
            mask_test = np.reshape(mask_test, [IMAGE_SIZE_W, IMAGE_SIZE_L])

            utils.save_image(image_test, test_saved_dir + 'image/',
                             name=img_name + "_image")
            utils.save_image(mask_test, test_saved_dir + 'mask/',
                             name=img_name + "_mask", category=True)
            utils.save_image(pred, test_saved_dir + 'pred',
                             name=img_name + "_pred", category=True)
            print("Saved image: %s" % (img_name + '.png'))

    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
