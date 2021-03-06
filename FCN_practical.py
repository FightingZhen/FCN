import FCN as FCN4
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

args = parser.parse_args()

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

os.environ["CUDA_DIVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

sess = tf.InteractiveSession()

image_list = utils.generate_image_mask_list_practical()

weight_init = args.WEIGHT_INIT
model_name = args.MODEL_NAME
logs_dir = './' + model_name + "/logs/"
model_dir = "./Model/"
saved_dir = './' + model_name + "/saved_model/"
threshold = args.threshold
test_saved_dir = './' + model_name + "/test_practical/" + str(threshold) + '/'

pre_threshold = False
pre_threshold_dir = test_saved_dir + 'h5_pre_threshold/'

test_batch_size = 1
test_batch_num = 85

NUM_OF_CLASSESS = 2
IMAGE_SIZE_L = 800
IMAGE_SIZE_W = 800

# print(sess.run(image_list))

print("WEIGHT_INIT:" + weight_init)
print("MODEL_NAME:" + model_name)
print("LOGS_DIR:" + logs_dir)
print("MODEL_DIR:" + model_dir)
print("SAVED_DIR:" + saved_dir)
print("TEST_SAVED_DIR:" + test_saved_dir)
print("TEST_BATCH_SIZE:" + str(test_batch_size))
print("TEST_BATCH_NUM:" + str(test_batch_num))
print("MAX_OF_CLASS:" + str(NUM_OF_CLASSESS))
print("IMAGE_SIZE:" + str(IMAGE_SIZE_L) + '*' + str(IMAGE_SIZE_W))
print("GPU:" + args.GPU)
print("Threshold:" + str(threshold))

filename_queue = tf.train.slice_input_producer([image_list], shuffle=False)

image_content = tf.read_file(filename_queue[0])

img = tf.image.decode_png(image_content, channels=3)
img = tf.cast(img, tf.float32)
img = tf.reshape(img, [IMAGE_SIZE_W, IMAGE_SIZE_L, 3])

print("Data sets processe finished !")

def main(argv=None):
    batch_size = test_batch_size
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image_batch, img_name = tf.train.batch([img,filename_queue[0]], batch_size)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    logits = FCN4.inference(image_batch, keep_probability)

    if pre_threshold:
        pred_annotation = utils.argmax_pre_threshold(logits)
        pred_annotation = tf.reshape(pred_annotation, [batch_size, IMAGE_SIZE_W, IMAGE_SIZE_L, 1])
    else:
        pred_annotation = utils.argmax_threshold(logits, threshold)
        pred_annotation = tf.reshape(pred_annotation, [batch_size, IMAGE_SIZE_W, IMAGE_SIZE_L, 1])

    print("Setting up Saver...")
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(saved_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if not os.path.isdir(test_saved_dir):
        os.makedirs(test_saved_dir)

    for counter in range(test_batch_num):
        pred, image_test, image_name = sess.run([pred_annotation, image_batch, img_name],
                                               feed_dict={keep_probability: 1.0})

        image_name = image_name[0].decode()
        image_name = image_name.split('/')[-1].strip('.png')

        image_test = np.reshape(image_test, [IMAGE_SIZE_W, IMAGE_SIZE_L, 3])
        pred = np.squeeze(pred, axis=3)
        pred = np.reshape(pred, [IMAGE_SIZE_W, IMAGE_SIZE_L])

        if pre_threshold:
            if not os.path.isdir(pre_threshold_dir):
                os.makedirs(pre_threshold_dir)
            f = h5.File(pre_threshold_dir + image_name + '.h5', 'w')
            f['data'] = pred
            f.close()
            pred = utils.apply_threshold(pred, threshold)

        utils.save_image(image_test, test_saved_dir + '' 'image/',
                         name=image_name + "_image")
        utils.save_image(pred, test_saved_dir + 'pred/',
                         name=image_name + "_pred", category=True)
        print("Saved no. %s image: %s" % (counter, image_name + '.png'))

    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
