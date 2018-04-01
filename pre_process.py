from PIL import Image
import PIL
import os
import numpy as np

def rename_file():
    # dir = "D:/Tuned_Picture/Eye_picture/Training/mask/"
    # file_list = os.listdir(dir)
    # for file in file_list:
    #     os.rename(dir + file, dir + file.strip('.png') + '_mask.png')

    dir = "D:/Tuned_Picture/Eye_picture/Training/imag/"
    file_list = os.listdir(dir)
    for file in file_list:
        if file[9] == '_':
            os.rename(dir + file, dir + file[:9] + 'ng' + file[9:])


def change_picture_shape_and_type(shape, practical=False, type=None, origin_size=False):
    # 训练数据
    if not practical:
        image_dir = "D:/Source_Picture/Eye_picture/" + type + "/image/"
        image_out_dir = "D:/Tuned_Picture/Eye_picture/" + type + "/image/"
        mask_dir = "D:/Source_Picture/Eye_picture/" + type + "/mask/"
        mask_out_dir = "D:/Tuned_Picture/Eye_picture/" + type + "/mask/"
    # 真实数据
    if practical:
        image_dir = "D:/Source_Picture/Practical_Eye_image/"
        if not origin_size:
            image_out_dir = "D:/Tuned_Picture/Practical_Eye_image/"
        else:
            image_out_dir = "D:/Tuned_Picture/Practical_Origin_Eye_image/"

    if not os.path.isdir(image_out_dir):
        os.makedirs(image_out_dir)
    if not practical:
        if not os.path.isdir(mask_out_dir):
            os.makedirs(mask_out_dir)

    image_list = os.listdir(image_dir)
    if not practical:
        mask_list = os.listdir(mask_dir)
    if not origin_size:
        for file in image_list:
            img = Image.open(image_dir + file)
            img.resize((shape[0],shape[1]), resample=PIL.Image.BILINEAR).save(image_out_dir + file[:-3] + 'png')
    else:
        for file in image_list:
            img = Image.open(image_dir + file)
            img.resize((648,486), resample=PIL.Image.BILINEAR).save(image_out_dir + file[:-3] + 'png')

    if not practical:
        for file in mask_list:
            mask = Image.open(mask_dir + file)
            mask.convert("L").resize((shape[0],shape[1]), resample=PIL.Image.BILINEAR).save(mask_out_dir + file[:-3] + 'png')

def data_augumentation(rotation=False):
    training_image_dir = "D:/Tuned_Picture/Eye_picture/Training/image/"
    training_mask_dir = "D:/Tuned_Picture/Eye_picture/Training/mask_2class/"

    training_image_files = os.listdir(training_image_dir)
    training_mask_files = os.listdir(training_mask_dir)

    for image_file, mask_file in zip(training_image_files, training_mask_files):
        image = Image.open(training_image_dir + image_file)
        mask = Image.open(training_mask_dir + mask_file)
        LR_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        TB_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if rotation:
            ROT_image = image.rotate(45, resample=PIL.Image.BILINEAR)
        LR_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        TB_mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        if rotation:
            ROT_mask = mask.rotate(45, resample=PIL.Image.BILINEAR)
        LR_image.save(training_image_dir + image_file[:-4]+'_LR.png')
        TB_image.save(training_image_dir + image_file[:-4] + '_TB.png')
        if rotation:
            ROT_image.save(training_image_dir + image_file[:-4] + '_ROT.png')
        LR_mask.save(training_mask_dir + mask_file[:-4]+'_LR.png')
        TB_mask.save(training_mask_dir + mask_file[:-4] + '_TB.png')
        if rotation:
            ROT_mask.save(training_mask_dir + mask_file[:-4] + '_ROT.png')

def data_augumentation_rotate():
    training_image_dir = "D:/Tuned_Picture/Eye_picture/Training/image/"
    training_mask_dir = "D:/Tuned_Picture/Eye_picture/Training/mask_2class/"

    training_image_files = sorted(os.listdir(training_image_dir))
    training_mask_files = sorted(os.listdir(training_mask_dir))

    for image_file in training_image_files:
        if not '_LR' in image_file and not '_TB' in image_file:
            image = Image.open(training_image_dir + image_file)
            rot_image = image.rotate(45, resample=PIL.Image.BILINEAR)
            rot_image.save(training_image_dir + image_file[:-4] + '_rot.png')

    for mask_file in training_mask_files:
        if not '_LR' in mask_file and not '_TB' in mask_file:
            mask = Image.open(training_mask_dir + mask_file)
            rot_mask = mask.rotate(45, resample=PIL.Image.BILINEAR)
            rot_mask.save(training_mask_dir + mask_file[:-4] + '_rot.png')


def generate_logfile():
    training_image_dir = "./Eye_picture/Training/image/"
    training_mask_dir = "./Eye_picture/Training/mask_2class/"

    test_image_dir = "./Eye_picture/Test/image/"
    test_mask_dir = "./Eye_picture/Test/mask_2class/"

    log_training_dir = "./Eye_picture/Training/log_training.txt"
    log_test_dir = "./Eye_picture/Test/log_test.txt"

    training_imagefiles = sorted(os.listdir(training_image_dir))
    training_maskfiles = sorted(os.listdir(training_mask_dir))

    test_imagefiles = sorted(os.listdir(test_image_dir))
    test_maskfiles = sorted(os.listdir(test_mask_dir))

    log_training = open(log_training_dir, "a")
    log_test = open(log_test_dir, "a")

    for image, mask in zip(training_imagefiles, training_maskfiles):
        log_training.write(image + ' ' + mask + '\n')
    for image, mask in zip(test_imagefiles, test_maskfiles):
        log_test.write(image + ' ' + mask + '\n')

def process_into_2classes():
    Training_mask_dir = "D:/Tuned_Picture/Eye_picture/Training/mask/"
    Training_mask_2classes_dir = "D:/Tuned_Picture/Eye_picture/Training/mask_2class/"
    Test_mask_dir = "D:/Tuned_Picture/Eye_picture/Test/mask/"
    Test_mask_2classes_dir = "D:/Tuned_Picture/Eye_picture/Test/mask_2class/"

    if not os.path.isdir(Training_mask_2classes_dir):
        os.makedirs(Training_mask_2classes_dir)
    if not os.path.isdir(Test_mask_2classes_dir):
        os.makedirs(Test_mask_2classes_dir)

    training_mask_files = os.listdir(Training_mask_dir)
    test_mask_files = os.listdir(Test_mask_dir)

    for training_file in training_mask_files:
        img = Image.open(Training_mask_dir + training_file)
        img = np.array(img)
        img[img < 100] = 0
        img[img >= 100] = 1
        img = Image.fromarray(img)
        img.save(Training_mask_2classes_dir+training_file)

    for test_file in test_mask_files:
        img = Image.open(Test_mask_dir + test_file)
        img = np.array(img)
        img[img < 100] = 0
        img[img >= 100] = 1
        img = Image.fromarray(img)
        img.save(Test_mask_2classes_dir+test_file)

def show_image():
    dir = "D:/Tuned_Picture/Eye_picture/Training/mask_2class/21_manual1_rot.png"

    img = Image.open(dir)
    img_array = np.array(img)
    img_array *= 255
    img_n = Image.fromarray(img_array)
    img_n.show()
    print(img_array)

if __name__ == '__main__':
    # data_augumentation(rotation=True)
    generate_logfile()
    # process_into_2classes()
    # change_picture_shape_and_type([648,486],practical=True,type="Test", origin_size=False)

    # rename_file()
    # show_image()