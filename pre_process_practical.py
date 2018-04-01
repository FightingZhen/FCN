import os
from PIL import Image

def gather_image_and_pred():
    image_dir = "D:/Result_From_Server/Test4/0.8/image/"
    pred_dir = "D:/Result_From_Server/Test4/0.8/pred/"
    out_dir = "D:/Result_From_Server/Test4/0.8/gather_out/"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    image_list = sorted(os.listdir(image_dir))
    pred_list = sorted(os.listdir(pred_dir))

    counter = len(image_list)

    for i in range(1,counter + 1):
        image = Image.open(image_dir + image_list[i-1])
        pred = Image.open(pred_dir + pred_list[i-1])

        gather_image = Image.new('RGB',[1600,800])
        gather_image.paste(image, (0,0,800,800))
        gather_image.paste(pred, (800,0,1600,800))
        if i < 10:
            gather_image.save(out_dir + '0' + str(i) + '.png')
            print("Image 0%s gather complete !" % str(i))
        else:
            gather_image.save(out_dir + str(i) + '.png')
            print("Image %s gather complete !" % str(i))

# 对真实医疗图像图片进行切分
def crop_picture(rownum, colnum, recover=False, crop=False):
    if crop == True:
        image_dir = "D:/Eye_picture_real/"
        new_image_dir = "D:/Eye_picture_real_crop/"

        if not os.path.isdir(new_image_dir):
            os.makedirs(new_image_dir)
        file_list = os.listdir(image_dir)
        for file in file_list:
            img = Image.open(image_dir + file)
            w, h = img.size
            print(w,h)
            rowheight = h // rownum
            colwidth = w // colnum

            for r in range(rownum):
                for c in range(colnum):
                    box = (c * colwidth, r * rowheight, (c+1) * colwidth, (r+1) * rowheight)
                    print(box)
                    crop_img = img.crop(box)
                    crop_img.save(new_image_dir + file.strip('.png') + str(r) + str(c) + '.png')

    if recover == True:
        image_dir = "D:/test_saved_practical/image/"
        pred_dir = "D:/test_saved_practical/pred/"
        recover_image_dir = "D:/recover/image/"
        recover_pred_dir = "D:/recover/pred/"
        if not os.path.isdir(recover_image_dir):
            os.makedirs(recover_image_dir)
        if not os.path.isdir(recover_pred_dir):
            os.makedirs(recover_pred_dir)

        image_file_list = sorted(os.listdir(image_dir))
        image_filenum = len(image_file_list)
        file_num = image_filenum // (rownum * colnum)

        for i in range(1,file_num + 1):
            recover_image = Image.new('RGB', [2592, 1944])
            for r in range(rownum):
                for c in range(colnum):
                    if i <10:
                        image_name = '0' + str(i) + str(r) + str(c) + '_image.png'
                    else:
                        image_name = str(i) + str(r) + str(c) + '_image.png'
                    img = Image.open(image_dir + image_name)
                    w, h = img.size

                    box = (c * w, r * h, (c + 1) * w, (r + 1) * h)
                    recover_image.paste(img, box)
            if i < 10:
                recover_image.save(recover_image_dir + '0' + str(i) + '.png')
                print('Image 0%s.png saved !' % str(i))
            else:
                recover_image.save(recover_image_dir + str(i) + '.png')
                print('Image %s.png saved !' % str(i))


        pred_file_list = sorted(os.listdir(pred_dir))
        pred_filenum = len(pred_file_list)
        file_num = pred_filenum // (rownum * colnum)

        for i in range(1, file_num + 1):
            recover_image = Image.new('RGB', [2592, 1944])
            for r in range(rownum):
                for c in range(colnum):
                    if i < 10:
                        image_name = '0' + str(i) + str(r) + str(c) + '_pred.png'
                    else:
                        image_name = str(i) + str(r) + str(c) + '_pred.png'
                    img = Image.open(pred_dir + image_name)
                    w, h = img.size

                    box = (c * w, r * h, (c + 1) * w, (r + 1) * h)
                    recover_image.paste(img, box)
            if i < 10:
                recover_image.save(recover_pred_dir + '0' + str(i) + '.png')
                print("Pred 0%s saved !" % str(i))
            else:
                recover_image.save(recover_pred_dir + str(i) + '.png')
                print("Pred %s saved !" % str(i))

def change_category():
    image_dir = ""
    image_out_dir = ""
    if not os.path.isdir(image_out_dir):
        os.makedirs(image_out_dir)

    file_list = os.listdir(image_dir)

    for file in file_list:
        Image.open(image_dir + file).save(image_dir)

def generate_logfile():
    training_image_dir = "./Practical_Eye_image/"

    log_dir = "./Practical_Eye_image/log.txt"

    imagefiles = sorted(os.listdir(training_image_dir))

    log = open(log_dir, "a")

    for image in imagefiles:
        log.write(image + '\n')

if __name__ == '__main__':
    # change_category()
    # crop_picture(4,4,recover=True)
    # generate_logfile()
    gather_image_and_pred()