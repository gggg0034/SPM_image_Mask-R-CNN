import math
import os
import random
import re
import sys
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image

#import utils
from mrcnn import model as modellib
from mrcnn import utils, visualize
from mrcnn.config import Config
from mrcnn.model import log

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Root directory of the project
ROOT_DIR = os.getcwd()

#ROOT_DIR = os.path.abspath("../")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num=0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80 # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6) # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 5000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20


config = ShapesConfig()
config.display()

class DrugDataset(utils.Dataset):
# 得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

        # 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # 重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        
        return mask

    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    # 并在self.image_info信息中添加了path、mask_path 、yaml_path
    # yaml_pathdataset_root_path = "/tongue_dateset/"
    # img_floder = dataset_root_path + "rgb"
    # mask_floder = dataset_root_path + "mask"
    # dataset_root_path = "/tongue_dateset/"
    def load_shapes1(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes,可通过这种方式扩展多个物体
        self.add_class("shapes", 1, "tank") # 黑色素瘤
        self.add_class("shapes", 2, "triangle")
        self.add_class("shapes", 3, "white")
        for i in range(count):
        # 获取图片宽和高

            filestr = imglist[i].split(".")[0]
            #print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            #print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            #filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
            width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)
    
    
    # 重新写load_shapes，里面包含自己的类别,可以任意添加
    def load_shapes(self, count, img_floder, mask_floder, imglist, yaml_floder):
        
        # self.add_class("shapes", 1, "circle")
        # self.add_class("shapes", 2, "square")
        # self.add_class("shapes", 3, "triangle")
        self.add_class("shapes", 1, "003a")
        self.add_class("shapes", 2, "020a")
        self.add_class("shapes", 3, "3")
        self.add_class("shapes", 4, "4")
        self.add_class("shapes", 5, "5")
        self.add_class("shapes", 6, "6")
        self.add_class("shapes", 7, "7")
        self.add_class("shapes", 8, "8")
        self.add_class("shapes", 9, "9")
        self.add_class("shapes", 10, "10")
        self.add_class("shapes", 11, "11")
        self.add_class("shapes", 12, "12")
        self.add_class("shapes", 13, "13")
        self.add_class("shapes", 14, "14")
        self.add_class("shapes", 15, "15")
        self.add_class("shapes", 16, "16")
        self.add_class("shapes", 17, "17")
        self.add_class("shapes", 18, "18")
        self.add_class("shapes", 19, "19")
        self.add_class("shapes", 20, "20")
        self.add_class("shapes", 21, "21")
        self.add_class("shapes", 22, "22")
        self.add_class("shapes", 23, "23")
        self.add_class("shapes", 24, "24")
        self.add_class("shapes", 25, "25")
        self.add_class("shapes", 26, "26")
        self.add_class("shapes", 27, "27")
        self.add_class("shapes", 28, "28")
        self.add_class("shapes", 29, "29")
        self.add_class("shapes", 30, "30")
        self.add_class("shapes", 31, "31")
        self.add_class("shapes", 32, "32")
        self.add_class("shapes", 33, "33")
        self.add_class("shapes", 34, "34")
        self.add_class("shapes", 35, "35")
        self.add_class("shapes", 36, "36")
        self.add_class("shapes", 37, "37")
        self.add_class("shapes", 38, "38")
        self.add_class("shapes", 39, "39")
        self.add_class("shapes", 40, "40")
        self.add_class("shapes", 41, "41")
        self.add_class("shapes", 42, "42")
        self.add_class("shapes", 43, "43")
        self.add_class("shapes", 44, "44")
        self.add_class("shapes", 45, "45")
        self.add_class("shapes", 46, "46")
        self.add_class("shapes", 47, "47")
        self.add_class("shapes", 48, "48")
        self.add_class("shapes", 49, "49")
        self.add_class("shapes", 50, "50")
        self.add_class("shapes", 51, "51")
        self.add_class("shapes", 52, "52")
        self.add_class("shapes", 53, "53")
        self.add_class("shapes", 54, "54")
        self.add_class("shapes", 55, "55")
        self.add_class("shapes", 56, "56")
        self.add_class("shapes", 57, "57")
        self.add_class("shapes", 58, "58")
        self.add_class("shapes", 59, "59")
        self.add_class("shapes", 60, "60")
        self.add_class("shapes", 61, "61")
        self.add_class("shapes", 62, "62")
        self.add_class("shapes", 63, "63")
        self.add_class("shapes", 64, "64")
        self.add_class("shapes", 65, "65")
        self.add_class("shapes", 66, "66")
        self.add_class("shapes", 67, "67")
        self.add_class("shapes", 68, "68")
        self.add_class("shapes", 69, "69")
        self.add_class("shapes", 70, "70")
        self.add_class("shapes", 71, "71")
        self.add_class("shapes", 72, "72")
        self.add_class("shapes", 73, "73")
        self.add_class("shapes", 74, "74")
        self.add_class("shapes", 75, "75")
        self.add_class("shapes", 76, "76")
        self.add_class("shapes", 77, "77")
        self.add_class("shapes", 78, "78")
        self.add_class("shapes", 79, "79")
        self.add_class("shapes", 80, "80")
        
        for i in range(count):
            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = img_floder + img
                # mask_path = mask_floder + img_name + ".png"
                mask_path = mask_floder + img_name + ".npz"
                yaml_path = yaml_floder + img_name + ".yaml"
                self.add_image("shapes", image_id=i, path=img_path, mask_path=mask_path,yaml_path=yaml_path)
    
    # 重写load_mask
    def load_mask1(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id",image_id)
        info = self.image_info[image_id]
        count = 1 # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img,image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("tank") != -1:
            # print "box"
                labels_form.append("tank")
            elif labels[i].find("triangle")!=-1:
            #print "column"
                labels_form.append("triangle")
            elif labels[i].find("white")!=-1:
            #print "package"
                labels_form.append("white")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)
    
    
    # 重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        '''
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        '''
        mask = np.load(info['mask_path'])['arr_0']
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        labels_form=[]
        for i in range(len(labels)):
            if labels[i].find("003a")!=-1:
                labels_form.append("003a")
            # if labels[i].find("circle")!=-1:
            #     labels_form.append("circle")
            elif labels[i].find("020a")!=-1:
                labels_form.append("020a")
            # elif labels[i].find("triangle")!=-1:
            #     labels_form.append("triangle")
            elif labels[i].find("3")!=-1:
                labels_form.append("3")
            elif labels[i].find("4")!=-1:
                labels_form.append("4")
            elif labels[i].find("5")!=-1:
                labels_form.append("5")
            elif labels[i].find("6")!=-1:
                labels_form.append("6")
            elif labels[i].find("7")!=-1:
                labels_form.append("7")
            elif labels[i].find("8")!=-1:
                labels_form.append("8")
            elif labels[i].find("9")!=-1:
                labels_form.append("9")
            elif labels[i].find("10")!=-1:
                labels_form.append("10")
            elif labels[i].find("11")!=-1:
                labels_form.append("11")
            elif labels[i].find("12")!=-1:
                labels_form.append("12")
            elif labels[i].find("13")!=-1:
                labels_form.append("13")
            elif labels[i].find("14")!=-1:
                labels_form.append("14")
            elif labels[i].find("15")!=-1:
                labels_form.append("15")
            elif labels[i].find("16")!=-1:
                labels_form.append("16")
            elif labels[i].find("17")!=-1:
                labels_form.append("17")
            elif labels[i].find("18")!=-1:
                labels_form.append("18")
            elif labels[i].find("19")!=-1:
                labels_form.append("19")
            elif labels[i].find("20")!=-1:
                labels_form.append("20")
            elif labels[i].find("21")!=-1:
                labels_form.append("21")
            elif labels[i].find("22")!=-1:
                labels_form.append("22")
            elif labels[i].find("23")!=-1:
                labels_form.append("23")
            elif labels[i].find("24")!=-1:
                labels_form.append("24")
            elif labels[i].find("25")!=-1:
                labels_form.append("25")
            elif labels[i].find("26")!=-1:
                labels_form.append("26")
            elif labels[i].find("27")!=-1:
                labels_form.append("27")
            elif labels[i].find("28")!=-1:
                labels_form.append("28")
            elif labels[i].find("29")!=-1:
                labels_form.append("29")
            elif labels[i].find("30")!=-1:
                labels_form.append("30")
            elif labels[i].find("31")!=-1:
                labels_form.append("31")
            elif labels[i].find("32")!=-1:
                labels_form.append("32")
            elif labels[i].find("33")!=-1:
                labels_form.append("33")
            elif labels[i].find("34")!=-1:
                labels_form.append("34")
            elif labels[i].find("35")!=-1:
                labels_form.append("35")
            elif labels[i].find("36")!=-1:
                labels_form.append("36")
            elif labels[i].find("37")!=-1:
                labels_form.append("37")
            elif labels[i].find("38")!=-1:
                labels_form.append("38")
            elif labels[i].find("39")!=-1:
                labels_form.append("39")
            elif labels[i].find("40")!=-1:
                labels_form.append("40")
            elif labels[i].find("41")!=-1:
                labels_form.append("41")
            elif labels[i].find("42")!=-1:
                labels_form.append("42")
            elif labels[i].find("43")!=-1:
                labels_form.append("43")
            elif labels[i].find("44")!=-1:
                labels_form.append("44")
            elif labels[i].find("45")!=-1:
                labels_form.append("45")
            elif labels[i].find("46")!=-1:
                labels_form.append("46")
            elif labels[i].find("47")!=-1:
                labels_form.append("47")
            elif labels[i].find("48")!=-1:
                labels_form.append("48")
            elif labels[i].find("49")!=-1:
                labels_form.append("49")
            elif labels[i].find("50")!=-1:
                labels_form.append("50")
            elif labels[i].find("51")!=-1:
                labels_form.append("51")
            elif labels[i].find("52")!=-1:
                labels_form.append("52")
            elif labels[i].find("53")!=-1:
                labels_form.append("53")
            elif labels[i].find("54")!=-1:
                labels_form.append("54")
            elif labels[i].find("55")!=-1:
                labels_form.append("55")
            elif labels[i].find("56")!=-1:
                labels_form.append("56")
            elif labels[i].find("57")!=-1:
                labels_form.append("57")
            elif labels[i].find("58")!=-1:
                labels_form.append("58")
            elif labels[i].find("59")!=-1:
                labels_form.append("59")
            elif labels[i].find("60")!=-1:
                labels_form.append("60")
            elif labels[i].find("61")!=-1:
                labels_form.append("61")
            elif labels[i].find("62")!=-1:
                labels_form.append("62")
            elif labels[i].find("63")!=-1:
                labels_form.append("63")
            elif labels[i].find("64")!=-1:
                labels_form.append("64")
            elif labels[i].find("65")!=-1:
                labels_form.append("65")
            elif labels[i].find("66")!=-1:
                labels_form.append("66")
            elif labels[i].find("67")!=-1:
                labels_form.append("67")
            elif labels[i].find("68")!=-1:
                labels_form.append("68")
            elif labels[i].find("69")!=-1:
                labels_form.append("69")
            elif labels[i].find("70")!=-1:
                labels_form.append("70")
            elif labels[i].find("71")!=-1:
                labels_form.append("71")
            elif labels[i].find("72")!=-1:
                labels_form.append("72")
            elif labels[i].find("73")!=-1:
                labels_form.append("73")
            elif labels[i].find("74")!=-1:
                labels_form.append("74")
            elif labels[i].find("75")!=-1:
                labels_form.append("75")
            elif labels[i].find("76")!=-1:
                labels_form.append("76")
            elif labels[i].find("77")!=-1:
                labels_form.append("77")
            elif labels[i].find("78")!=-1:
                labels_form.append("78")
            elif labels[i].find("79")!=-1:
                labels_form.append("79")
            elif labels[i].find("80")!=-1:
                labels_form.append("80")


        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def train_main():
    #基础设置
    dataset_root_path="./train_dataset/"
    # dataset_root_path="./train_dataset/"
    img_floder = dataset_root_path + "imgs/"
    mask_floder = dataset_root_path + "rw_mask/"
    yaml_floder = dataset_root_path + "yaml/"
    imglist = os.listdir(img_floder)

    count = len(imglist)

    # dataset_root_path="train_data/"
    # img_floder = dataset_root_path + "pic"
    # mask_floder = dataset_root_path + "cv2_mask"
    # #yaml_floder = dataset_root_path
    # imglist = os.listdir(img_floder)
    # count = len(imglist)

    #train与val数据集准备
    dataset_train = DrugDataset()
    dataset_train.load_shapes(count, img_floder, mask_floder, imglist, yaml_floder)
    dataset_train.prepare()

    #print("dataset_train-->",dataset_train._image_ids)

    dataset_val = DrugDataset()
    dataset_val.load_shapes(7, img_floder, mask_floder, imglist,yaml_floder)
    dataset_val.prepare()

    #print("dataset_val-->",dataset_val._image_ids)

    # Load and display random samples
    #image_ids = np.random.choice(dataset_train.image_ids, 4)
    #for image_id in image_ids:
    # image = dataset_train.load_image(image_id)
    # mask, class_ids = dataset_train.load_mask(image_id)
    # visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco" # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs= 1,
                layers='heads')



        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also
        # pass a regular expression to select which layers to
        # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs= 2, 
                layers="all")

if __name__ == '__main__':
    train_main()
