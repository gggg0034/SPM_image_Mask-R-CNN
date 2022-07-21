# encoding: utf-8

"""
@author: Libing Wang
@time: 2021/1/12 9:19
@file: rewrite_save_mask.py
@desc: 
"""

import os
import shutil
import sys
from datetime import datetime
from multiprocessing import Pool
from queue import Queue

import cv2
import numpy as np
import yaml
from PIL import Image

# from Images_Augmentation.color_aug_img_json import aug_times

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

import pre_draw_mask
from mrcnn import utils
from mrcnn.config import Config


def get_files(file_path, file_type):                            #用os.walk获取该目录下的所有文件绝对路径并制作list
   
    all_filename_list = [] 
    filename_list = []
    file_path= str(file_path)
    #for filepath,dirname,filename in os.walk(r'C:\Users\73594\Desktop\voc dataset\image'):   #右键获取datadet文件夹的路径并粘贴
    for filepath,dirname,filename in os.walk(file_path):
        for filename in filename:
            all_filename_list += [os.path.join(filepath,filename)]
    for each_name in all_filename_list:                                                 #筛选 特定文件类型并组成新的list
        if each_name[-3:] == file_type[-3:]:                                                     #请修改文件类型
            filename_list += [each_name]
    
    return filename_list

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
    NUM_CLASSES = 1 + 80 # background + 80 shapes

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
    STEPS_PER_EPOCH = 500

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 20


class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IoU_THRESHOLD = 0.7


class DrugDataset(utils.Dataset):
    # the count of instances (objects) in the graph
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # Parse the yaml file obtained in the labelme to get the instance tag
    # corresponding to each layer of the mask.
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    # rewrite draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(int(np.shape(image)[0]-1)):
                for j in range(int(np.shape(image)[1]-1)):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        np.savez_compressed(os.path.join(ROOT_DIR, 'train_dataset', 'rw_mask', info["path"].split("/")[-1].split(".")[0]).zfill(5), mask)
        return mask

    # rewrite load_shapes, which contains your own own categories
    # added path, mask_path, yaml_path to the self.image_info information.
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
                mask_path = mask_floder + img_name + ".png"
                yaml_path = yaml_floder + img_name + ".yaml"
                self.add_image("shapes", image_id=i, path=img_path, mask_path=mask_path,yaml_path=yaml_path)

    # rewrite load_mask
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        np.savez_compressed(os.path.join(ROOT_DIR, 're_mask_cache', 'rw_mask', info["path"].split("/")[-1].split(".")[0]).zfill(5), mask)


def train_model():  # dataset 代表本次训练样本的文件夹名称
    # 训练模型的配置
    dataset_root_path="./re_mask_cache/"
    # dataset_root_path="./train_dataset/"
    img_floder = dataset_root_path + "imgs/"
    mask_floder = dataset_root_path + "mask/"
    yaml_floder = dataset_root_path + "yaml/"
    rw_mask_data = dataset_root_path + "rw_mask/"
    imglist = os.listdir(img_floder)

    count = len(imglist)

    # train and val data set preparation
    data_set_train = DrugDataset()
    data_set_train.load_shapes(count, img_floder, mask_floder, imglist, yaml_floder)
    data_set_train.prepare()

    a = datetime.now()
    # for image_id in range(count):
    #     data_set_train.load_mask(image_id)
    
    queue = Queue()
    thread_num = 16
    [queue.put(id) for id in data_set_train.image_ids]
    print("start! ")
    pthread = Pool(thread_num)

    while not queue.empty():
        image_id = queue.get()
        pthread.apply_async(data_set_train.load_mask, args=(image_id,))

    pthread.close()
    pthread.join()

    print("queue is empty!")
    
    b = datetime.now()
    print('time:',(b-a).seconds)

def main():
    # pre_draw_mask.main()
    if not os.path.exists("re_mask_cache"):
        os.mkdir("re_mask_cache")
    label_path = "re_mask_cache/mask"
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    img_path = "re_mask_cache/imgs"
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    yaml_path = "re_mask_cache/yaml"
    if not os.path.exists(yaml_path):
        os.mkdir(yaml_path)
    label_viz_path = "re_mask_cache/label_viz"
    if not os.path.exists(label_viz_path):
        os.mkdir(label_viz_path)
    label_rw_mask = "re_mask_cache/rw_mask"
    if not os.path.exists(label_rw_mask):
        os.mkdir(label_rw_mask)
    

    if not os.path.exists("train_dataset"):
        os.mkdir("train_dataset")
    label_path = "train_dataset/mask"
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    img_path = "train_dataset/imgs"
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    yaml_path = "train_dataset/yaml"
    if not os.path.exists(yaml_path):
        os.mkdir(yaml_path)
    label_viz_path = "train_dataset/label_viz"
    if not os.path.exists(label_viz_path):
        os.mkdir(label_viz_path)
    label_rw_mask = "train_dataset/rw_mask"
    if not os.path.exists(label_rw_mask):
        os.mkdir(label_rw_mask)

        
    train_model()
    print("start copy!")
    # aug_times = color_aug_img_json.aug_times
    aug_times = 56
    dataset_root_path="./re_mask_cache/"
    rw_mask_data = dataset_root_path + "rw_mask/"
    rw_label_viz = dataset_root_path + "label_viz/"
    yaml_data = dataset_root_path + "yaml/"
    mask_data = dataset_root_path + "mask/"




    rw_mask_list = get_files(rw_mask_data, 'npz')
    for rw_mask in rw_mask_list:
        rw_mask_count = int(os.path.basename(rw_mask).split('.')[0])*aug_times
        for inter_count in range(0, aug_times):
            rw_mask_inter_count = rw_mask_count + inter_count
            shutil.copy(rw_mask, './train_dataset/rw_mask/' + str(rw_mask_inter_count).zfill(5) + '.npz')

    rw_label_list = get_files(rw_label_viz, 'png')
    for label_viz in rw_label_list:
        label_viz_count = int(os.path.basename(label_viz).split('.')[0])*aug_times
        for inter_count in range(0, aug_times):
            label_viz_inter_count = label_viz_count + inter_count
            shutil.copy(label_viz, './train_dataset/label_viz/' + str(label_viz_inter_count).zfill(5) + '.png')

    yaml_data_list = get_files(yaml_data, 'yaml')
    for yaml_yaml in yaml_data_list:
        yaml_count = int(os.path.basename(yaml_yaml).split('.')[0])*aug_times
        for inter_count in range(0, aug_times):
            yaml_inter_count = yaml_count + inter_count
            shutil.copy(yaml_yaml, './train_dataset/yaml/' + str(yaml_inter_count).zfill(5) + '.yaml')
    
    mask_data_list = get_files(mask_data, 'png')
    for mask in mask_data_list:
        mask_count = int(os.path.basename(mask).split('.')[0])*aug_times
        for inter_count in range(0, aug_times):
            mask_inter_count = mask_count + inter_count
            shutil.copy(mask, './train_dataset/mask/' + str(mask_inter_count).zfill(5) + '.png')    
    
    
    
    
    print("all done!")


if __name__ == '__main__':
    main()