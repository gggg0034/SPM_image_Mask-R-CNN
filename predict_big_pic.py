
import os
import sys
from datetime import datetime
from itertools import combinations
from tkinter import image_names

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from cv2 import split
from PIL import Image
from sklearn.feature_extraction import img_to_graph
from tqdm import tqdm

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
        if each_name[-3:] == file_type:                                                     #请修改文件类型
            filename_list += [each_name]
    
    return filename_list

def overlap_cut(img_name):                                                                  #交叠截图并保存
    #获取图片基本信息
    img_path = os.path.join(os.getcwd(), 'images', img_name)
    img = Image.open(img_path)
    img = img.resize((1024,1024),Image.ANTIALIAS)
    img_width = img.size[0]
    img_hight = img.size[1]

    #定义分割单元
    mini_width = int(img.size[0]//5)
    mini_hight = int(img.size[1]//5)

    #分割四张有最小分割单元的图片重叠的图片
    mini_img_1 = img.crop((0, 0, mini_width*3, mini_hight*3))
    mini_img_1.save(os.path.join(os.getcwd(), 'img_cache', 'pre_predict', '1.png'))                                #保存左上

    mini_img_2 = img.crop((mini_width*2, 0, mini_width*5, mini_hight*3))
    mini_img_2.save(os.path.join(os.getcwd(), 'img_cache', 'pre_predict', '2.png'))                                #保存右上

    mini_img_3 = img.crop((0, mini_hight*2, mini_width*3, mini_hight*5))
    mini_img_3.save(os.path.join(os.getcwd(), 'img_cache', 'pre_predict', '3.png'))                                #保存左下

    mini_img_4 = img.crop((mini_width*2, mini_width*2, mini_width*5, mini_hight*5))
    mini_img_4.save(os.path.join(os.getcwd(), 'img_cache', 'pre_predict', '4.png'))                                #保存右下

def cut_white_edge(img):
    mini_no_egde = img.crop((223, 210, 1417, 1404))                                                                #经验参数
    out = mini_no_egde.resize((1200,1200),Image.ANTIALIAS)
    return out

def crop_paste(img_path_list, save_path, new_img_name):
   
    img1 = cut_white_edge(Image.open(img_path_list[0]))
 
    img2 = cut_white_edge(Image.open(img_path_list[1]))

    img3 = cut_white_edge(Image.open(img_path_list[2]))

    img4 = cut_white_edge(Image.open(img_path_list[3]))

    # #裁剪多余的边角
    p_mini_img_erlu = img1.crop((0, 0, 1000, 1000))
    p_mini_img_erru = img2.crop((200, 0, 1200, 1000))
    p_mini_img_erld = img3.crop((0, 200, 1000, 1200))
    p_mini_img_errd = img4.crop((200, 200, 1200, 1200))
    # #拼接四张图
    new_img = Image.new('RGB', (2000,2000))
    new_img.paste(p_mini_img_erlu,(0, 0))
    new_img.paste(p_mini_img_erru,(1000, 0))
    new_img.paste(p_mini_img_erld,(0, 1000))
    new_img.paste(p_mini_img_errd,(1000, 1000))
    new_img.save(os.path.join(save_path, new_img_name + '_new.png'))


def compute_IoU(rec1,rec2):
    left_column_max  = max(rec1[1],rec2[1])
    right_column_min = min(rec1[3],rec2[3])
    up_row_max       = max(rec1[0],rec2[0])
    down_row_min     = min(rec1[2],rec2[2])
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)


def last_NMS_fillter( result_rec, r_mask_list, result_label, r_score_list, NMS_iou_threshold):
    death_note = []
    b = range(len(result_rec))
    c = list(combinations(b,2))

    for i in c:
        iou = compute_IoU(result_rec[i[0]], result_rec[i[1]])
        if iou > NMS_iou_threshold:
            death_note.append(i[1])
    
    new_result_rec = np.delete (result_rec, list(sorted(death_note,reverse=True)), axis=0)
    new_r_mask_list = np.delete (r_mask_list, list(sorted(death_note,reverse=True)), axis=2)
    new_result_label = np.delete (result_label, list(sorted(death_note,reverse=True)), axis=0)
    new_r_score_list = np.delete (r_score_list, list(sorted(death_note,reverse=True)), axis=0)

    
    return new_result_rec, new_r_mask_list, new_result_label, new_r_score_list

def predict_big_pic(count, MODEL_PATH = None, class_names = None, img_path = None , colors = None, DETECTION_MIN_CONFIDENCE = 0.5):
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    import mrcnn.model as modellib
    from mrcnn import utils, visualize

    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version


    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # MODEL_PATH = r"C:\Users\73594\Desktop\my python\mask-R-CNN\Mask_RCNN-master\logs\shapes20220330T1002\mask_rcnn_shapes_0010.h5"
    '''
    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)
    '''
    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = "coco"

        NUM_CLASSES = 1 + 80

        GPU_COUNT = 1

        IMAGES_PER_GPU = 1

        DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE

    config = InferenceConfig()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(MODEL_PATH, by_name=True)

    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')

    '''class_names = ['BG', 'triangle', 'hexagon', '3','4','5','6','7','8','9','10',
                '11','12','13','14','15','16','17','18','19','20','21','22','23',
                '24','25','26','27','28','29','30','31','32','33','34','35','36',
                '37','38','39','40','41','42','43','44','45','46','47','48','49',
                '50','51','52','53','54','55','56','57','58','59','60','61','62',
                '63','64','65'',66','67','68'',69','70','71','72','74','75','76',
                '77','78','79','80']'''
    '''class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']'''

    # Load a random image from the images folder
    # file_names = next(os.walk(IMAGE_DIR))[2]

    # for count,mini_img in enumerate(mini_imgs)
    save_path = os.path.join(ROOT_DIR, "img_cache", "post_predict")
    image = skimage.io.imread(img_path)


    a = datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    b = datetime.now()


    # Visualize results
    print('time:',(b-a).seconds)

    r = results[0]
    print(r)
    colors = [(0.7999999999999998, 1.0, 0.0), (0.0, 0.6727272727272728, 1.0)]
    NMS_iou_threshold = 0.3
    new_result_rec, new_r_mask_list, new_result_label, new_r_score_list = last_NMS_fillter(r['rois'], r['masks'], r['class_ids'],r['scores'],NMS_iou_threshold)

    visualize.save_instances_sc(count , save_path, image, new_result_rec, new_r_mask_list, new_result_label, 
                                class_names, new_r_score_list, colors = colors, show_bbox= False, captions= True)





if __name__ == '__main__':
    ###################################################################################################################################
    # 可能需要修改的参数
    ###################################################################################################################################
    MODEL_PATH = r"mask_rcnn_shapes_0002.h5"            #权重路径
 
    class_names = ['BG', '003a', '020a']                                                                                            #识别类别列表（按顺序）

    colors = [(0.7999999999999998, 1.0, 0.0), (0.0, 0.6727272727272728, 1.0)]                                                       #颜色列表（HVS）

    img_names = '00.png'                                                                                                            #图片名称

    DETECTION_MIN_CONFIDENCE_LIST = [0.0,0.1,0.15,0.20,0.25,0.30,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99,1.0] #置信度列表
    ###################################################################################################################################
    a = datetime.now() 
    for DETECTION_MIN_CONFIDENCE in DETECTION_MIN_CONFIDENCE_LIST[0:1]:
        overlap_cut(img_names)                                                                                                      #切分图片
        mini_img_names = img_names.split('.')[0] + img_names.split('.')[1]
        mini_img_list = get_files(os.path.join(os.getcwd(), 'img_cache', 'pre_predict') , 'png')                                    #获取图片路径
             
        for count,img_name in enumerate(mini_img_list):                                                                             #逐张预测
            predict_big_pic(count,MODEL_PATH = MODEL_PATH
                            , class_names = class_names
                            , img_path = img_name
                            , colors = colors
                            ,DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE
                            )
        CONFIDENCE = str(DETECTION_MIN_CONFIDENCE)
        post_img_list_ = get_files(os.path.join(os.getcwd(), 'img_cache', 'post_predict') , 'png')
        crop_paste(post_img_list_, os.path.join(os.getcwd(), 'img_cache'), mini_img_names + '_' + CONFIDENCE )                      #拼接图片
    b = datetime.now()
    print('time:',(b-a).seconds)
