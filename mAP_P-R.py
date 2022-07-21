import json
import os
import sys
from datetime import datetime
from itertools import combinations
from tkinter import image_names

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from cv2 import split
from numba import jit
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
        if each_name[-3:] == file_type[-3:]:                                                     #请修改文件类型
            filename_list += [each_name]
    
    return filename_list

def polygon_to_square(point_list):
    single_point = np.array(point_list)

    max_x = np.max(single_point[:,0])
    min_x = np.min(single_point[:,0])
    max_y = np.max(single_point[:,1])
    min_y = np.min(single_point[:,1])

    return [min_y,min_x,max_y,max_x]

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


def json_to_gt(json_path):

    # image_name = '1'

    # json_path = os.path.join(os.getcwd(), image_name +'.json')

    load_original = open(json_path,'r', encoding='utf-8')
    json_dict = json.load(load_original)                                                                #读取标记json
    load_original.close()

    shape_list = json_dict["shapes"]                                                                    #获取整个一级字典shapes的值
    gt_bound = []
    gt_label = []
    for all_label_dict in shape_list:                                                                   #遍历所有的标签 提取单个标签
            
        single_label_point = all_label_dict['points']                                                   #获得单个标签里所有关键点的（二级字典）
        
        rec1 = polygon_to_square(single_label_point)                                                    #gt_polygon_to_square
        
        rec1_label = all_label_dict["label"]                                                            #获得单个标签二级字典）

        gt_bound += [rec1]
        
        if rec1_label[0:4] == '003a':
            gt_label += '1'
        elif rec1_label[0:4] == '020a':
            gt_label += '2'    

    gt_label = list(map(int, gt_label))

    json_name = os.path.basename(json_path).split('.')[0]
    json_name_list = [json_name for index in range(len(gt_bound))]                                      #生成文件名称标签 用于框图校验
    
    return gt_bound, gt_label, json_name_list


#@jit(nopython=False)
#初始化
def compute_pr(result_rec, result_label, r_name_list, gt_bound, gt_label, gt_name_list,iou_threshold):
    TP = 0
    TN = 0
    p_r = []
    p_list = []
    r_list = []
    
    rec1_label_list_copy = gt_label
    rec1_list_copy = gt_bound
    label1_count = len([i for i in gt_label if i == '1'])                                                       #标签 1 的数量
    label2_count = len([i for i in gt_label if i == '2'])                                                       #标签 2 的数量
    gt_count = len(gt_label)


    for count1,rec1 in enumerate(result_rec):                                                                     #遍历gt_bound
        r_number = r_name_list[count1]
        iou_list = []
        for count2,rec2 in enumerate(gt_bound):                                                               #遍历result_bound   
            gt_number = gt_name_list[count2]
            if gt_number == r_number:                                                                               #判断是否来自同一张图片        
                iou = compute_IoU(rec1,rec2)
                iou_list += [iou]    
                if iou >= iou_threshold:                                                                  #调整iou阈值
                    if gt_label[count2] == result_label[count1]:                             #认了  是对的
                        TP += 1

                    elif gt_label[count2] != result_label[count1]:                           #认了  但错了   认成别的东西了                                                        

                        TN += 1
        
        if max(iou_list) < iou_threshold:                                                                 #认了  但错了  实际上什么也没有
            TN += 1 
        
        precision = TP/(TP+TN)
        recall = TP/gt_count
        p_list += [precision]
        r_list += [recall]
        p_r += [[precision,recall]]    

    return p_list, r_list, p_r                                                                                  #返回P R 与P_R列表

def predict_result(MODEL_PATH = None, img_path = None ,DETECTION_MIN_CONFIDENCE = 0.0):
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

        DETECTION_MIN_CONFIDENCE = 0.0

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
    json_name = os.path.basename(img_path).split('.')[0]
    r_name_list = [json_name for index in range(len(r['rois']))]
    return r, r_name_list

#最后的NMS  注意！！  输入参数为array对象   不是list对象
def last_NMS_fillter( result_rec, result_label, r_score_list, NMS_iou_threshold):
    death_note = []
    b = range(len(result_rec))
    c = list(combinations(b,2))

    for i in c:
        iou = compute_IoU(result_rec[i[0]], result_rec[i[1]])
        if iou > NMS_iou_threshold:
            death_note.append(i[1])
      
    new_result_rec = np.delete (result_rec, list(sorted(death_note,reverse=True)), axis=0)
    new_result_label = np.delete (result_label, list(sorted(death_note,reverse=True)), axis=0)
    new_r_score_list = np.delete (r_score_list, list(sorted(death_note,reverse=True)), axis=0)
    
    return new_result_rec, new_result_label, new_r_score_list
#过滤不同类型的gt与r     如 只计算其中某一种类别的pr曲线
def label_fillter(result_rec_list, result_label_list, r_name_list, gt_bound_list, gt_label_list, gt_name_list, datafillter = 'all'):

    if isinstance(datafillter,int) :                                                                            #仅提取label == datafillter的所有项
        list1 = [result_rec_list[count] for count,i in  enumerate(result_label_list) if i == datafillter]
        list2 = [i for count,i in  enumerate(result_label_list) if i == datafillter]
        list3 = [r_name_list[count] for count,i in  enumerate(result_label_list) if i == datafillter]
        list4 = [gt_bound_list[count] for count,i in  enumerate(gt_label_list) if i == datafillter]
        list5 = [i for count,i in  enumerate(gt_label_list) if i == datafillter]
        list6 = [gt_name_list[count] for count,i in  enumerate(gt_label_list) if i == datafillter]



        return list1, list2, list3, list4, list5, list6
                                                                                                                
    else:                                                                                                       #提取所有项
        return result_rec_list,result_label_list, r_name_list, gt_bound_list, gt_label_list, gt_name_list

def p_r_interpolation(p_list):

    max_list = []
    max_index = []
    p_inter = []
    for count , i in enumerate(p_list[1:-1]):
        if p_list[count] <= i and p_list[count+2] < i:
            max_list.append(p_list[count+1])
            max_index.append(count+2)
        
    max_list.append(p_list[-1:][0])
    max_index.append(len(p_list))

    p_inter = [max_list[0] for i in p_list[:max_index[0]]]

    for count, i in enumerate(max_index[1:]):
        p_inter += [max_list[count+1] for j in p_list[max_index[count]:max_index[count+1]]]
    
    print(p_inter)

    return p_inter, max_list, max_index

def compute_mAP(max_list,max_index,r):
    mAP = max_list[0]*r[max_index[0]-1]
    for count, mini_shape in enumerate(max_list[1:]):
        mAP += max_list[count+1]*(r[max_index[count+1]-1]-r[max_index[count]-1])
    return mAP


if __name__ == '__main__':
    ###################################################################################################################################
    # 可能需要修改的参数
    ###################################################################################################################################
    MODEL_PATH = r"(21)lrbt_15_Elastic_Dropout_56_5000_(2)10.h5"                                                                     #权重路径
    datafillter_list = [2,1,'all' ]
    NMS_iou_threshold = 0.4                                                                                                         #识别类别
    iou_threshold_list = [ 0.4, 0.5, 0.6]
    ###################################################################################################################################
    a = datetime.now() 
    result_rec_list = []
    result_label_list = []
    result_scores_list = []
    gt_bound_list = []
    gt_label_list = []
    gt_name_list = []
    r_name_lists = []

    mini_img_list = get_files(os.path.join(os.getcwd(), 'img_cache', 'smaller8') , 'png')                                    #获取图片路径
    mini_json_list = get_files(os.path.join(os.getcwd(), 'img_cache', 'smaller8') , 'json')                                  #获取json路径
    for mini_json in mini_json_list:
        gt_bound, gt_label, json_name_list = json_to_gt(mini_json)
        gt_bound_list += gt_bound
        gt_label_list += gt_label
        gt_name_list += json_name_list


    for count,img_path in enumerate(mini_img_list):                                                                             #逐张预测
        r,r_name_list = predict_result(MODEL_PATH = MODEL_PATH
                        , img_path = img_path
                        , DETECTION_MIN_CONFIDENCE = 0.0
                        )
        
        new_result_rec, new_result_label, new_r_score_list = last_NMS_fillter(r['rois'],r['class_ids'], r['scores'], NMS_iou_threshold) # NMS过滤

        result_rec_list +=   new_result_rec.tolist()
        result_label_list += list(new_result_label)
        result_scores_list += list(new_r_score_list)
        r_name_lists += r_name_list[:len(list(new_result_label))]

    
    
    zipped = zip(result_scores_list,result_label_list,result_rec_list,r_name_lists)                                             #对所有预测结果及标签进行排序
    sort_zipped = sorted(zipped,key=lambda x:(x[0],x[1]),reverse=True)
    result = zip(*sort_zipped)
    result_scores_list, result_label_list, result_rec_list,r_name_list= [list(x) for x in result]

    

    for datafillter in datafillter_list:                                                                                            #遍历每一种标签

        p_result_rec_list,p_result_label_list, p_r_name_list, p_gt_bound_list, p_gt_label_list, p_gt_name_list = label_fillter(                 #过滤获得所需标签类
            result_rec_list, result_label_list, r_name_list, gt_bound_list, gt_label_list, gt_name_list,
            datafillter = datafillter)

        if not os.path.exists('./all_mAP_P_R_data/'+ MODEL_PATH.split('.')[0] +'/'):                                                    #新建目录
            os.mkdir('./all_mAP_P_R_data/'+ MODEL_PATH.split('.')[0] +'/')
        all_P_R_data = open('./all_mAP_P_R_data/'+ MODEL_PATH.split('.')[0] +'/'+ str(datafillter)+ '.txt','w',encoding='utf-8')        #新建TXT文档
        all_P_R_data.seek(0)                                                                                                        #重置指针
        all_P_R_data.truncate()                                                                                                     #清空内容

        
        for iou_threshold in iou_threshold_list:
            plt.clf()
            print('start compute pr!')
            p_list, r_list, p_r = compute_pr(p_result_rec_list,p_result_label_list, p_r_name_list, p_gt_bound_list, p_gt_label_list, p_gt_name_list, iou_threshold)     #计算pr值
            
            p_inter, max_list, max_index= p_r_interpolation(p_list)
            mAP = compute_mAP(max_list,max_index,r_list)
            print(p_inter)
            print(r_list)
            

                                                                                                                                    #记录各类模型性能值
            all_P_R_data.writelines('p_inter = '+ str(p_inter))
            all_P_R_data.writelines('\n\n')
            all_P_R_data.writelines('r_list = ' + str(r_list))
            all_P_R_data.writelines('\n\n')
            all_P_R_data.writelines('mAP = ' + str(mAP))
            all_P_R_data.writelines('\n\n')
            all_P_R_data.writelines('iou_threshold = ' + str(iou_threshold))
            all_P_R_data.writelines('\n\n')
            all_P_R_data.writelines('\n\n')
            [all_P_R_data.writelines('#') for i in range(40)]                                                                      #分隔符
            all_P_R_data.writelines('\n\n')
            all_P_R_data.writelines('\n\n')
            
            
                
            plt.plot(r_list,p_list,label='P-R Curve',linewidth=2,color='tomato')
            #marker='o',markerfacecolor='blue',markersize=10)
            plt.plot(r_list,p_inter,label='interpolation line',linewidth=3,color='cornflowerblue')
            plt.fill_between(r_list, p_inter, 0.0, color='cornflowerblue', alpha=.1)
            plt.xlabel('recall')
            plt.ylabel('precision')
            
            plt.title('P-R Line '+'iou_threshold = ' + str(iou_threshold) +' mAP:'+str(format(mAP, '.3f')))
            my_x_ticks = np.arange(0, 1, 0.1)
            my_y_ticks = np.arange(0, 1, 0.1)
            plt.xticks(my_x_ticks)
            # plt.yticks(my_y_ticks)
            #plt.legend(frameon=False,title='mAP:'+str(format(mAP, '.3f')))
            # plt.show()
            plt.savefig('./all_mAP_P_R_data/'+ MODEL_PATH.split('.')[0]+ '/'+ str(datafillter) +' '+ str(iou_threshold)+ '.png', bbox_inches='tight' ,dpi = 450) #画图并保存
            
        all_P_R_data.close()
    print('Down!')
    b = datetime.now()
    
    print('time:',(b-a).seconds)

