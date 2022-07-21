import argparse
import base64
import json
import os
import os.path as osp
import warnings
from datetime import datetime
from multiprocessing import Pool
from queue import Queue

import PIL.Image
import yaml
from labelme import utils
from tqdm import tqdm


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


def json_to_dataset(path, count):
    if os.path.isfile(path) and path.endswith('json'):
        data = json.load(open(path))
        
        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)
        label_name_to_value = {'_background_': 0}
        for shape in data['shapes']:
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        
        # label_values must be dense
        label_values, label_names = [], []
        for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
            label_values.append(lv)
            label_names.append(ln)
        
        assert label_values == list(range(len(label_values)))
        
        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        
        captions = ['{}: {}'.format(lv, ln)
            for ln, lv in label_name_to_value.items()]
        lbl_viz = utils.draw_label(lbl, img, captions)

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

        PIL.Image.fromarray(img).save(osp.join(img_path, str(count).zfill(5)+'.jpg'))

        utils.lblsave(osp.join(label_path, str(count).zfill(5)+'.png'), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(label_viz_path, str(count).zfill(5)+'.png'))

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(yaml_path, str(count).zfill(5)+'.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)


def main(): 
    count_list =[]
    json_list = get_files("./before/", 'json')
    [count_list.append(i) for i in range(len(json_list))]

    zip_args = list(zip(json_list,count_list))


    a = datetime.now()
    queue = Queue()                                                                         #创建一个queue队列
    thread_num = 16                                                                         #设置同时运行的进程数量
    print("start! ")
    pthread = Pool(thread_num)                                                              #创建一个进程池

    for i  in zip_args:

        queue.put(pthread.apply_async(json_to_dataset, args=i))                                 #给函数赋值并将100个子程序put进入queue队列

    pthread.close()                                                                         #主进程等待所有子进程中的子程序运行完毕
    pthread.join()                                                                          #关闭多进程

    print("queue is empty!")
    
    b = datetime.now()
    print('time:',(b-a).seconds)                                                            #打印整体运行时间


if __name__ == '__main__':
    main()