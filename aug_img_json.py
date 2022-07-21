import base64
import cmath
import json
import math
import os
import shutil

import cv2
import numpy as np
from PIL import Image
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

#def save_rename_image( image , rename , path):

#旋转增强图片
def rotation_aug_jpg_png(image_path , rotation):
    file_type_list = ['jpg','png']        #定义可被查找的文件类型
    for file_type in file_type_list:

        image_fullname = os.path.basename(image_path) #获取文件全名  含尾缀
        
        image_fbname = image_fullname.split('.')

        image_frontname = image_fbname[0]     #获取文件名  无尾缀  方便后续更名
        
        image_suffix = image_fbname[1]        #获取文件尾缀   不含 '.' 
        
        if image_suffix in file_type:

            image = Image.open(image_path)   #读入一张图片
        
            image_newfullname = image_frontname + "({})".format(rotation) + "." +image_suffix  #重命名
            
            image = image.rotate(-rotation, expand=True)
            image = image.resize(image.size)
            image_newfullname = os.path.join(os.path.dirname(image_path), image_newfullname)
            image.save(image_newfullname)  #旋转并保存


    file_type_list = ['jpg','png']        #定义可被查找的文件类型
    for file_type in file_type_list:

        image_fullname = os.path.basename(image_path) #获取文件全名  含尾缀
        
        image_fbname = image_fullname.split('.')

        image_frontname = image_fbname[0]     #获取文件名  无尾缀  方便后续更名
        
        image_suffix = image_fbname[1]        #获取文件尾缀   不含 '.' 
        
        if image_suffix in file_type:

            image = Image.open(image_path)   #读入一张图片
        
            image_newfullname = image_frontname + "({})".format(rotation) + "." +image_suffix  #重命名
            
            image = image.rotate(-rotation, expand=True)
            image = image.resize(image.size)
            image_newfullname = os.path.join(os.path.dirname(image_path), image_newfullname)
            image.save(image_newfullname)  #翻转并保存

        #第一个参数是 关键坐标点列表 ;  二 逆时针旋转角度;三 旋转中心默认为[0,0]
def get_point_rotation( point_list , rotation, rotate_center = [0,0]) :                                                    
    new_point_list = []
    for point in point_list:
        rad = math.radians(rotation)                                     #构建旋转相位

        X1 = point[0]-rotate_center[0]
        Y1 = point[1]-rotate_center[1]
                                                        
        Z1 = complex(X1 , Y1)                                            #构建复数
        module_of_vecter1 = abs(Z1)                                      #构建向量模
        phase1 = cmath.phase(Z1)                                         #构建相位
        
        phase2 = phase1 + rad                                            #原相位加上旋转相位    顺时针旋转 所以是phase1 + rad

        X2 = module_of_vecter1*math.cos(phase2)+rotate_center[0]                       
        Y2 = module_of_vecter1*math.sin(phase2)+rotate_center[1]       
        new_point = [X2 , Y2]
        new_point_list += [new_point]

    return new_point_list


        #第一个参数是 关键坐标点列表 ;  二 是移动的向量      
def get_point_move( point_list , move_vecter = [0,0]) :                                                    
    new_point_list = []
    for point in point_list:

        X1 = point[0]
        Y1 = point[1]
  
        X2 =X1 + move_vecter[0]                        
        Y2 =Y1 + move_vecter[1]        
        new_point = [X2 , Y2]
        new_point_list += [new_point]

    return new_point_list


#第一个参数是 图片路径 ; 二 翻转方向 tb = FLIP_TOP_BOTTOM , lr = FLIP_LEFT_RIGHT
def flip_aug_jpg_png(image_path, flip_way):
    file_type_list = ['jpg','png']        #定义可被查找的文件类型
    for file_type in file_type_list:

        image_fullname = os.path.basename(image_path) #获取文件全名  含尾缀
        
        image_fbname = image_fullname.split('.')

        image_frontname = image_fbname[0]     #获取文件名  无尾缀  方便后续更名
        
        image_suffix = image_fbname[1]        #获取文件尾缀   不含 '.' 
        
        if image_suffix in file_type:

            image = Image.open(image_path)   #读入一张图片
        
            image_newfullname = image_frontname + "({})".format(flip_way) + "." +image_suffix  #重命名
            if flip_way == 'tb':
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_way == 'lr':
                image = image.transpose(Image.FLIP_LEFT_RIGHT)    
            image = image.resize(image.size)
            image_newfullname = os.path.join(os.path.dirname(image_path), image_newfullname)
            image.save(image_newfullname)  #翻转并保存

#第一个参数是 关键坐标点列表 ; 二 翻转方向 tb = FLIP_TOP_BOTTOM , lr = FLIP_LEFT_RIGHT ; 三 翻转中心坐标
def get_point_flip( point_list ,flip_way , flip_center = [0,0]) :                                                    
    new_point_list = []
    for point in point_list:

        X1 = point[0]
        Y1 = point[1]
        
        if flip_way == 'tb':
            X2 =X1                         
            Y2 =2*flip_center[1] - Y1                                                             #上下翻转只需要改变y的坐标
        elif flip_way == 'lr':
            X2 =2*flip_center[0] - X1                                                             #左右翻转只需要改变x的坐标  
            Y2 =Y1        
        
        new_point = [X2 , Y2]
        new_point_list += [new_point]

    return new_point_list

# 第一参数是 图片路径，第二参数是顺时针旋转角度    img 与 对应json同时增强
def aug_img_json(image_path , rotation ): 
    
    rad = math.radians(rotation)                                                                        #构建旋转相位
    coefficient_of_expansion = math.fabs(math.sin(rad))+ math.fabs(math.cos(rad))                       #计算膨胀系数

    #读取图片
    ######################################################################
    
    image_fullname = os.path.basename(image_path)                                                       #获取文件全名  含尾缀
    
    image_fbname = image_fullname.split('.')

    image_frontname = image_fbname[0]                                                                   #获取文件名  无尾缀  方便后续更名
    image_suffix = image_fbname[1]                                                                      #获取文件尾缀   不含 '.'     
  
    image_newfullname = image_frontname + "({})".format(rotation) + "." +image_suffix                   #生成新图名字 
    image_newpath = os.path.join(os.path.dirname(image_path), image_newfullname)                                        #生成新图的路径
    
    rotation_aug_jpg_png(image_path , rotation)                                                                  #给原图增强  生成新图片
    
    ######################################################################
    
       
    #打开开原json  读取
    ######################################################################
    json_path = os.path.join(os.path.dirname(image_path), image_frontname +'.json')

    load_original = open(json_path,'r', encoding='utf-8')
    json_dict = json.load(load_original)                                                                #读取标记json
    load_original.close()
    ######################################################################
    
    
    #修改imagePath
    ######################################################################
    
    json_fullname = os.path.basename(json_path)                                                          #获取文件全名  含尾缀
            
    json_fbname = json_fullname.split('.')

    json_frontname = json_fbname[0]                                                                     #获取文件名  无尾缀  方便后续更名
    json_suffix = json_fbname[1]                                                                       #获取文件尾缀   不含 '.' 

    json_newfullname = json_frontname + "({})".format(rotation) + "." +json_suffix                    #重命名

    json_dict["imagePath"] = image_newfullname
    #######################################################################
    
    
    #修改imageData
    ##########################################################################
    with open(image_newpath, 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    json_dict["imageData"] = imageData                                                                 #写入新的imageData
    ######################################################################


    #修改旋转后的图片大小
    ######################################################################
    imageHeight = json_dict["imageHeight"]                                                             #获取图片的高和宽
    imageWidth = json_dict["imageWidth"]

    new_imageHeight = round(imageHeight*coefficient_of_expansion)                                      #旋转之后四舍五入图片像素
    new_imageWidth = round(imageWidth*coefficient_of_expansion)
  
    json_dict["imageHeight"] = new_imageHeight                                                         #将旋转后的图片大小写入json
    json_dict["imageWidth"] = new_imageWidth
    #####################################################################
    
    #修改points
    #####################################################################
    shape_list = json_dict["shapes"]                                                                   #获取整个一级字典shapes的值

    rotate_center = [new_imageHeight/2,new_imageWidth/2]                                               #旋转中心
    
    move_vecter = [(new_imageWidth-imageWidth)/2,(new_imageHeight-imageHeight)/2]                      #算出位移矢量

    for all_label_dict in shape_list:                                                                  #遍历所有的标签 提取单个标签
        
        single_label_point = all_label_dict['points']                                                  #获得单个标签里所有关键点的（二级字典）
        
        
        
        new_all_point_list = get_point_move(single_label_point, move_vecter = move_vecter)
        new_all_point_list = get_point_rotation(new_all_point_list, rotation ,rotate_center = rotate_center)  #将所有关键点绕 旋转中心 旋转
        
        

        all_label_dict['points'] = new_all_point_list                                                  #将新关键点坐标写入二级字典point关键字
    

    ######################################################################

    #将新内容写入新json
    ######################################################################
    json_new_dict = json.dumps(json_dict, indent=2)
    
    load_precessed = open( os.path.join( os.path.dirname(image_path),json_newfullname) , 'w')
    load_precessed.write(json_new_dict)
    load_precessed.close()
    ######################################################################
# 第一参数是 图片路径，第二参数是翻转方式    img 与 对应json同时增强    
def aug_flip_img_json(image_path , flip_way ): 
    

    #读取图片
    ######################################################################
    
    image_fullname = os.path.basename(image_path)                                                       #获取文件全名  含尾缀
    
    image_fbname = image_fullname.split('.')

    image_frontname = image_fbname[0]                                                                   #获取文件名  无尾缀  方便后续更名
    image_suffix = image_fbname[1]                                                                      #获取文件尾缀   不含 '.'     
  
    image_newfullname = image_frontname + "({})".format(flip_way) + "." +image_suffix                   #生成新图名字 
    image_newpath = os.path.join(os.path.dirname(image_path), image_newfullname)                                        #生成新图的路径
    
    flip_aug_jpg_png(image_path , flip_way)                                                                  #给原图增强  生成新图片
    
    ######################################################################
    
       
    #打开开原json  读取
    ######################################################################
    json_path = os.path.join(os.path.dirname(image_path), image_frontname +'.json')

    load_original = open(json_path,'r', encoding='utf-8')
    json_dict = json.load(load_original)                                                                #读取标记json
    load_original.close()
    ######################################################################
    
    
    #修改imagePath
    ######################################################################
    
    json_fullname = os.path.basename(json_path)                                                          #获取文件全名  含尾缀
            
    json_fbname = json_fullname.split('.')

    json_frontname = json_fbname[0]                                                                     #获取文件名  无尾缀  方便后续更名
    json_suffix = json_fbname[1]                                                                       #获取文件尾缀   不含 '.' 

    json_newfullname = json_frontname + "({})".format(flip_way) + "." +json_suffix                    #重命名

    json_dict["imagePath"] = image_newfullname
    #######################################################################
    
    
    #修改imageData
    ##########################################################################
    with open(image_newpath, 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    json_dict["imageData"] = imageData                                                                 #写入新的imageData
    ######################################################################


    #读取图片大小
    ######################################################################
    imageHeight = json_dict["imageHeight"]                                                             #获取图片的高和宽
    imageWidth = json_dict["imageWidth"]

    #####################################################################
    
    #修改points
    #####################################################################
    shape_list = json_dict["shapes"]                                                                   #获取整个一级字典shapes的值

    flip_center = [imageHeight/2,imageWidth/2]                                                       #旋转中心
    
    for all_label_dict in shape_list:                                                                  #遍历所有的标签 提取单个标签
        
        single_label_point = all_label_dict['points']                                                  #获得单个标签里所有关键点的（二级字典）
        
        new_all_point_list = get_point_flip(single_label_point, flip_way , flip_center = flip_center)  #将所有关键点绕 旋转中心 旋转
        
        all_label_dict['points'] = new_all_point_list                                                  #将新关键点坐标写入二级字典point关键字
    

    ######################################################################

    #将新内容写入新json
    ######################################################################
    json_new_dict = json.dumps(json_dict, indent=2)
    
    load_precessed = open( os.path.join( os.path.dirname(image_path),json_newfullname) , 'w')
    load_precessed.write(json_new_dict)
    load_precessed.close()

if __name__ == '__main__':

    #file_path = os.path.join( os.getcwd(), 'before' )
    # file_path =r"C:\Users\73594\Desktop\aug_test_in"
    file_path =r'C:\Users\73594\Desktop\my python\mask-R-CNN\Mask_RCNN-master\img_cache\Ground_Truth_benchmark'
    file_type_list = ['jpg','png']
    
    
    ######################################################################
    #可能需要修改的参数    (旋转角度列表)
    #rotation_list = [15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345]
    rotation_list = [180]
    #是否翻转图片  列表中包含'tb'==上下翻转，'lr'==左右翻转
    flip_list = ['tb','lr']
    ####################################################################

   
   
    for file_type in file_type_list:
        image_list = get_files(file_path, file_type)
        for image_path in tqdm(image_list):
            for rotation in rotation_list:        
                aug_img_json(image_path , rotation)
    
    for file_type in file_type_list:
        image_list = get_files(file_path, file_type)
        for image_path in tqdm(image_list):
            for flip_way in flip_list:        
                aug_flip_img_json(image_path , flip_way )
