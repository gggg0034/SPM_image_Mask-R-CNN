
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy
from sklearn.manifold import TSNE
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
        if each_name[-3:] == file_type:                                                     #请修改文件类型
            filename_list += [each_name]
    
    return filename_list

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def main(count, files1 = r".\003a&020a\003a", files2 = r".\003a&020a\020a"):
    #img_dir = r"E:\PyProjects\Report\T-sne\N2\Image"  # Enter Directory of all images
    all_name_list = []
    all_files1 = get_files(files1,'png')
    all_files2 = get_files(files2,'png')
    all_files = all_files1 + all_files2
    for path in all_files:
        image_fullname = os.path.basename(path)                                         #获取文件全名  含尾缀
        
        image_fbname = image_fullname.split('.')

        image_frontname = image_fbname[0]                                               #获取文件名  无尾缀  

        all_name_list.append(image_frontname)



    data_path = os.path.join(files1, '*g')
    files = glob.glob(data_path)
    
    data = []
    for num, f1 in enumerate(files):
        img = cv2.imread(f1)                                                            #读取图片文件
        img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)                                  #转换颜色通道
        deg = numpy.random.randint(-10,10)                                              #生成随机     
        img = rotate_image(img,deg)
        img = cv2.resize(img, (60, 60))
        data.append(img.reshape(-1))

    

    L = len(data)



      # Enter Directory of all images
    data_path = os.path.join(files2, '*g')
    files = glob.glob(data_path)

    for num, f1 in enumerate(files):
        img = cv2.imread(f1)
        img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
        deg = numpy.random.randint(-10, 10)
        img = rotate_image(img, deg)
        img = cv2.resize(img, (60, 60))
        data.append(img.reshape(-1))
    
    L2 = len(data)


    tsne = TSNE(n_components=2, angle=0.1, perplexity=10 ,init='random', verbose=0, method='exact' ,learning_rate=100.0, n_iter=2000)
    #tsne = TSNE(n_components=2, angle=0.1)
    tsne_results = tsne.fit_transform(data)

    # print(tsne_results)

    plt.figure()
    fontsize = 9
    ml003a, = plt.plot(tsne_results[:L, 0], tsne_results[:L, 1], 'wo')
    ml020a, = plt.plot(tsne_results[L+1:,0],tsne_results[L+1:,1],'wo')                              # 画出点  白色
    plt.legend(handles = [ml003a, ml020a, ], labels = ['003a','020a'],loc = 'best')

    for i in range(L):
        plt.text(tsne_results[i][0],tsne_results[i][1],all_name_list[i], size = 5 , color = 'r', style = 'normal', weight = 'bold' )   #标注文件名
    for i in range(L+1,L2):
        plt.text(tsne_results[i][0],tsne_results[i][1],all_name_list[i], size = 5 , color = 'b', style = 'normal', weight = 'bold' )   #标注文件名



    # plt.figure()
    # plt.plot(tsne_results[:L, 0], tsne_results[:L, 1], 'bo')


    # plt.figure()
    # plt.plot(tsne_results[L+1:,0],tsne_results[L+1:,1],'ro')
    # plt.show()
    plt.savefig(f".\\resutls2_annotation\{count}.png",dpi=300)



if __name__ == '__main__':
    for count in tqdm(range(50)):
        count = int(count+100)
        main(count, files1 = r".\003a&020a\003a-sgl", files2 = r".\003a&020a\020a-sgl")

'''
单组分分离良好     双组分分离异常图像    红42 33 蓝84
'''
