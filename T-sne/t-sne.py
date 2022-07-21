
import glob
import os
import time
from datetime import datetime
from multiprocessing import Pool
from queue import Queue

import cv2
import matplotlib.pyplot as plt
import numpy
from sklearn.manifold import TSNE
from tqdm import tqdm


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

def main(i):
    #img_dir = r"E:\PyProjects\Report\T-sne\N2\Image"  # Enter Directory of all images
    img_dir = r".\003a&020a\003a-sd"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    all_files = glob.glob(data_path)
    data = []
    for num, f1 in enumerate(files):
        img = cv2.imread(f1)
        img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
        deg = numpy.random.randint(-10,10)
        img = rotate_image(img,deg)
        img = cv2.resize(img, (60, 60))
        data.append(img.reshape(-1))

    

    L = len(data)



    img_dir = r".\003a&020a\020a-sd"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    all_files = all_files.append(glob.glob(data_path))
    for num, f1 in enumerate(files):
        img = cv2.imread(f1)
        img = cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
        deg = numpy.random.randint(-10, 10)
        img = rotate_image(img, deg)
        img = cv2.resize(img, (60, 60))
        data.append(img.reshape(-1))

    


    tsne = TSNE(n_components=2, angle=0.1, perplexity=10 ,init='random', verbose=0, method='exact' ,learning_rate=100.0, n_iter=2000)
    #tsne = TSNE(n_components=2, angle=0.1)
    tsne_results = tsne.fit_transform(data)
    
    # print(tsne_results)

    plt.figure(dpi=1000,figsize=(5,4))
    fontsize = 12
    ax=plt.gca()  #gca:get current axis得到当前轴
    #设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ml003a, = plt.plot(tsne_results[:L, 0], tsne_results[:L, 1], 'o', color = 'lightgreen')
    ml020a, = plt.plot(tsne_results[L+1:,0],tsne_results[L+1:,1],'o', color = 'royalblue')
    plt.legend(handles = [ml003a, ml020a, ], labels = ['Molecules A','Molecules B'],loc = 'best', frameon=False)
    plt.xticks(weight='bold',fontsize = fontsize)
    plt.yticks(weight='bold',fontsize = fontsize)
    ax.spines['bottom'].set_linewidth(1.7);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.7);####设置左边坐标轴的粗细
    plt.xlabel('Dimension 1',weight='bold')
    plt.ylabel('Dimension 2',weight='bold')
    # plt.figure()
    # plt.plot(tsne_results[:L, 0], tsne_results[:L, 1], 'bo')


    # plt.figure()
    # plt.plot(tsne_results[L+1:,0],tsne_results[L+1:,1],'ro')
    # plt.show()
    plt.savefig(f".\\resutls2\{i}.png",bbox_inches='tight',dpi=1000)
 


if __name__ == '__main__':
    # main(1)
    res = []
    re_list =[]
    a = datetime.now()
    queue = Queue()                                                                         #创建一个queue队列
    thread_num = 16                                                                         #设置同时运行的进程数量
    print("start! ")
    pthread = Pool(thread_num)                                                              #创建一个进程池

    for i  in tqdm(range(100)):

        queue.put(pthread.apply_async(main, args=(i,)))                                 #给函数赋值并将100个子程序put进入queue队列

    pthread.close()                                                                         #主进程等待所有子进程中的子程序运行完毕
    pthread.join()                                                                          #关闭多进程
    b = datetime.now()
    print('time:',(b-a).seconds)


