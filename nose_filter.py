%matplotlib inline
import numpy as np
import scipy as sp
import numpy as np
import os
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from PIL import Image
import math as m

def Grayscale(image):
    img = imread(image)
    width = img.shape[1]
    height = img.shape[0]
   
    for i in range(height):
        for j in range(width):
            rgb = img[i,j,0]/3 + img[i,j,1]/3 + img[i,j,2]/3
            img[i,j,0] = rgb
            img[i,j,1] = rgb
            img[i,j,2] = rgb

    return img


def add_salt_and_pepper(input_image2, noise_probability2):      #이미지를 입력받아 grayscale한다.
    input_image = imread(input_image2)   # 노이즈를 추가 할 입력 이미지 파일 이름 
    gray_image = Grayscale(input_image2)
    noise_probability = noise_probability2   #각 픽셀당 노이즈가 발생할 확률 값
    Blacknoise_p = 0.5
    Whitenoise_p = 1 - 0.5
   
    noisy = gray_image.copy()

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            randfixel = np.random.random(1) #이미지의 "세로 X 세로" pixel 각각에 0~1사이의값을대입
            if randfixel <= noise_probability:
                if randfixel <= Blacknoise_p:
                    noisy[i][j][0] = 0
                else:
                    noisy[i][j][0] = 255
                    
    noisy[:,:,0] =  noisy[:,:,0]
    noisy[:,:,1] =  noisy[:,:,0]
    noisy[:,:,2] =  noisy[:,:,0]
    
    plt.imshow(noisy)
    plt.show()
    return noisy


def add_gaussian(input_image2, noise_probability2, standard_deviation):      #이미지를 입력받아 grayscale한다.
    input_image = imread(input_image2)   # 노이즈를 추가 할 입력 이미지 파일 이름 
    noise_probability = noise_probability2   #각 픽셀당 노이즈가 발생할 확률 값
    gray_image = Grayscale(input_image2)
    deviation2 = standard_deviation * standard_deviation    
    randfixel = np.random.rand(input_image.shape[0], input_image.shape[1]) #이미지의 "세로*세로" pixel 각각에 0~1사이의값을대입    
    
    fixel_R = gray_image[:,:,0]
    fixel_G = gray_image[:,:,1]
    fixel_B = gray_image[:,:,2]
    
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            if randfixel[i][j] > noise_probability:
                n = np.random.normal(0,deviation2)
                if (n + fixel_R[i][j]) <= 255:
                    fixel_R[i][j] += n
                if (n + fixel_G[i][j]) <= 255:
                    fixel_G[i][j] += n
                if (n + fixel_B[i][j]) <= 255:
                    fixel_B[i][j] += n 
    
    gray_image[:,:,0] = fixel_R
    gray_image[:,:,1] = fixel_G
    gray_image[:,:,2] = fixel_B
    plt.imshow(gray_image)
    plt.show()
    return gray_image


def detect_arround(n, fixel_x, index_ocha):  #n[x,y] 주변에서 중간값을 검사
    arround_x = []
    width = fixel_x.shape[1]     #사진의 세로 512(0~511)
    height = fixel_x.shape[0]    #사진의 가로 512(0~511)
    
    for i in range(-index_ocha,index_ocha+1):   #범위끝은 '+1' 이여야 한다.
        for j in range(-index_ocha,index_ocha+1):
            if (n[0] + i >= 0) and (n[0] + i < height):   #picture의 index범위를 넘어가지 않는다면,
                if (n[1] + j >= 0) and (n[1] + j < width):
                    arround_x.append(fixel_x[n[0]+i][n[1]+j])
    
    arround_x.sort()
    middle_value = arround_x[len(arround_x) / 2]
    arround_x = [0,0]
    return middle_value
   

def remove_noise(input_image2, filter_size2):
    input_image = input_image2.copy()
    width = input_image.shape[1]     #사진의 세로 512(0~511)
    height = input_image.shape[0]    #사진의 가로 512(0~511)
    filter_size = filter_size2       #filter size는 홀수
    index_ocha = filter_size / 2     #검사해야할 index의 오차
    n = [0,0]
    
    
    fixel_R = input_image[:,:,0].copy()

    for i in range(height):           #'fixel_R'지우기 i= 0~511
        n[0] = i
        for j in range(width):        # j= 0~511
            n[1] = j
            fixel_R[n[0]][n[1]] = detect_arround(n, input_image2[:,:,0].copy(), index_ocha)            
        n[1] += 0
              
    n = [0,0]
                
    input_image[:,:,0] = fixel_R
    input_image[:,:,1] = fixel_R
    input_image[:,:,2] = fixel_R  
    plt.imshow(input_image)
    plt.show()
    return input_image
    


def plot_error(before_image, after_image):
    width = before_image.shape[1]     #사진의 세로 512(0~511)
    height = before_image.shape[0]    #사진의 가로 512(0~511)
    sume = 0.0
    E = 0.0
    d = width * height
    
    for i in range(height):
        for j in range(width):
            a = float(after_image[i][j][0])
            b = float(before_image[i][j][0]) 
            sume += float((b - a)*(b - a))
            sume = sume/d
            E += sume
            sume = 0   

    return  m.sqrt(E)

#sume += (before_image[i][j][0] - after_image[i][j][0]) * (before_image[i][j][0] - after_image[i][j][0])
           
            


original_picture = Grayscale('c:\users\sooyoung\Lenna.jpg')
noise_picture1 = add_salt_and_pepper('c:\users\sooyoung\Lenna.jpg',0.2)   #사진경로, 노이즈확률을 매개변수로 img 생성
filter_picture1 = remove_noise(noise_picture1, 3)   # 3 x 3 filter 적용
print plot_error(noise_picture1,filter_picture1)    # -> 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture1 = remove_noise(noise_picture1,5)  # 5 x 5 filter 적용
print plot_error(noise_picture1, filter_picture1)  # -> 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture, filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture1 = remove_noise(noise_picture1,7)  # 7 x 7 filter 적용
print plot_error(noise_picture1,filter_picture1)  # -> 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture1 = remove_noise(noise_picture1,9)  # 9 x 9 filter 적용
print plot_error(noise_picture1,filter_picture1)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture2 =add_salt_and_pepper('c:\users\sooyoung\Lenna.jpg',0.4)   #사진경로, 노이즈확률을 매개변수로 img 생성
filter_picture2 = remove_noise(noise_picture2,3)  # 3 x 3 filter 적용
print plot_error(noise_picture2,filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture2 = remove_noise(noise_picture2,5)  # 5 x 5 filter 적용
print plot_error(noise_picture2, filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture2 = remove_noise(noise_picture2,7)  # 7 x 7 filter 적용
print plot_error(noise_picture2, filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



filter_picture2 = remove_noise(noise_picture2,9)  # 9 x 9 filter 적용
print plot_error(noise_picture2, filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture3 =add_salt_and_pepper('c:\users\sooyoung\Lenna.jpg',0.6)   #사진경로, 노이즈확률을 매개변수로 img 생성

filter_picture3 = remove_noise(noise_picture3,3)  # 3 X 3 filter적용 
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture3 = remove_noise(noise_picture3,5)  # 5 x 5 filter 적용
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



filter_picture3 = remove_noise(noise_picture3,7)  # 7 x 7 filter 적용
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



filter_picture3 = remove_noise(noise_picture3,9)  # 9 x 9 filter 적용
print plot_error(noise_picture3, filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture4 =add_salt_and_pepper('c:\users\sooyoung\Lenna.jpg',0.8)   #사진경로, 노이즈확률을 매개변수로 img 생성
filter_picture4 = remove_noise(noise_picture4,3)  # 3 x 3 filter 적용
print plot_error(noise_picture4,filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



filter_picture4 = remove_noise(noise_picture4,5)  # 5 x 5 filter 적용
print plot_error(noise_picture4, filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture4 = remove_noise(noise_picture4,7)  # 7 x 7 filter 적용
print plot_error(noise_picture4, filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


filter_picture4 = remove_noise(noise_picture4,9)  # 9 x 9 filter 적용
print plot_error(noise_picture4,filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력




noise_picture1 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.2, 2)   #noise probabaility :0.2 standard deviation: 2
filter_picture1 = remove_noise(noise_picture1,7)  # filter size :7
print plot_error(noise_picture1,filter_picture1)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture2 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.2, 4)   #noise probabaility :0.2 standard deviation: 4
filter_picture2 = remove_noise(noise_picture2,7)  # filter size : 7
print plot_error(noise_picture2,filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture3 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.2, 6)   #noise probabaility :0.2 standard deviation: 6
filter_picture3 = remove_noise(noise_picture3,7)  # filter size : 7
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture4 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.2, 8)   #noise probabaility :0.2 standard deviation: 8
filter_picture4 = remove_noise(noise_picture4,7)  # filter size : 7
print plot_error(noise_picture4,filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture1 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.4, 2)   #noise probabaility :0.4 standard deviation: 2
filter_picture1 = remove_noise(noise_picture1,7)  # filter size :7
print plot_error(noise_picture1,filter_picture1)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture2 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.4, 4)   #noise probabaility :0.4 standard deviation: 4
filter_picture2 = remove_noise(noise_picture2,7)  # filter size : 7
print plot_error(noise_picture2,filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture3 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.4, 6)   #noise probabaility :0.4 standard deviation: 6
filter_picture3 = remove_noise(noise_picture3,7)  # filter size : 7
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture4 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.4, 8)   #noise probabaility :0.4 standard deviation: 8
filter_picture4 = remove_noise(noise_picture4,7)  # filter size : 7
print plot_error(noise_picture4,filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture1 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.6, 2)   #noise probabaility :0.6 standard deviation: 2
filter_picture1 = remove_noise(noise_picture1,7)  # filter size :7
print plot_error(noise_picture1,filter_picture1)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture2 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.6, 4)   #noise probabaility :0.6 standard deviation: 4
filter_picture2 = remove_noise(noise_picture2,7)  # filter size : 7
print plot_error(noise_picture2,filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력

noise_picture3 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.6, 6)   #noise probabaility :0.6 standard deviation: 6
filter_picture3 = remove_noise(noise_picture3,7)  # filter size : 7
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture4 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.6, 8)   #noise probabaility :0.6 standard deviation: 8
filter_picture4 = remove_noise(noise_picture4,7)  # filter size : 7
print plot_error(noise_picture4,filter_picture4)   # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력


noise_picture1 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.8, 2)   #noise probabaility :0.8 standard deviation: 2
filter_picture1 = remove_noise(noise_picture1,7)  # filter size :7
print plot_error(noise_picture1,filter_picture1)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture1) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture2 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.8, 4)   #noise probabaility :0.8 standard deviation: 4
filter_picture2 = remove_noise(noise_picture2,7)  # filter size : 7
print plot_error(noise_picture2,filter_picture2)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture2) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture3 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.8, 6)   #noise probabaility :0.8 standard deviation: 6
filter_picture3 = remove_noise(noise_picture3,7)  # filter size : 7
print plot_error(noise_picture3,filter_picture3)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture3) #-> 원본그림(흑백), 필터그림 적용결과 E 출력



noise_picture4 = add_gaussian('c:\users\sooyoung\Lenna.jpg',0.8, 8)   #noise probabaility :0.8 standard deviation: 8
filter_picture4 = remove_noise(noise_picture4,7)  # filter size : 7
print plot_error(noise_picture4,filter_picture4)  # 노이즈그림, 필터그림 적용결과 E 출력
print plot_error(original_picture,filter_picture4) #-> 원본그림(흑백), 필터그림 적용결과 E 출력




|probability|0.2|0.4|0.6|0.8||salt and pepper noise_picture vs filter_picture compare|
|--------|:------:|-----:|-----:|-----:|
| 3 X 3 filter|    58.45    |  76.65    |  96.31    |  125.05    |  
| 5 X 5 filter|    58.79    |  79.97    |  101.61   |  129.53    |
| 7 X 7 filter|    58.92    |  80.10    |  103.53    | 131.159    |
| 9 X 9 filter|    58.99    |  79.56    |  104.33    | 132.50     |   |



|probability|0.2|0.4|0.6|0.8||original picture vs filter_picture compare|
|--------|:------:|-----:|-----:|-----:|
| 3 X 3 filter|    20.03    |  69.44    |  95.31    |  104.06    |
| 5 X 5 filter|    10.20    |  54.06    |  95.75   |  97.11   |
| 7 X 7 filter|    11.71    |  41.01    |  96.38    | 96.13    |
| 9 X 9 filter|    13.22    |  32.25    |  96.91    | 96.31     |   |



| gaussian noise  probability|0.2|0.4|0.6|0.8||gaussian noise_picture vs filter_picture compare|
|----------|:------:|-----:|-----:|-----:|
| deviation : 2|    9.37    |  9.21   |  9.03  | 8.84     |
| deviation : 4|    16.81   |  15.20  |  13.34 | 11.23    |
| deviation : 6|    35.11   |  30.81  |  25.53 | 18.93    |
| deviation : 8|    53.11   |  47.32  |  39.26 | 28.50    |   |
  


| gaussian noise  probability|0.2|0.4|0.6|0.8||original picture vs filter_picture compare|
|----------|:------:|-----:|-----:|-----:|
| deviation : 2|    3.72    |  8.71    |  8.69    |  8.66    |
| deviation : 4|    9.17    |  8.97    |  8.83    |  8.74    |
| deviation : 6|    10.27   |  9.60    |  9.14    |  8.85    |
| deviation : 8|    13.82   |  10.86   |  9.63    |  8.98    |   |



ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [58.45 , 76.65, 96.31, 125.05]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [20.03, 69.44, 95.31, 104.06]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('salt, pepper_noise(filter size : 3X3)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')


ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [ 58.79 , 79.97, 101.61 , 129.53 ]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [10.20 , 54.06 , 95.75 , 97.11]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('salt, pepper_noise(filter size : 5X5)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')



ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [ 58.92 , 80.10 , 103.53, 131.159  ]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [11.71 , 41.01 ,96.38 , 96.13]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('salt, pepper_noise(filter size : 7X7)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')


ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [ 58.99 ,79.56 ,104.33 , 132.50 ]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [13.22 , 32.25 , 96.91, 96.31 ]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('salt, pepper_noise(filter size : 9X9)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')


ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [ 9.37, 9.21 ,9.03, 8.84 ]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [3.72 , 8.71, 8.69 ,  8.66   ]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('gussian probability(filter size :7X7 and standard deviation: 2)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')


 ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [ 16.81, 15.20 ,13.34, 11.23 ]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [9.17 , 8.97, 8.83 ,8.74  ]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('gussian probability(filter size :7X7 and standard deviation: 4)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')


 ax = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]
x = [ 35.11 ,30.81 ,25.53 ,18.93]
ax.plot(n,x,'b')   #'ax'는 blue


bx = plt.subplot(1,1,1)
n = [0.2, 0.4, 0.6, 0.8]  
y = [10.27, 9.60, 9.14 ,8.85]
bx.plot(n,y,'r')  #'bx'는 red

bx.set_title('gussian probability(filter size :7X7 and standard deviation: 6)')
bx.set_xlabel('noise_probility\nblue:salt and pepper,red:filter')
bx.set_ylabel('difference with original')


