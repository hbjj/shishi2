import cv2
import os
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import skimage.transform as transform



def pic_detect(filename):
    # cv2级联分类器CascadeClassifier.xml文件为训练数据
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#人脸
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#人眼
    mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth - 副本.xml')#人嘴
    
    img = cv2.imread(filename) # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# 转灰度图
    faces = face_cascade.detectMultiScale(gray, 1.1, 5,minSize = (5,5))#  进行人脸检测，调整参数
    eyes = eye_cascade.detectMultiScale(gray,1.1,5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.1,5)
    print('发现了{0}个人的脸！'.format(len(faces)))

    # 绘制人脸矩形框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y),(x+w, y+h), (0, 250, 0), 1)#中心位置、长宽、画笔颜色及大小
        print(x+w/2,y+h/2)
        for (ex, ey, ew, eh) in eyes:
            img = cv2.rectangle(img, (ex, ey),(ex + ew, ey + eh), (250, 0, 0), 1)
            print(ex+ew/2,ey+eh/2)  #输出左眼右眼中心坐标
        for (mx, my, mw, mh) in mouth:
            img = cv2.rectangle(img, (mx, my),(mx + mw, my + mh), (250, 0, 0), 1)
            print(mx+mw/2,my+mh/2)  #输出嘴巴中心坐标

                
    img = Image.open(os.path.join(filename))

    plt.figure("Image") # 图像窗口名称
    plt.imshow(img)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()    
    
    img = cv2.imread(filename)
    
    img_blank = np.zeros((h, w, 3), np.uint8)
    for k in range(len(faces)):
        for i in range(h):
            for j in range(w):
                img_blank[i][j] = img[y+i][x+j]
        cv2.imshow("face_"+str(k+1), img_blank)
        cv2.imwrite("img_face_"+str(k+1)+".jpg", img_blank)

        cut = Image.open(os.path.join("img_face_"+str(k+1)+".jpg"))
        
    scaled_img = transform.resize(cut, [101, 101], mode='constant')   #使得输出的面部区域大小为 101 行*101 列
    plt.imshow(scaled_img)


# 1.找到眼睛倾斜的角度和两眼距离

    p1 = np.array(eyes[0])[::-1] # 左眼坐标
    p2 = np.array(eyes[1])[::-1] # 右眼坐标
    dist = np.sqrt(np.sum(p1-p2)**2) # 两只眼睛之间的距离

    dp = p1 - p2
    angle = np.arctan(dp[0] / dp[1])
    
    # 旋转图片
    rot_img = ndimage.rotate(faces, angle=+angle*180/np.pi)
    # 旋转后图像的中点
    rot_image_center = np.array((np.array(rot_img.shape[:2]) - 1) / 2, 
    dtype=np.int)
# 2. 在旋转后的图片中找到眼睛的坐标
    # 原两眼距离的中点
    org_eye_center = np.array((p1 + p2) / 2, dtype=np.int)
    # 原图像的中点
    org_image_center = np.array((np.array(faces.shape[:2]) - 1) / 2, dtype=np.int)
    # 以图片中心进行旋转，在旋转后的图片中找到眼睛的中点
    R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    rot_eye_center = np.dot(R, org_eye_center[::-1]-org_image_center[::-1])[::-1] + rot_image_center
    rot_eye_center = np.array(rot_eye_center, dtype=int)
    mid_y, mid_x = rot_eye_center   #根据眼睛坐标找到包含面部的框的宽度和高度
    rot_eye_center = ([25,24],[25,76]) #指定调整之后眼镜和嘴巴的坐标
    rot_mouth_center = ([75,50])
    MUL = 2
    y_top = int(max(mid_y - MUL * dist, 0))
    y_bot = int(min(mid_y + MUL * dist, rot_img.shape[0]))
    x_left = int(max(mid_x - MUL * dist, 0))
    x_right = int(min(mid_x + MUL * dist, rot_img.shape[1]))

    cropped_img = rot_img[y_top:y_bot+1, x_left:x_right+1, :]
    scaled_img = transform.resize(cropped_img, [101, 101], mode='constant')   #裁剪图像的尺寸为 101*101
    plt.imshow(scaled_img)

#两种不同灰度插值方法下的输出图像
    img1 = cv2.imread(filename,0)    # 参数0为灰度，1为彩色
    img2 = cv2.imread(filename,1)
    cv2.imshow('src', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('src', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return None

filename = 'lena.png' 
pic_detect(filename)
