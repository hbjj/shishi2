import numpy as np
import cv2
import scipy as sp
import pylab as pl
from scipy.optimize import leastsq  # 引入最小二乘函数

#图像配准

def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des
 
def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
 
def siftImageAlignment(img1,img2):
    kp1,des1 = sift_kp(img1)
    kp2,des2 = sift_kp(img2)
    goodMatch = get_good_match(des1,des2)
    if len(goodMatch) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
        #其中H为求得的单应性矩阵矩阵
        #status则返回一个列表来表征匹配成功的特征点。
        #ptsA,ptsB为关键点
        #cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关
        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgOut,H,status



img1 = cv2.imread('lalala.png')
img2 = cv2.imread('opopop.png')
while img1.shape[0] >  1000 or img1.shape[1] >1000:
    img1 = cv2.resize(img1,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
while img2.shape[0] >  1000 or img2.shape[1] >1000:
    img2 = cv2.resize(img2,None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)

# 最小二乘法
n = 9 
# 目标函数
def real_func(x):
    return np.sin(2 * np.pi * x)
 
# 多项式函数
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)
 
# 残差函数
def residuals_func(p, y, x):
    ret = fit_func(p, x) - y
    return ret
 
x = np.linspace(0, 1, 9)  # 随机选择9个点作为x
x_points = np.linspace(0, 1, 1000)  # 画图时需要的连续点
 
y0 = real_func(x)  # 目标函数
y1 = [np.random.normal(0, 0.1) + y for y in y0]  # 添加正太分布噪声后的函数
 
p_init = np.random.randn(n)  # 随机初始化多项式参数
 
plsq = leastsq(residuals_func, p_init, args=(y1, x))
 
print 'Fitting Parameters: ', plsq[0]  # 输出拟合参数
 
pl.plot(x_points, real_func(x_points), label='real')
pl.plot(x_points, fit_func(plsq[0], x_points), label='fitted curve')
pl.plot(x, y1, 'bo', label='with noise')
pl.legend()
pl.show()
    
    
    
result,_,_ = siftImageAlignment(img1,img2)
allImg = np.concatenate((img1,img2,result),axis=1)
cv2.namedWindow('1',cv2.WINDOW_NORMAL)
cv2.namedWindow('2',cv2.WINDOW_NORMAL)
cv2.namedWindow('Result',cv2.WINDOW_NORMAL)
cv2.imshow('1',img1)
cv2.imshow('2',img2)
cv2.imshow('Result',result)
 
if cv2.waitKey(2000) & 0xff == ord('q'):
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    
#图像拼接
leftgray = cv2.imread('lalala.png')
rightgray = cv2.imread('opopop.png')
 
hessian=400
surf=cv2.SURF(hessian) #将Hessian Threshold设置为400,阈值越大能检测的特征就越少
kp1,des1=surf.detectAndCompute(leftgray,None)  #查找关键点和描述符
kp2,des2=surf.detectAndCompute(rightgray,None)
 
FLANN_INDEX_KDTREE=0   #建立FLANN匹配器的参数
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5) #配置索引，密度树的数量为5
searchParams=dict(checks=50)    #指定递归次数
#FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
flann=cv2.FlannBasedMatcher(indexParams,searchParams)  #建立匹配器
matches=flann.knnMatch(des1,des2,k=2)  #得出匹配的关键点
 
good=[]
#提取优秀的特征点
for m,n in matches:
    if m.distance < 0.7*n.distance: #如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        good.append(m)

src_pts = np.array([ kp1[m.queryIdx].pt for m in good])    #查询图像的特征描述子索引
dst_pts = np.array([ kp2[m.trainIdx].pt for m in good])    #训练(模板)图像的特征描述子索引
H=cv2.findHomography(src_pts,dst_pts)         #生成变换矩阵
 
h,w=leftgray.shape[:2]
h1,w1=rightgray.shape[:2]
shft=np.array([[1.0,0,w],[0,1.0,0],[0,0,1.0]])
M=np.dot(shft,H[0])            #获取左边图像到右边图像的投影映射关系
dst_corners=cv2.warpPerspective(leftgray,M,(w*2,h))#透视变换，新图像可容纳完整的两幅图
cv2.imshow('xxxx1',dst_corners)   #显示，第一幅图已在标准位置
dst_corners[0:h,w:w*2]=rightgray  #将第二幅图放在右侧
cv2.imwrite('xxx.jpg',dst_corners)


#两种不同灰度插值方法下的输出图像
img1 = cv2.imread(filename,0)    # 参数0为灰度，1为彩色
img2 = cv2.imread(filename,1)
cv2.imshow('src', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('src', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
cv2.imshow('tiledImg',dst_corners)
cv2.imshow('leftgray',leftgray)
cv2.imshow('rightgray',rightgray)
cv2.waitKey()
cv2.destroyAllWindows()
