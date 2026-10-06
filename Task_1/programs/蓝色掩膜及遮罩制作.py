# 导入 OpenCV 库
import cv2# 导入opencv库
import numpy as np

# 1. 读取图像
image_path = "/home/feizhou/下载/test1.jpg"#设置路径
image = cv2.imread(image_path)# 读取图片文件



# 检查图像是否成功读取
if image is None:
    print("错误：无法加载图像，请检查路径是否正确。")
    exit()

# b , g , r = cv2.split(image)#split拆分通道

image = cv2.GaussianBlur(image, (5, 5), 0)# GaussianBlur高斯滤波先对图像进行降噪（图像，（ ， 核大小）， x方向的标准差）

hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)# COLOR_BGR2 HSV转换为HSV格式（色相，饱和度，明度），GRAY转换成灰度图
mask_blue = cv2.inRange(hsv ,(90 , 80 , 60),(135 , 255 , 255))# 制作蓝色掩膜，inRange遍历每一个像素，将在范围外的点变黑，范围内的变白，输出单通道图

# result = cv2.bitwise_and(image , image , mask = mask_blue)# bitwise_and函数通过遮罩，将原图像和遮罩白色区块重合的地方保留

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))# getStructuringElement函数制作核（也可以说是画笔），MORPH_RECT是矩形，(4, 4)是大小4*4的像素
opened = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)# morphologyExu函数对遮罩过后的图像进行优化降噪
closed_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)# MORPH_OPEN先缩小再变大，MORPH_CLOSE先变大再缩小

contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)# RETR_EXTERNAL表示只检测最外层轮廓，CHAIN_APPROX_SIMPLE表示只保留端点

for contour in contours :
    box = cv2.minAreaRect(contour)# minAreaRect返回值是中心、宽高、角度 ex.（（1,1），（1,2），40.0）注意：这一步返回值就是浮点数！！！
    
    corners = cv2.boxPoints(box)# boxPoints将参数转换为直角坐标，便于阅读——个人认为经过计算可输出作为定位用参数
    
    corners = np.int32(corners)# drawContours函数只能接受整数！！！需要格式转换（没有四舍五入） //或者corners = corners.astype(np.int32)
    cv2.drawContours(image, [corners], 0, (0, 255, 0), 2)# drawContours函数在原图上画 
    # 注意！第二个参数需是列表，第三个参数（-1：全部；0：第一个），第四个参数是颜色，第五个参数是框的大小

cv2.imshow("Display Image" , image)# imshow函数输出图像，建议转换成BGR格式图像输出
cv2.waitKey(0)

# 5. 关闭所有窗口
cv2.destroyAllWindows()