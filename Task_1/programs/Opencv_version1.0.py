# 1. 导入 OpenCV 库
import cv2
import numpy as np
import time

frame_count = 0 # 帧号初始化

cap = cv2.VideoCapture("/home/feizhou/视频/test_video2.mp4")
# 2. 获取视频信息
fps = cap.get(cv2.CAP_PROP_FPS) # CAP_PROP_FPS获取视频帧率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # CAP_PROP_FRAME_WIDTH获取宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # CAP_PROP_FRAME_HEIGHT获取高度
# 3. 定义编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # VideoWriter_fourcc函数定义编码器
# 4. 创建写入对象（在循环外！只创建一次）
out = cv2.VideoWriter("/home/feizhou/视频/TEST_output_version1.1_information.mp4", fourcc, fps, (width, height)) # 严格与输入视频一致（文件名，编码器，帧率，（宽，高））
# 5. 进入循环
while True:
    ret, image = cap.read() # 读取结果，当前图像矩阵
    if not ret:
        break

    lightnum_count = 0 #重置灯条个数

    # 开始计时
    tstart = time.perf_counter() # perf_counter函数进行精确计时

    # 蓝色掩膜制作
    #image = cv2.GaussianBlur(image, (3, 3), 0) # GaussianBlur高斯滤波先对图像进行降噪（图像，（奇数！ ， 核大小）， x方向的标准差）
    #image = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV) # COLOR_BGR2 HSV转换为HSV格式（色相，饱和度，明度），GRAY转换成灰度图
    mask_blue = cv2.inRange(hsv , (90, 80, 60), (135, 255, 255)) # 制作蓝色掩膜，inRange遍历每一个像素，将在范围外的点变黑，范围内的变白，输出单通道图
    # 优化掩膜
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) # getStructuringElement函数制作核（也可以说是画笔），MORPH_RECT是矩形，(4, 4)是大小4*4的像素
    opened = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel) # morphologyExu函数对遮罩过后的图像进行优化降噪
    closed_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) # MORPH_OPEN先缩小再变大，MORPH_CLOSE先变大再缩小
    # 大致框选
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL表示只检测最外层轮廓，CHAIN_APPROX_SIMPLE表示只保留端点
    # 逐一框选
    for contour in contours :
        area = cv2.contourArea(contour) # 计算轮廓面积
    # 过滤面积太小的噪点!!!
        if area > 25:
            box = cv2.minAreaRect(contour) # minAreaRect返回值是中心、宽高、角度 ex.（（1,1），（1,2），40.0）注意：这一步返回值就是浮点数！！！    
            corners = cv2.boxPoints(box) # boxPoints将参数转换为直角坐标，便于阅读——个人认为经过计算可输出作为定位用参数    
            corners = np.int32(corners) # drawContours函数只能接受整数！！！需要格式转换（没有四舍五入） //或者corners = corners.astype(np.int32)
            cv2.drawContours(image, [corners], 0, (0, 255, 0), 1) # drawContours函数在原图上画 
            # 注意！第二个参数需是列表，第三个参数（-1：全部；0：第一个），第四个参数是颜色，第五个参数是框的大小
            lightnum_count += 1 # 表示灯条数量

    frame_count += 1 # 表示帧号

    # 结束计时
    tend = time.perf_counter()
    costtime = (tend - tstart) * 1000 # ns转ms

    text_frame = f"Frame{frame_count}"
    text_number = f"Number{lightnum_count}"
    text_time = f"Time{costtime:.2f}" # 保留两位小数    
    # 帧号，灯条数量，处理时间（原图，字符串，（距左，距上），字体，字体缩放比例，文字颜色，粗细）
    cv2.putText(image, text_frame, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, text_number, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, text_time, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # imshow函数输出当前帧
    cv2.imshow("Display Image" , image)
    # 写入视频 
    out.write(image)
    
    if cv2.waitKey(25) & 0xFF == ord('q'): # 退出条件
        break

# 6. 释放资源
cap.release()
out.release() 

cv2.waitKey(0)

# 7. 关闭所有窗口
cv2.destroyAllWindows()