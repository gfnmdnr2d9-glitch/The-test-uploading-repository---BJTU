# 导入 OpenCV 库
import cv2#导入opencv库
import numpy as np

# 1. 读取图像
image_path = "/home/feizhou/下载/test1.jpg"#设置路径
image = cv2.imread(image_path)#读取图片文件

# 检查图像是否成功读取
if image is None:
    print("错误：无法加载图像，请检查路径是否正确。")
    exit()

b , g , r = cv2.split(image)#split拆分通道
'''
image = cv2.GaussianBlur(image, (5, 5), 0)#GaussianBlur高斯滤波先对图像进行降噪（图像，（ ， 核大小）， x方向的标准差）

hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)#COLOR_BGR2 HSV转换为HSV格式（色相，饱和度，明度），GRAY转换成灰度图
mask_blue = cv2.inRange(hsv ,(90 , 80 , 60),(135 , 255 , 255))#制作蓝色掩膜，inRange遍历每一个像素，将在范围外的点变黑，范围内的变白，输出单通道图
result = cv2.bitwise_and(image , image , mask = mask_blue)#bitwise_and函数通过遮罩，将原图像和遮罩白色区块重合的地方保留

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))#getStructuringElement函数制作核（也可以说是画笔），MORPH_RECT是矩形，(4, 4)是大小4*4的像素
opened = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)#morphologyExu函数对遮罩过后的图像进行优化降噪
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)#MORPH_OPEN先缩小再变大，MORPH_CLOSE先变大再缩小
'''
'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
'''
cv2.imshow("Display Image" , r)#imshow函数输出图像，建议转换成BGR格式图像输出
# cv2.waitKey(0)

# 2. 显示图像
# 创建一个名为 "Display Image" 的窗口，并在其中显示图像

# cv2.imshow("Display Image", image)

# 3. 等待用户按键
# 参数 0 表示无限等待，直到用户按下任意键

key = cv2.waitKey(0)

# 4. 根据用户按键执行操作
if key == ord('s'):  # 如果按下 's' 键
    # 保存图像
    output_path = "/home/feizhou/图片/saved_image_version0.1_b.jpg"
    cv2.imwrite(output_path, b)
    print(f"图像已保存为 {output_path}")
else:  # 如果按下其他键
    print("图像未保存。")

# 5. 关闭所有窗口
cv2.destroyAllWindows()