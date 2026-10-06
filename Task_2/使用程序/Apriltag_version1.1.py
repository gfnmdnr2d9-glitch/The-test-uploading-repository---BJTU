import cv2
import numpy as np
from pupil_apriltags import Detector
import math


# 1. 连接摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60) # 帧率 帧/秒

    # 2. Detector初始化
detector = Detector(
    families = 'tag36h11',   # 标签家族
    nthreads = 1,            # 线程数
    quad_decimate = 1.0,     # 图像降采样比例
    quad_sigma = 0.325,        # 高斯模糊标准差
    refine_edges = 1,        # 是否优化边缘
    decode_sharpening = 0.325 # 解码锐化程度
)
TAG_SIZE = 0.111

fourcc = cv2.VideoWriter_fourcc(*'mp4v') # VideoWriter_fourcc函数定义编码器
out = cv2.VideoWriter("/home/feizhou/文档/Data//Task_2/Apriltag_version1.0_information.mp4", fourcc, 15, (1280, 720))

if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

while True:
    ret, image = cap.read() # 读取结果，当前图像矩阵
    if not ret:
        break

    # 3. 加载之前标定保存的参数文件
    # 如果写了绝对路径就用绝对路径，比如 '/home/feizhou/文档/camera_params.npz'
    params = np.load('/home/feizhou/文档/Data/Task_2/camera_params.npz') 
    # 提取数据（标签名就是我们保存时起的名字）
    camera_matrix = params['mtx']    # 3x3 内参矩阵
    dist_coeffs = params['dist']     # 畸变系数 (比如 1x5)
    img_w = int(params['width'])     # 标定时的图像宽度
    img_h = int(params['height'])    # 标定时的图像高度
    # 从矩阵中拆出 apriltag 需要的四个参数
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    # 4. 检测
    tags = detector.detect(
        gray,                           # gray是灰度图
        estimate_tag_pose = True,
        camera_params = (fx, fy, cx, cy),  # 你的相机内参(fx、fy 是以像素为单位的焦距参数，cx、cy 是主点位置)
        tag_size = TAG_SIZE                  # 实测边长，单位：米
    ) # 会返回det.tag_id(标签的ID号)、det.center(中心坐标)、det.corners(四个角点坐标)、det.pose_R(旋转矩阵)、det.pose_t(平移矩阵)

    # 5. 从所有结果中选出ID 0
    target = None
    for det in tags:
        if det.tag_id == 0:
            target = det
            break

    if target is not None:
        corners = target.corners.astype(int)
        center = target.center.astype(int)
        R = target.pose_R
        t = target.pose_t

        # 6. 绘制角点
        for i in range(4):
            cv2.line(image, tuple(corners[i]),tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)

        # 旋转向量
        rvec, _ = cv2.Rodrigues(R)

        # 提取 x, y, z 坐标
        x = t[0, 0]
        y = t[1, 0]
        z = t[2, 0]
        # 用 sqrt(x*x + y*y + z*z) 计算直线距离（单位：米）
        distance = math.sqrt(x*x + y*y + z*z)

        # 7. 绘制坐标轴
        axis_length = TAG_SIZE / 2
        axis_3d = np.float32([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ])
        imgpts, _ = cv2.projectPoints(axis_3d, rvec, t, camera_matrix, dist_coeffs)
        origin = tuple(imgpts[0].ravel().astype(int))
        cv2.line(image, origin, tuple(imgpts[1].ravel().astype(int)), (0, 0, 255), 3)
        cv2.line(image, origin, tuple(imgpts[2].ravel().astype(int)), (0, 255, 0), 3)
        cv2.line(image, origin, tuple(imgpts[3].ravel().astype(int)), (255, 0, 0), 3)

        # 8. 显示信息
        cv2.putText(image, f"ID:{target.tag_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(image, f"t=({t[0,0]:.3f},{t[1,0]:.3f},{t[2,0]:.3f})m",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, f"distant={distance:.3f}m", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # 显示平移向量 t 的三个分量（单位：米）
        cv2.putText(image, f"t=(x:{t[0,0]:.2f}, y:{t[1,0]:.2f}, z:{t[2,0]:.2f})m",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # 显示旋转向量 rvec 的三个分量（单位：弧度）
        cv2.putText(image, f"rvec=({rvec[0,0]:.2f}, {rvec[1,0]:.2f}, {rvec[2,0]:.2f})",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)      
    else:
        cv2.putText(image, "Target ID 0 not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow('AprilTag', image)
    out.write(image)

    if cv2.waitKey(25) & 0xFF == ord('q'): # 退出条件
        break

cap.release()
out.release()
cv2.destroyAllWindows()