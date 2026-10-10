import cv2
import numpy as np
from pupil_apriltags import Detector
import math
from collections import namedtuple

ApriltagResults = namedtuple('ApriltagResults', ['valid', 'id', 'x', 'y', 'z', 'rx', 'ry', 'rz'])
fail_result = ApriltagResults(False, -1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# 1. 连接摄像头    
def Camera_Set(Camera = 0, Frame_Width = 1080, Frame_Height = 720, 
               Fps = 30, TAG_SIZE = 0.111,
               fourcc = 'mp4v', Save_Location = '/home/feizhou/文档/Data/Task_2/information.mp4'):

    Cameras = namedtuple('Cameras', ['ret', 'cap', 'out'])
    fail_Cameras = Cameras(0, None, None)

    
    cap = cv2.VideoCapture(Camera)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return fail_Cameras
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Frame_Width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Frame_Height)
    cap.set(cv2.CAP_PROP_FPS, Fps) # 帧率 帧/秒

    TAG_SIZE = TAG_SIZE
    fourcc = cv2.VideoWriter_fourcc(*fourcc) # VideoWriter_fourcc函数定义编码器
    out = cv2.VideoWriter(Save_Location, fourcc, Fps, (Frame_Width, Frame_Height))
    return Cameras(1, cap, out)

# 2. Detector初始化
def Detector_Set(families = 'tag36h11', nthreads = 1, quad_decimate = 1.0, quad_sigma = 0.325, refine_edges = 1, decode_sharpening = 0.325):
    
    detector = Detector(
        families = families,   # 标签家族
        nthreads = nthreads,            # 线程数
        quad_decimate = quad_decimate,     # 图像降采样比例
        quad_sigma = quad_sigma,        # 高斯模糊标准差
        refine_edges = refine_edges,        # 是否优化边缘
        decode_sharpening = decode_sharpening # 解码锐化程度
    )
    return detector


def Params_Set(Location = '/home/feizhou/文档/Data/Task_2/camera_params.npz'):
    
    Params = namedtuple('Params', ['ret', 'camera_matrix', 'dist_coeffs', 'img_w', 'img_h', 'mix_location'])
    fail_params = Params(0, None, None, 0, 0, (0.0, 0.0, 0.0, 0.0))

    try:
        # 加载之前标定保存的参数文件
        # 如果写了绝对路径就用绝对路径，比如 '/home/feizhou/文档/camera_params.npz'
        params = np.load(Location) 
        # 提取数据（标签名就是我们保存时起的名字）
        camera_matrix = params['mtx']    # 3x3 内参矩阵
        dist_coeffs = params['dist']     # 畸变系数 (比如 1x5)
        img_w = int(params['width'])     # 标定时的图像宽度
        img_h = int(params['height'])    # 标定时的图像高度
        # 从矩阵中拆出 apriltag 需要的四个参数
        fx = float(camera_matrix[0, 0])
        fy = float(camera_matrix[1, 1])
        cx = float(camera_matrix[0, 2])
        cy = float(camera_matrix[1, 2])
        mix_location = (fx, fy, cx, cy)
        return Params(1, camera_matrix, dist_coeffs, img_w, img_h, mix_location)
    
    except Exception as e:
        print(f'错误，找不到文件{Location}')
        return fail_params


def Process_Frame(image, detector, mix_location, TAG_SIZE, camera_matrix, dist_coeffs):

    if image is None:
        return fail_result, None, None
    
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    # 检测
    tags = detector.detect(
        gray, estimate_tag_pose = True,
        camera_params = mix_location, tag_size = TAG_SIZE # 实测边长，单位：米
    ) # 会返回det.tag_id(标签的ID号)、det.center(中心坐标)、det.corners(四个角点坐标)、det.pose_R(旋转矩阵)、det.pose_t(平移矩阵)

# 3. 从所有结果中选出ID 0
    target = None
    for det in tags:
        if det.tag_id in [0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8]:
            target = det
            break
    # 拆解坐标
    if target is not None:
        R = target.pose_R
        t = target.pose_t
        # 旋转向量
        rvec, _ = cv2.Rodrigues(R)
        # 提取 x, y, z 坐标
        x, y, z = t[0, 0], t[1, 0], t[2, 0]
        distance = math.sqrt(x*x + y*y + z*z)
# 4. 绘图
        # 绘制角点
        corners = target.corners.astype(int)
        center = target.center.astype(int)
        for i in range(4):
            cv2.line(image, tuple(corners[i]),tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.circle(image, tuple(center), 5, (0, 0, 255), -1)

        # 绘制坐标轴
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

        # 显示信息
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

        result = ApriltagResults(
            valid = True, id = target.tag_id,
            x = x, y = y, z = z,
            rx = rvec[0, 0], ry = rvec[1, 0], rz = rvec[2, 0]
        )
  
        return result, image, target
     
    else:
        cv2.putText(image, "Target not found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return fail_result, image, None