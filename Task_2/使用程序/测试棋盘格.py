import cv2
import numpy as np
import glob

# 棋盘格内角点数（比如9×6）
pattern_size = (9, 6)
# 实测方格边长（毫米）
square_size = 12.0

# 生成棋盘格角点的三维坐标
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints, imgpoints = [], []

#import glob

image_list = glob.glob('/home/feizhou/图片/test_pictures/*.jpg')
print(f"找到的图片数量: {len(image_list)}")

success_count = 0

for fname in image_list:
    img = cv2.imread(fname)
    if img is None:
        print(f"[读取失败] {fname}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        success_count += 1
        print(f"[成功检测] {fname}")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
    else:
        print(f"[未检测到棋盘格] {fname}")

print(f"总计成功检测的图片数量: {success_count}")

if success_count > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    print("内参矩阵:\n", mtx)
    print("畸变系数:", dist.ravel())
    print("重投影误差:", ret)
else:
    print("错误：没有任何图片成功检测到棋盘格，请检查棋盘格规格和图片质量！")
print("内参矩阵:\n", mtx)
print("畸变系数:", dist.ravel())
print("重投影误差:", ret)  # 越小越好，一般<0.5像素

# 保存
np.savez('camera_params.npz', mtx=mtx, dist=dist,
         width=gray.shape[1], height=gray.shape[0])