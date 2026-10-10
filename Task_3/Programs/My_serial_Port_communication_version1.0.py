import time
import serial
import cv2
import datetime
import def_Task_2 as apriltag_module

ser = serial.Serial('/dev/pts/2', 115200, timeout = 0.1)

cam_result = apriltag_module.Camera_Set(Camera = '/home/feizhou/文档/Apriltag_version1.1_information-添加R和t.mp4')
if cam_result.ret == 0:
    exit()
cap, out = cam_result.cap, cam_result.out

params = apriltag_module.Params_Set()
if params.ret == 0:
    exit()
detector = apriltag_module.Detector_Set()

seq = 0
start_time = time.time()

# 2. 循环外面打开日志文件
# 强烈建议使用英文路径，或者带时间戳的路径，方便查阅
#log_file = open("serial_log.txt", "w", encoding="utf-8")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 1. 获取结果（不再传入 cap）
    result, proceed_image, raw_tag = apriltag_module.Process_Frame(
        frame, detector, params.mix_location, 0.111,
        params.camera_matrix, params.dist_coeffs
    )
    # 2. 外部负责显示
    if proceed_image is not None:
        cv2.imshow("Apriltag", proceed_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 3. 组装 CV1 串口报文
    t_ms = int((time.time() - start_time) * 1000)

    if result.valid:
        x_mm = result.x * 1000.0
        y_mm = result.y * 1000.0
        z_mm = result.z * 1000.0
        packet = f"$CV1,{seq},{t_ms},1,{result.id},{x_mm:.1f},{y_mm:.1f},{z_mm:.1f},{result.rx:.6f},{result.ry:.6f},{result.rz:.6f}"
    else:
        packet = f"$CV1,{seq},{t_ms},0,-1,0.0,0.0,0.0,0.000000,0.000000,0.000000"

    checksum = 0
    for char in packet[1:]:
        checksum ^= ord(char)

    full_packet = f"{packet}*{checksum:02X}\r\n"
    # 4. 发送串口数据
    ser.write(full_packet.encode('ascii'))

    # 5. 写入日志（放在发送代码之后）
    # 记录时间和原始报文内容
    #log_file.write(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {full_packet}")
    #log_file.flush() # 💡 flush 非常重要：强制立即写入磁盘，防止程序意外崩溃导致日志丢失

    seq = (seq + 1) % (2 ** 32)
    # 5. 频率控制 (10Hz)
    time.sleep(0.1)

cap.release()
out.release()
ser.close()
cv2.destroyAllWindows()