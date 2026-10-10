import serial
import time

ser = serial.Serial('/dev/pts/2', 115200, timeout = 0.1)
#print(serial.__file__)
#print(serial.Serial)
x= 0

while x <= 1:
    if x == 0:
        packet = "$CV1,,42,12345,1,0,100.0,-50.0,800.0,0.000000,0.000000,0.000000*33"
    else:
        packet = "$CV1,43,12445,0,-1,0.0,0.0,0.0,0.000000,0.000000,0.000000*09"
    
    full_packet = f"{packet}\r\n"
    ser.write(full_packet.encode('ascii'))
    print(full_packet)

    time.sleep(3)

    x += 1

ser.close()