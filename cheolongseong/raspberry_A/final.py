import serial
import cv2
import socket
import numpy as np

# arduino part
port = "/dev/ttyACM0"
serialFromArduino = serial.Serial(port, 9600)
serialFromArduino.flushInput()


## socket part
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
## server ip, port
s.connect(('192.168.137.1', 8493))

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    input_s=serialFromArduino.readline()
    serialnumber = b'None'
    temp = b'None'
    
    if b"1KB" in input_s:
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        result, encode_frame = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(encode_frame)
        stringData = data.tostring()
    
    
    #if b"1KB" in input_s:
        input_s= input_s.split(b' 1KB ')[1]
        serialnumber, temp = input_s.split(b'::')
        temp = temp[:-2]
        print(serialnumber)
        print(temp)
        send_data = stringData + b':::::' + serialnumber + b':::::' + temp
        s.sendall((str(len(send_data))).encode().ljust(16) + send_data)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cam.release()
    
cv2.destroyAllWindows()