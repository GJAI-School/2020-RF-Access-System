import socket
import numpy as np
import cv2
import requests

#cap = cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
URL = 'http://192.168.137.227:8000/accesshistory_save/'
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf
HOST=''
PORT=8493
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')
s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')
conn,addr=s.accept()
while True:
    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    result, encode_frame = cv2.imencode('.jpg', frame, encode_param)
    data = np.array(encode_frame)
    redimg = data.tostring()
    send_data = stringData + b':::::' + redimg
    
    print(stringData)
    
    requests.post(URL, data=send_data)
    cv2.imshow('frame.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cam.release()
cv2.destroyAllWindows()