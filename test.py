import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# 연산에 사용한 디바이스 정보 출력 설정
tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)
print(tf.__version__)