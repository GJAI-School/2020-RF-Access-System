from keras.models import load_model
import tensorflow as tf
import mtcnn

# 모델 불러오기
model = load_model(r'model\facenet_keras.h5')

print(model.inputs)
print(model.outputs)

print(tf.__version__)

print(mtcnn.__version__)