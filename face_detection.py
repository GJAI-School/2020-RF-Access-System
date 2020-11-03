import mtcnn

import cv2
import numpy as np
import matplotlib.pyplot as plt


# image load
img = cv2.imread(r'img\archive\data\train\ben_afflek\httpcsvkmeuaeccjpg.jpg', cv2.IMREAD_COLOR)
pixels = np.asarray(img)

# print(pixels)

# 얼굴 감지기 생성, 기본 가중치 이용
detector = MTCNN()

# 이미지에서 얼굴 감지
results = detector.detect_face(pixels)
