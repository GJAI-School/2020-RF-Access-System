import cv2
import yaml
import sys
import time
import numpy as np
import tensorflow as tf
from absl import logging
# from modules.dataset import load_tfrecord_dataset
from PIL import Image
import math
import os
from os import path
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                logging.info(
                    "Detect {} Physical GPUs, {} Logical GPUs.".format(
                        len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def load_dataset(cfg, priors, shuffle=True, buffer_size=10240):
    """load dataset"""
    logging.info("load dataset from {}".format(cfg['dataset_path']))
    dataset = load_tfrecord_dataset(
        tfrecord_name=cfg['dataset_path'],
        batch_size=cfg['batch_size'],
        img_dim=cfg['input_size'],
        using_bin=cfg['using_bin'],
        using_flip=cfg['using_flip'],
        using_distort=cfg['using_distort'],
        using_encoding=True,
        priors=priors,
        match_thresh=cfg['match_thresh'],
        ignore_thresh=cfg['ignore_thresh'],
        variances=cfg['variances'],
        shuffle=shuffle,
        buffer_size=buffer_size)
    return dataset


class ProgressBar(object):
    """A progress bar which can print the progress modified from
       https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
    def __init__(self, task_num=0, completed=0, bar_width=25):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width
                          if bar_width <= max_bar_width else max_bar_width)
        self.completed = completed
        self.first_step = completed
        self.warm_up = False

    def _get_max_bar_width(self):
        if sys.version_info > (3, 3):
            from shutil import get_terminal_size
        else:
            from backports.shutil_get_terminal_size import get_terminal_size
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            logging.info('terminal width is too small ({}), please consider '
                         'widen the terminal for better progressbar '
                         'visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def reset(self):
        """reset"""
        self.completed = 0
        self.fps = 0

    def update(self, inf_str=''):
        """update"""
        self.completed += 1

        if not self.warm_up:
            self.start_time = time.time() - 1e-1
            self.warm_up = True

        if self.completed > self.task_num:
            self.completed = self.completed % self.task_num
            self.start_time = time.time() - 1 / self.fps
            self.first_step = self.completed - 1
            sys.stdout.write('\n')

        elapsed = time.time() - self.start_time
        self.fps = (self.completed - self.first_step) / elapsed
        percentage = self.completed / float(self.task_num)
        mark_width = int(self.bar_width * percentage)
        bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
        stdout_str = '\rTraining [{}] {}/{}, {}  {:.1f} step/sec'
        sys.stdout.write(stdout_str.format(
            bar_chars, self.completed, self.task_num, inf_str, self.fps))

        sys.stdout.flush()
###############################################################################
#   DB load                                                                   #
###############################################################################
def load_db(db_path, retina_model, FLAGS, cfg_retina, facenet_model, resize_height, resize_width):
    
    # db 폴더가 없는 경우 에러 처리
    if os.path.isdir(db_path) == True:
        
        # db 얼굴 임베딩 파일 불러오기
        file_name = "representations.pkl"
        
        if path.exists(db_path+file_name) and not FLAGS.db_reset:
            
            f = open(db_path+file_name, 'rb')
            representations = pickle.load(f)
            f.close()
            
        else:
            employees = []
            
            # db에 이미지 파일 경로 추가
            for r, d, f in os.walk(db_path): # r=root, d=directories, f = files
                for file in f:
                    if '.jpg' in file:
                        exact_path = r + file
                        employees.append(exact_path)
                        
            # db에 이미지 없으면 에러
            if len(employees) == 0:
                raise ValueError("There is no image in ", db_path," folder!")

            representations = []
            
            pbar = tqdm(range(0,len(employees)), desc='Finding representations')
            
            for index in pbar:
                employee = employees[index]
                
                # 이미지 불러오기
                frame = cv2.imread(employee)
                
                img = np.float32(frame.copy())
                frame_height, frame_width, _ = frame.shape

                outputs = detect_face_landm(img, retina_model, FLAGS, cfg_retina)

                for prior_index in range(len(outputs)):
                    face = detect_face(frame, outputs[prior_index], 
                                       frame_height, frame_width,
                                       resize_height, resize_width)
                    face_embedding = facenet_model.predict(face[np.newaxis, ...])[0]
                
                    representations.append([employee, face_embedding])
                    plt.imshow(face) 
                    plt.show()
            f = open(db_path+'/'+file_name, "wb")
            pickle.dump(representations, f)
            f.close()
        
        # db 얼굴 임베딩 파일 데이터프레임으로 불러오기
        df = pd.DataFrame(representations, columns = ["identity", "representation"])
        
        return df
    else:
        raise ValueError("Passed db_path does not exist!")
        
    return None
###############################################################################
#   Testing                                                                   #
###############################################################################
def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv2.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs

###############################################################################
#   Detect Face                                                               #
###############################################################################
def detect_face_landm(img, retina_model, cfg_retina):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # pad input image to avoid unmatched shape problem
    img, pad_params = pad_input_image(img, max_steps=max(cfg_retina['steps']))
    
    # run model
    outputs = retina_model(img[np.newaxis, ...]).numpy()
    
    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)

    return outputs

def detect_face(img, ann, img_height, img_width, resize_height, resize_width):
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    # 색변환
    cv2.circle(img, (int(ann[4] * img_width),
                        int(ann[5] * img_height)), 1, (255, 255, 0), 2)
    cv2.circle(img, (int(ann[6] * img_width),
                        int(ann[7] * img_height)), 1, (0, 255, 255), 2)
    cv2.circle(img, (int(ann[8] * img_width),
                        int(ann[9] * img_height)), 1, (255, 0, 0), 2)
    cv2.circle(img, (int(ann[10] * img_width),
                        int(ann[11] * img_height)), 1, (0, 100, 255), 2)
    cv2.circle(img, (int(ann[12] * img_width),
                        int(ann[13] * img_height)), 1, (255, 0, 100), 2)

    img = img[y1:y2, x1:x2]

    left_eye_x, left_eye_y = int(ann[4] * img_width), int(ann[5] * img_height)
    right_eye_x, right_eye_y = int(ann[6] * img_width), int(ann[7] * img_height)
    left_eye = left_eye_x, left_eye_y
    right_eye = right_eye_x, right_eye_y

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
        
    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
        
        if direction == -1:
            angle = 90 - angle
        
        # img = (img * 255).astype(np.uint8)
        
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
        img = cv2.resize(img, (resize_width, resize_height))


        
    return img

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
#     print(output.shape)
#     print(output)
    return output  # (10,128)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y
###############################################################################
#   Visulization                                                              #
###############################################################################
def draw_bbox_landm(img, ann, img_height, img_width, name, color):
    """draw bboxes and landmarks"""
    # bbox
    x1, y1, x2, y2 = int(ann[0] * img_width), int(ann[1] * img_height), \
                     int(ann[2] * img_width), int(ann[3] * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # confidence
    # text = "{:.4f}".format(ann[15])
    # cv2.putText(img, name, (int(ann[0] * img_width), int(ann[1] * img_height)),
    #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # landmark
    if ann[14] > 0:
        cv2.circle(img, (int(ann[4] * img_width),
                         int(ann[5] * img_height)), 1, (255, 255, 0), 2)
        cv2.circle(img, (int(ann[6] * img_width),
                         int(ann[7] * img_height)), 1, (0, 255, 255), 2)
        cv2.circle(img, (int(ann[8] * img_width),
                         int(ann[9] * img_height)), 1, (255, 0, 0), 2)
        cv2.circle(img, (int(ann[10] * img_width),
                         int(ann[11] * img_height)), 1, (0, 100, 255), 2)
        cv2.circle(img, (int(ann[12] * img_width),
                         int(ann[13] * img_height)), 1, (255, 0, 100), 2)


def draw_anchor(img, prior, img_height, img_width):
    """draw anchors"""
    x1 = int(prior[0] * img_width - prior[2] * img_width / 2)
    y1 = int(prior[1] * img_height - prior[3] * img_height / 2)
    x2 = int(prior[0] * img_width + prior[2] * img_width / 2)
    y2 = int(prior[1] * img_height + prior[3] * img_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
