from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from modules.models import RetinaFaceModel
from modules.utils import *
from facenet import InceptionResNetV2


flags.DEFINE_string('cfg_retina_path', './configs/retinaface_mbv2.yaml',
                    'retina config file path')
flags.DEFINE_boolean('db_reset', False, 'db reset')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_float('iou_th', 0.4, 'iou threshold for nms')
flags.DEFINE_float('score_th', 0.5, 'score threshold for nms')
flags.DEFINE_float('down_scale_factor', 1.0, 'down-scale factor for inputs')


def main(_argv):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg_retina = load_yaml(FLAGS.cfg_retina_path)

    # define retina
    retina_model = RetinaFaceModel(cfg_retina, training=False, iou_th=FLAGS.iou_th,
                            score_th=FLAGS.score_th)

    # load checkpoint retina
    checkpoint_dir = './checkpoints/' + cfg_retina['sub_name']
    checkpoint = tf.train.Checkpoint(model=retina_model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    # define facenet + load
    # facenet_model_path = './checkpoints/facenet_keras.h5'
    # facenet_model = load_model(facenet_model_path)
    facenet_model = InceptionResNetV2()
    facenet_model.load_weights('./checkpoints/facenet_weights.h5')
    resize_height, resize_width = facenet_model.layers[0].input_shape[0][1:3]

    # DB load
    db = load_db('./db/', retina_model, FLAGS, cfg_retina, facenet_model, resize_height, resize_width)

    cam = cv2.VideoCapture(0)

    start_time = time.time()
    while True:
        _, frame = cam.read()
        if frame is None:
            print("no cam input")

        img = np.float32(frame.copy())
        frame_height, frame_width, _ = frame.shape

        outputs = detect_face_landm(img, retina_model, FLAGS, cfg_retina)

        # draw results
        for prior_index in range(len(outputs)):
            db_sample = db.copy()
            try:
                face = detect_face(frame, outputs[prior_index], 
                                    frame_height, frame_width,
                                    resize_height, resize_width)

                target_embedding = facenet_model.predict(face[np.newaxis, ...])[0]

                distances = []

                for index, instance in db_sample.iterrows():
                    db_embedding = instance["representation"]
                    
                    # 코사인 거리를 사용하여 현재 이미지와 db 이미지 비교
                    distance = findEuclideanDistance(l2_normalize(db_embedding), l2_normalize(target_embedding))
                    distances.append(distance)
                
                print(distances) #===============================================================================================
                threshold = 0.35
                db_sample["distance"] = distances
                db_sample = db_sample.drop(columns = ["representation"])
                db_sample = db_sample[db_sample.distance <= threshold]
                db_sample = db_sample.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

                name = "Unknown"
                color = (0,0,255)
                if db_sample.size:
                    name = db_sample['identity'][0]
                    color = (0,255,0)

                draw_bbox_landm(frame, outputs[prior_index], frame_height, frame_width, name, color)
            except:
                pass

        # calculate fps
        fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
        start_time = time.time()
        cv2.putText(frame, fps_str, (25, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

        # show frame
        cv2.imshow('frame', frame)
        #cv2.imshow('frame', face)
        if cv2.waitKey(1) == ord('q'):
            exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
