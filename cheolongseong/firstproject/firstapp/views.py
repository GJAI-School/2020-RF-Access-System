from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import JsonResponse
import numpy as np
import cv2
import os
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
from .modules.models import RetinaFaceModel
from .facenet import InceptionResNetV2
from .modules.utils import *
from .models import Embedding, AccessHistory

FACE_MODE = True
VEIN_MODE = True

FACE_MODEL_PATH = './firstapp/checkpoints/facenet_weights.h5'
FACE_THRESHOLD = 0.8
VEIN_MODEL_PATH = './firstapp/checkpoints/realfinal_150_0.0149.h5'
VEIN_THRESHOLD = 1.1

# gpu 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_memory_growth()

# retina load
cfg_retina = load_yaml('./firstapp/configs/retinaface_mbv2.yaml')
retina_model = RetinaFaceModel(cfg_retina, training=False, iou_th=0.4, score_th=0.5)
checkpoint_dir = './firstapp/checkpoints/' + cfg_retina['sub_name']
checkpoint = tf.train.Checkpoint(model=retina_model)
if tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("[*] load ckpt from {}.".format(
        tf.train.latest_checkpoint(checkpoint_dir)))
else:
    print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
    exit()
# facenet load
facenet_model = InceptionResNetV2()
facenet_model.load_weights(FACE_MODEL_PATH)
# tweetynet load
vein_model = load_model(VEIN_MODEL_PATH, custom_objects={'tf':tf}, compile=False)

def home(request):
    history = AccessHistory.objects.all()
    # history.delete()
    return render(request, 'home.html', {'history':history})

def json_load(request):
    last = AccessHistory.objects.last()
    send_data = {
        "no": last.id, 
        "date":last.date, 
        "rfid": last.rfid.rfid, 
        "face_check": last.face_check, 
        'temp': last.temp, 
        'vein_check': last.vein_check
    }
    return JsonResponse(send_data)

# Create your views here.
def accesshistory_save(request):
    img, serialnumber, temp, redimg = request.body.split(b':::::')

    img = np.frombuffer(img, dtype = 'uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    serialnumber = np.frombuffer(serialnumber, dtype='uint8')-48
    serialnumber = ''.join(map(str, serialnumber)).replace('240',' ')

    temp = np.frombuffer(temp, dtype='uint8')-48
    temp = ''.join(map(str, temp)).replace('254','.')
    
    redimg = np.frombuffer(redimg, dtype = 'uint8')
    redimg = cv2.imdecode(redimg, cv2.IMREAD_COLOR)

    # print(img.shape)
    # print(serialnumber)
    # print(temp)
    # print(redimg.shape)
    
    face_check = "None"
    vein_check = "None"

    try:
        # db에 embedding 값 가져오기
        target_object = Embedding.objects.get(rfid=serialnumber)
    except:
        target_object = 'Unknown'
    
    if target_object != 'Unknown':
        with open(target_object.face_embedding, 'rb') as f:
            target_face_embeddings = pickle.load(f)
        with open(target_object.vein_embedding, 'rb') as f:
            target_vein_embedding = pickle.load(f)
        
        if FACE_MODE:
        # 얼굴인식
            img_copy = np.float32(img.copy())
            frame_height, frame_width, _ = img.shape
            resize_height, resize_width = facenet_model.layers[0].input_shape[0][1:3]
            
            outputs = detect_face_landm(img_copy, retina_model, cfg_retina)

            try:
                outputs = outputs[0]
            except:
                print('얼굴 집어 넣어라!')
                outputs = None
                face_check = "X"
            
            if outputs is not None:
                face = detect_face(img, outputs, 
                                    frame_height, frame_width,
                                    resize_height, resize_width)

                face = prewhiten(face)

                face_embedding = facenet_model.predict(face[np.newaxis, ...])[0]

                distances = []
                for target_face_embedding in target_face_embeddings:
                    distance = findEuclideanDistance(l2_normalize(face_embedding), l2_normalize(target_face_embedding))
                    distances.append(distance)
                
                mean = np.mean(distances)            # 10개 사진과 들어온 사진의 평균 거리
                # max = np.max(distances)              # 10개 사진과 들어온 사진의 최대 거리
                # distances = (mean+max)/2 
                distances = mean
                print("얼굴거리 : ", distances)
                if FACE_THRESHOLD > distances:
                    face_check = "O"
                else:
                    face_check = "X"

        if VEIN_MODE:
            # 정맥인식
            img_height, img_width, _ = vein_model.layers[0].input_shape[1:]
            redimg = cv2.cvtColor(redimg, cv2.COLOR_BGR2GRAY)
            redimg = redimg[:,150:550]
            redimg = cv2.adaptiveThreshold(redimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 2)
            redimg = cv2.resize(redimg, (img_width, img_height))
            redimg = redimg / 255
            redimg = redimg[..., np.newaxis]
            vein_embedding = vein_model.predict(redimg[np.newaxis, ...])[0]
            distance = findEuclideanDistance(vein_embedding, target_vein_embedding)
            print("정맥거리 : ", distance)
            if VEIN_THRESHOLD > distance:
                vein_check = "O"
            else:
                vein_check = "X"
        
        if face_check == "X" or vein_check == "X":
            print("인증실패")
        else:
            print("인증성공")

        AccessHistory.objects.create(
            rfid = target_object,
            face_check = face_check,
            temp = float(temp),
            vein_check = vein_check,
        )

    else:
        print("등록되지 않은 사용자입니다.")

    return render(request, 'home.html')
    
def dbreset(request):
    # example
    # http://192.168.137.227:8000/dbreset?name=seung&rfid=182 224 172 43
    # http://192.168.137.227:8000/dbreset?name=Hwan&rfid=233 72 52 193
    name = request.GET.get('name')
    rfid = request.GET.get('rfid')

    # 얼굴 임베딩
    resize_height, resize_width = facenet_model.layers[0].input_shape[0][1:3]
    dir_path = f'./firstapp/db/{name}/face_img'
    face_img_paths = os.listdir(dir_path)
    face_embeddings = []
    for face_img_path in face_img_paths:
        target_img_path = dir_path + '/' + face_img_path
        target_img = cv2.imread(target_img_path)
        img = np.float32(target_img.copy())
        frame_height, frame_width, _ = target_img.shape

        outputs = detect_face_landm(img, retina_model, cfg_retina)
        face = detect_face(target_img, outputs[0], 
                            frame_height, frame_width,
                            resize_height, resize_width)
        face = prewhiten(face)
        face_embedding = facenet_model.predict(face[np.newaxis, ...])[0]
        face_embeddings.append(face_embedding)
    face_embeddings = np.array(face_embeddings)
    face_pickle_path = f'./firstapp/db/face_pickle/{name}_face.pickle'
    with open(face_pickle_path, 'wb') as f:
        pickle.dump(face_embeddings, f)
    # 정맥 임베딩
    model = load_model(VEIN_MODEL_PATH, custom_objects={'tf':tf}, compile=False)
    img_height, img_width, _ = model.layers[0].input_shape[1:]
    dir_path = f'./firstapp/db/{name}/vein_img'
    vein_img_path = os.listdir(dir_path)[0]
    target_img_path = dir_path + '/' + vein_img_path
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = target_img[:,150:550]
    target_img = cv2.adaptiveThreshold(target_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 2)
    target_img = cv2.resize(target_img, (img_width, img_height))
    target_img = target_img / 255
    target_img = target_img[..., np.newaxis]
    vein_embedding = model.predict(target_img[np.newaxis, ...])[0]
    vein_pickle_path = f'./firstapp/db/vein_pickle/{name}_vein.pickle'
    with open(vein_pickle_path, 'wb') as f:
        pickle.dump(vein_embedding, f)

    Embedding.objects.create(
        name = name,
        rfid = rfid,
        face_embedding = face_pickle_path,
        vein_embedding = vein_pickle_path,
    )
    return render(request, 'upload.html')

def upload(request):
    return render(request, 'upload.html')

def dbsave(request):
    print(request.POST.get('face'))
    print(request.POST.get('vein'))
    return render(request, 'upload.html')