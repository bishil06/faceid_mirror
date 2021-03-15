from mtcnn import MTCNN # mtcnn 불러오기
import numpy as np
from absl import logging
import cv2
import os
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# 대충 모델관련된것들 가져오는중
from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm

cfg_yaml_path = "./configs/arc_res50.yaml"
cfg = load_yaml(cfg_yaml_path)

model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         training=False)

ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
if ckpt_path is not None:
    print("[*] load ckpt from {}".format(ckpt_path))
    model.load_weights(ckpt_path)
else:
    print("[*] Cannot find ckpt from {}.".format(ckpt_path))
    exit()

def get_vect_face_img(align_face_img): # 벡터화된 이미지 값을 직접 넣도록 했습니다.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 기본값 0
    
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    img = align_face_img
#     img = cv2.imread(img_path) # opencv 로 이미지 읽어옴
    img = cv2.resize(img, (cfg['input_size'], cfg['input_size'])) # 이미지 크기 조정
    img = img.astype(np.float32) / 255. # 0. ~ 255. 사이 값 변환 
    if len(img.shape) == 3: # 차원 조절
        img = np.expand_dims(img, 0)
    embeds = l2_norm(model(img)) # l2 normalization?
    return embeds


def find_face(img):
    detector = MTCNN()  
    face_values = detector.detect_faces(img)
    return face_values

def align_face(img):
    value = find_face(img)
    if (len(value) != 1):
        # 얼굴이 1개가 아니다
        return False
    value = value[0]
    bounding_box = value['box'] # 얼굴 위치
    # left = bounding_box[0]
    # top = bounding_box[1]
    # right = left + bounding_box[2]
    # bottom = top + bounding_box[3]

    keypoints = value['keypoints'] # 눈코입 위치

    left_eye_x = keypoints['left_eye'][0]
    left_eye_y = keypoints['left_eye'][1]
    right_eye_x = keypoints['right_eye'][0]
    right_eye_y = keypoints['right_eye'][1]

    delta_x = right_eye_x - left_eye_x # 가로
    delta_y = right_eye_y - left_eye_y # 세로
    angle = np.arctan(delta_y / delta_x) 
    angle = (angle * 180) / np.pi # 각도 구하기
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2) # 이미지 중심 좌표

    M = cv2.getRotationMatrix2D(center, (angle), 1) # 중심좌표 기준으로 각도 만큼 회전  
    rotated = cv2.warpAffine(img, M, (width,height)) # 이미지를 전달해준 변환행렬처럼 위치 변경
    
    align_value = find_face(rotated)

    b_box = align_value[0]['box']
    dst = rotated[b_box[1]:b_box[1] + b_box[3], b_box[0]:b_box[0]+b_box[2]]
    return dst

def read_align_vectorized(img_path):
    img = cv2.imread(img_path)
    align_img = align_face(img)
    return get_vect_face_img(align_img)