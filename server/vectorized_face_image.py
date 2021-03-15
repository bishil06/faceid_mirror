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
    # img = cv2.resize(img, (cfg['input_size'], cfg['input_size'])) # 이미지 크기 조정
    img = img.astype(np.float32) / 255. # 0. ~ 255. 사이 값 변환 
    if len(img.shape) == 3: # 차원 조절
        img = np.expand_dims(img, 0)
    embeds = l2_norm(model(img)) # l2 normalization?
    return embeds
