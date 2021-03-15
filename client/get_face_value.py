from absl import logging
import cv2
import os
import numpy as np
import tensorflow as tf
import time
import sys

from modules.models import RetinaFaceModel
from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

set_memory_growth()
# tf.debugging.set_log_device_placement(True)

cfg_path = './configs/retinaface_mbv2.yaml'
gpu = '0'
iou_th = 0.4
score_th = 0.5

cfg = load_yaml(cfg_path)

model = RetinaFaceModel(cfg, training=False, iou_th=iou_th,
                            score_th=score_th)

checkpoint_dir = './checkpoints/' + cfg['sub_name']
checkpoint = tf.train.Checkpoint(model=model)
if tf.train.latest_checkpoint(checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("[*] load ckpt from {}.".format(
        tf.train.latest_checkpoint(checkpoint_dir)))
else:
    print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
    exit()

# 부모모듈 경로 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util import FaceValue as FV

def get_face_value(img_raw, down_scale_factor=0.3):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    

    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    # 빠르게 얼굴을 찾기위해 이미지 크기를 줄여서 탐색
    if down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=down_scale_factor,
                            fy=down_scale_factor,
                            interpolation=cv2.INTER_LINEAR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    outputs = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)
    
    #output된 얼굴들에 대한 정보가 들어있는 배열
    fvalues = map(lambda output: FV.FaceValue(output, img_width_raw, img_height_raw), outputs)

    result = list(fvalues)
    print('get face value', result)
    return result
