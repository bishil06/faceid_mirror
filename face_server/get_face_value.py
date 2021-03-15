from absl import logging
import cv2
import os
import numpy as np
import tensorflow as tf
import time

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

def get_face_value(img_raw, down_scale_factor=0.3):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    

    img_height_raw, img_width_raw, _ = img_raw.shape
    img = np.float32(img_raw.copy())

    if down_scale_factor < 1.0:
        img = cv2.resize(img, (0, 0), fx=down_scale_factor,
                            fy=down_scale_factor,
                            interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img, pad_params = pad_input_image(img, max_steps=max(cfg['steps']))

    outputs = model(img[np.newaxis, ...]).numpy()

    # recover padding effect
    outputs = recover_pad_output(outputs, pad_params)
    
    def output_to_fvalue(out, w, h):
        result = {}

        left = int(out[0]*w)
        top = int(out[1]*h)
        right = int(out[2]*w)
        bottom = int(out[3]*h)
        result['bbox'] = [left, top, right, bottom]
        result['bbox_size'] = (result['bbox'][3] - result['bbox'][1]) * (result['bbox'][2] - result['bbox'][0])
        
        result['landm'] = {}
        result['landm']['left_eye'] = (int(out[4] * w), int(out[5] * h))
        result['landm']['right_eye'] = (int(out[6] * w), int(out[7] * h))
        result['landm']['nose'] = (int(out[8] * w), int(out[9] * h))
        result['landm']['mouse_left'] = (int(out[10] * w), int(out[11] * h))
        result['landm']['mouse_right'] = (int(out[12] * w), int(out[13] * h))
        return result
    fvalues = map(lambda output: output_to_fvalue(output, img_width_raw, img_height_raw), outputs)
    return list(fvalues)


# with tf.device('/GPU:0'):
# start = time.time()
# r = get_face_value(cv2.imread('gogo.jpg'))
# print(r)
# print("time :", time.time() - start)
