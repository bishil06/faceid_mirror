import cv2

from mtcnn import MTCNN # mtcnn 불러오기
import numpy as np

import uuid

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from capture import capture_face_img as cap

def face_alignment(img_save_path=''):
    img_name = '{}/{}.jpg'.format(img_save_path, uuid.uuid4()) # 사진이 저장될 경로와 이름 지정
    detector = MTCNN()
    while (True):
        img = cap.capture_face_img()
        value = detector.detect_faces(img)
        print(value)
        if (len(value) == 1):
            break
        print('얼굴이 한명이 아닙니다. 다시 찍어 주세요.')

    bounding_box = value[0]['box']
    dst = img.copy()
    dst = img[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    
    point = value[0]['keypoints']
    left_eye_x = point['left_eye'][0]
    left_eye_y = point['left_eye'][1]
#     print(left_eye_x, left_eye_y)

    right_eye_x = point['right_eye'][0]
    right_eye_y = point['right_eye'][1]
#     print(right_eye_x, right_eye_y)

    dst2 = img.copy() # 원본 이미지 복사

    # 눈에 원 그리기
#     dst2 = cv2.circle(dst2, point['left_eye'], 10, (0,255,0))
#     dst2 = cv2.circle(dst2, point['right_eye'], 10, (0,255,0))
#     plt.imshow(dst2)
    
    delta_x = right_eye_x - left_eye_x # 가로
    delta_y = right_eye_y - left_eye_y # 세로
    angle = np.arctan(delta_y / delta_x) 
    angle = (angle * 180) / np.pi # 각도 구하기
#     print(angle)
    
    height, width = img.shape[:2]
    center = (width // 2, height // 2) # 이미지 중심 좌표

    M = cv2.getRotationMatrix2D(center, (angle), 1) # 중심좌표 기준으로 각도 만큼 회전  
    rotated = cv2.warpAffine(img, M, (width,height)) # 이미지를 전달해준 변환행렬처럼 위치 변경
    
    value = detector.detect_faces(rotated)
    bounding_box = value[0]['box']
    dst = rotated.copy()
    dst = rotated[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    cv2.imwrite(img_name, dst)
    return dst