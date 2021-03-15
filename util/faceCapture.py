from mtcnn import MTCNN # mtcnn 불러오기
import numpy as np
import cv2
import os
import requests
import uuid
import datetime

def find_face(img):
    detector = MTCNN()  
    face_values = detector.detect_faces(img)
    return face_values

# 제대로된 한명의 얼굴이 나올경우 잘려진 얼굴사진 반환, 실패한경우 False 반환
def align_face(img):
    value = find_face(img)
    if (len(value) != 1):
        # 얼굴이 1개가 아니다
        return {'success':False, 'faceImg':None}
    value = value[0]
    bounding_box = value['box'] # 얼굴 위치
    # left = bounding_box[0]
    # top = bounding_box[1]
    # right = left + bounding_box[2]
    # bottom = top + bounding_box[3]

    if (bounding_box[2] * bounding_box[3] < 100*100):
        return {'success':False, 'faceImg':None}

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
    if (len(align_value) != 1):
        # 얼굴이 1개가 아니다
        return {'success':False, 'faceImg':None}

    b_box = align_value[0]['box']
    if (b_box[2] * b_box[3] < 100*100):
        return {'success':False, 'faceImg':None}
    dst = rotated[b_box[1]:b_box[1] + b_box[3], b_box[0]:b_box[0]+b_box[2]]
    return {'success':True, 'faceImg':dst}