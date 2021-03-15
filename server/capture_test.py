import time

import capture as cap # capture.py 에서 함수를 불러온다
import cv2

img = cv2.imread('data/BruceLee.jpg')

start = time.time()
align_img = cap.align_face(img)
align_img = cap.align_face(img)
align_img = cap.align_face(img)
align_img = cap.align_face(img)
print('5번의 face alignment 수행 시간 ', time.time()-start)
# cv2.imshow('test', align_img)
# cv2.waitKey(0)
# cap.read_align_vectorized('data/BruceLee.jpg')
# cap.read_align_vectorized('data/BruceLee.jpg')
# cap.read_align_vectorized('data/BruceLee.jpg')
# cap.read_align_vectorized('data/BruceLee.jpg')
# cap.read_align_vectorized('data/BruceLee.jpg')

start = time.time()
cap.get_vect_face_img(align_img)
cap.get_vect_face_img(align_img)
cap.get_vect_face_img(align_img)
cap.get_vect_face_img(align_img)
cap.get_vect_face_img(align_img)
print('5번의 face vectorized 수행 시간 ', time.time()-start)