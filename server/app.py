import cv2

# from capture import capture_face_img as cap # capture
from capture import face_alignment as face # face alignment
from capture import show_img as imshow # img show
from DB import db # use db
import vectorized_face_image as vect

# img_path = cap.capture_face_img(img_save_path='./captured_img')
# img = cv2.imread(img_path)
face_alignmented_img = face.face_alignment(img_save_path='./captured_img')
imshow.show_img('captured_img', face_alignmented_img)

print(vect.get_vectorized_face_image(face_alignmented_img))