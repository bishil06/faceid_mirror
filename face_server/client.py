import requests
import cv2
import codecs
import pickle

server_address = 'localhost:5001'
myurl = 'http://{}/photoUpload'.format(server_address)
print(myurl)

response = requests.post(myurl, data=pickle.dumps(cv2.imread('data/iu.jpg')))
# 얼굴 사진 받아오기
face = response.json()['face']
face = pickle.loads(codecs.decode(face.encode(), "base64"))
print(face)
# 잘 잘렸는지 결과 확인
print(response.json()['success'])