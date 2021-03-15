import cv2

import generate_key as gk

import encrypt as enc
import decrypt as dec

import time

img = cv2.imread('pic3.png')
keyList = gk.gen_key(img, 1)

start = time.time()
enc_img = enc.encrypt(img, keyList)
print("enc time :", time.time() - start)
cv2.imwrite('enc_img.png', enc_img)

start = time.time()
dec_img = dec.decrypt(enc_img, keyList)
print("dec time :", time.time() - start)
cv2.imwrite('dec_img.png', dec_img)
