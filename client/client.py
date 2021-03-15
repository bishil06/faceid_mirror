# 사진 촬영
# mtcnn
# face alignment

# from mtcnn import MTCNN # mtcnn 불러오기
import numpy as np
import cv2
import os
import requests
import uuid
import time 
import sys
import pickle
import time

from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

# 부모모듈 경로 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#Roll, Yaw, Ptich calc
def find_roll(pts):
    return pts[6] - pts[5]

def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    return le2n - re2n

def find_pitch(pts):
    eye_y = (pts[5] + pts[6]) / 2
    mou_y = (pts[8] + pts[9]) / 2
    e2n = eye_y - pts[7]
    n2m = pts[7] - mou_y
    return e2n/n2m
#####################

# from util import faceCapture as f_cap
import get_face_value as get_fv

myurl = 'http://61.82.106.114:5000/fileUpload'

content_type = 'image/jpeg'
headers = {'content-type': content_type}

myname = 'lhj' # 클라이언트 실행전에 미리 설정해주세요 이 이름으로 사진이 라벨링되어 저장됩니다.

def capture_face_img():
    gauid = cv2.imread('g.png')

    capture = cv2.VideoCapture(0) # 카메라 불러오기
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start_time = time.time()
    timer = time.time()
    timer_flag = 0
    name_count = 0
    while True:
        ret, frame = capture.read() # 카메리의 입력을 읽어와서
        frame = cv2.flip(frame, 1) ##출력영상 좌우반전
        # cv2.imshow("Frame", frame) # 화면에 출력합니다
        
        key = cv2.waitKey(33)
        
        outputs = get_fv.get_face_value(frame)
        print(outputs)

        frame_height, frame_width, _ = frame.shape
        gauid = cv2.resize(gauid, dsize=(frame_width, frame_height), interpolation=cv2.INTER_AREA)

        # landmark save
        Nfaces = len(outputs)
        lms = np.zeros((Nfaces,10))  
        lms[:,0:5] = outputs[:,[4,6,8,10,12]]*frame_width
        lms[:,5:10] = outputs[:,[5,7,9,11,13]]*frame_height
        #print(lms)

        if len(outputs) == 0:
                cv2.putText(frame, "No Person in Frame", (200, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 1)
            # 1명이상 탐지됨
        elif len(outputs) >= 1:
            box_arr = []
            for prior_index in range(len(outputs)):
                x1, y1, x2, y2 = int(outputs[prior_index][0] * frame_width), int(outputs[prior_index][1] * frame_height), \
                int(outputs[prior_index][2] * frame_width), int(outputs[prior_index][3] * frame_height)
                box_arr.append((x2-x1)*(y2-y1))                
            max_size = max(box_arr)
            max_idx = box_arr.index(max_size)
            max_x1, max_y1, max_x2, max_y2 = int(outputs[max_idx][0] * frame_width), int(outputs[max_idx][1] * frame_height), \
            int(outputs[max_idx][2] * frame_width), int(outputs[max_idx][3] * frame_height)
    
            #얼굴박스의 중심값 좌표
            center_x = int(outputs[max_idx][0] * frame_width) + ((int(outputs[max_idx][2] * frame_width) - int(outputs[max_idx][0] * frame_width)) / 2)
            center_y = int(outputs[max_idx][1] * frame_height) + ((int(outputs[max_idx][3] * frame_height) - int(outputs[max_idx][1] * frame_height)) / 2)
            is_center_x = center_x < (int(frame_width / 2) + 40) and center_x > (int(frame_width / 2) - 40)
            is_center_y = center_y < (int(frame_height / 2) + 20) and center_y > (int(frame_height / 2) - 60)
            cv2.circle(frame, (int(frame_width / 2), int(frame_height / 2)), 1, (255, 255, 0), 1)
            #좌표 확인
            cv2.putText(frame, "bbox_center"+"("+str(center_x)+","+str(center_y)+")", (400,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, "bbox_center"+"("+str(is_center_x)+","+str(is_center_y)+")", (400,60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)    
            #얼굴크기 및 가이드영역(프레임 가운데)으로 얼굴 맞추기
            if max_size >= 150*150 and is_center_x and is_center_y:
                draw_bbox_landm(frame, outputs[box_arr.index(max_size)], frame_height, frame_width)
                cv2.putText(frame, "("+str(x2-x1)+","+str(y2-y1)+")" , (25,140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)

                # 3축 확인 필터
                if (find_roll(lms[max_idx]) >= -10 and \
                    find_roll(lms[max_idx]) <= 10) and \
                (find_yaw(lms[max_idx]) >= -15 and \
                    find_yaw(lms[max_idx]) <= 15) and \
                (find_pitch(lms[max_idx]) >= -2 and \
                    find_pitch(lms[max_idx]) <= 2):
                    cv2.putText(frame, "Angle Filter Pass", (200, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
                    #타이머 작동
                    if time.time() - timer < 0:
                        pass
                    elif time.time() - timer < 3:
                        cv2.putText(frame, str(3 - int(time.time() - timer))+'sec ', (500, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0),1)
                    if (time.time() - timer >= 3): # 얼굴 검출 상태에서 3초간 대기                            
                        cv2.imwrite('{}.jpg'.format(name_count), frame) # 캡쳐
                        name_count += 1
                        timer = time.time()
                        timer_flag = 1
                else:
                    cv2.putText(frame, "Angle Filter NO Pass", (200, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 1)
                    timer = time.time()
            elif max_size < 150*150 and is_center_x and is_center_y:
                    cv2.putText(frame, "Please come closer", (200, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 1)
                    timer = time.time()
            else:
                cv2.putText(frame, "Please come to the center", (200, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255), 1)
                timer = time.time()
            
        # calculate fps
        fps_str = "FPS: %.2f" % (1 / (time.time() - start_time))
        start_time = time.time()
        cv2.putText(frame, fps_str, (25, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
        cv2.imshow('frame', frame&gauid)
                    
                    

        # if key == 99: # c 키 의 ascii 코드가 99
        #     img_path = 'captured/{}.png'.format(uuid.uuid4())
        #     start = time.time()
        #     f_value = get_fv.get_face_value(frame)
        #     print('얼굴사진 추출', time.time()-start)
        #     print(f_value)
            # if(f_value['success']):
            #     print(f_value['face_value'])
            #     # x1, y1, x2, y2 = f_value['face_value'][:4]
            #     # start = time.time()
            #     # response = requests.post(myurl, data=pickle.dumps({'label': myname, 'img': frame[y1:y2, x1:x2]}), headers=headers)
            #     # print('face img 전송 시간', time.time()-start)
            #     # print(response)
            # # img_rgb[top:bottom, left: right]
            # # alignment = f_cap.align_face(frame)
            # # print(alignment['success'])
            # # if (alignment['success']):
            # #     # cv2.imwrite(img_path, alignment['faceImg'])
            # #     _, img_encoded = cv2.imencode('.jpg', alignment['faceImg'])
            # #     print('face img 전송 시작', datetime.datetime.now())

            # #     # files = {'file': open(img_path, 'rb')}
            # #     response = requests.post(myurl, data=pickle.dumps({'label': myname, 'img': img_encoded.tostring()}), headers=headers)
            # #     print(response)
            # else:
            #     continue
        if key == 27:
            capture.release()
            cv2.destroyAllWindows()
            break
    return frame

capture_face_img()