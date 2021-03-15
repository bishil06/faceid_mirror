import numpy as np
import cv2
import os
import requests
import uuid
import time 
import sys
import pickle
import time
import math
from playsound import playsound
import threading

from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

import pyttsx3
# TTS 엔진 초기화
engine = pyttsx3.init()

# 말하는 속도
engine.setProperty('rate', 190)
rate = engine.getProperty('rate')

# 소리 크기
engine.setProperty('volume', 1) # 0~1 
volume = engine.getProperty('volume')

# 목소리
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# 부모모듈 경로 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util import face_resize as fr

import isMasked as isM

# from util import faceCapture as f_cap
import get_face_value as get_fv

server_ip = '127.0.0.1'
myurl = 'http://{}:50000/certification'.format(server_ip)

content_type = 'image/jpeg'
headers = {'content-type': content_type}

kind = '퇴근'
LABEL = '2번 게이트' # 클라이언트 실행전에 미리 설정해주세요 이 이름으로 사진이 라벨링되어 저장됩니다.
w = 640
h = 480
center = (w//2, h//2)

response = None
res_time = None
emp_name = None

def get_w_from_bbox(bbox):
    return bbox[3] - bbox[1]
def get_h_from_bbox(bbox):
    return bbox[2] - bbox[0]
def get_center_from_bbox(bbox):
    return (bbox[0] + (get_h_from_bbox(bbox))//2, bbox[1] + (get_w_from_bbox(bbox))//2)
def get_size_from_bbox(bbox):
    return get_w_from_bbox(bbox) * get_h_from_bbox(bbox)

def find_roll(landm):
    return landm['left_eye'][1] - landm['right_eye'][1] # 2번 눈

def find_yaw(landm):
    le2n = landm['nose'][0] - landm['left_eye'][0]
    re2n = landm['right_eye'][0] - landm['nose'][0]
    return le2n - re2n

def find_pitch(landm):
    eye_y = (landm['left_eye'][1] + landm['right_eye'][1]) / 2
    mou_y = (landm['mouse_left'][1] + landm['mouse_right'][1]) / 2
    e2n = eye_y - landm['nose'][1]
    n2m = landm['nose'][1] - mou_y
    return e2n/n2m

 


def capture_face_img():
    global res_time, response
    # gauid = cv2.imread('g.png')
    wait = cv2.imread('wait.jpg')
    wait = cv2.resize(wait, dsize=(w, h), interpolation=cv2.INTER_AREA)
    auth_false = cv2.imread('auth_false.png')
    auth_false = cv2.resize(auth_false, dsize=(w, h), interpolation=cv2.INTER_AREA)
    auth_true = cv2.imread('auth_true.jpg')
    auth_true = cv2.resize(auth_true, dsize=(w, h), interpolation=cv2.INTER_AREA)
    shouldMask = cv2.imread('shouldMask.jpg')
    shouldMask = cv2.resize(shouldMask, dsize=(w, h), interpolation=cv2.INTER_AREA)

    capture = cv2.VideoCapture(0) # 카메라 불러오기
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    one_face = False
    roll = False
    yaw = False
    pitch = False
    close_center = False
    check_mask = False

    timer = None
    sthread = None
    isSending = False
    isShowResult = None # None, True, False
    needCheckMask = None # None, True, False
    detectMask = None
    #isMasked = None
   
    r_timer = None
    m_timer = None
    mask_timer = None
    
    
    while True:
        ret, frame = capture.read() # 카메리의 입력을 읽어와서
        frame = cv2.flip(frame, 1) ##출력영상 좌우반전
        frame_raw = frame.copy()
        
        cv2.ellipse(frame, center, (w//4, h//4), 90, 0, 360, (255, 255, 0), 2)
        #cv2.rectangle(frame, (w//2 - h//3, h//2 - w//3), (w//2 + h//3, h//2 + w//3), (0, 255, 0), 2)
        
        # 촬영중
        if ret & (isSending == False) & (isShowResult != True) & (needCheckMask != True) & (detectMask == None):
            print('촬영모드')
            
            class SendingThread(threading.Thread):
                def __init__(self):
                    threading.Thread.__init__(self, name='sending data', args=(frame_raw, fvalues, LABEL))
            
                # CounterThread가 실행하는 함수
                def run(self):
                    global response, myurl, headers, emp_name

                    fvalue = fvalues[0]
                    landm = fvalue['landm']

                    delta_x = landm['right_eye'][0] - landm['left_eye'][0] # 가로
                    delta_y = landm['right_eye'][1] - landm['left_eye'][1] # 세로
                    angle = np.arctan(delta_y / delta_x) 
                    angle = (angle * 180) / np.pi # 각도 구하기
                    # print(angle)
                    h, w, _ = frame_raw.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, (angle), 1)
                    rotated = cv2.warpAffine(send_frame.copy(), M, (w,h))

                    rotated_fvs = get_fv.get_face_value(send_frame)
                    #cv2.imwrite('./before_send.jpg', send_frame)
                    response = requests.post(myurl, data=pickle.dumps({'kind': kind, 'label': LABEL, 'img': send_frame, 'angle':angle, 'fvalue':rotated_fvs}), headers=headers)
                    emp_name = response.json()['emp_name']
                    response = response.json()['cert']
                    
                    print(response)
                    # response = True # 기능 테스트용 코드입니다.
                    # response = response['succsss']
                    time.sleep(1) 
                    
            fvalues = get_fv.get_face_value(frame_raw)
            # 얼굴 감지 모드
            if (len(fvalues) == 1):
                bbox = fvalues[0]['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                one_face = True
                rollv = find_roll(fvalues[0]['landm'])
                if abs(rollv ) < 7:
                    roll = True
                else:
                    roll = False
                pitchv = find_pitch(fvalues[0]['landm'])
                if 1 < abs(pitchv) < 1.5:
                    pitch = True
                else:
                    pitch = False
                yawv = find_yaw(fvalues[0]['landm'])
                if abs(yawv) < 10:
                    yaw = True
                else:
                    yaw = False                 
                
                reszie_result, face = fr.face_resize(frame_raw, bbox, 112)
                if reszie_result == True:
                    check_maskv = isM.isMasked(face)
                    if check_maskv == False:
                        check_mask = True
                    else:
                        check_mask = False
                             
                # print(rollv, yawv, pitchv)
                cv2.putText(frame, '{}, {}, {}'.format(rollv, yawv, pitchv), (400,40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 0), 1)
                bbox_center = get_center_from_bbox(fvalues[0]['bbox'])
                a = center[0] - bbox_center[0]
                b = center[1] - bbox_center[1]
                center_len = math.sqrt((a * a) + (b * b))
                if center_len < 100:
                    close_center = True
                else:
                    close_center = False
                # print(center_len)
            else:
                one_face = False
            # 필터 체크
            if one_face & roll & yaw & pitch & close_center & check_mask:
                if timer == None:
                    timer = time.time()
                elif time.time() - timer < 1:
                    send_frame = frame_raw.copy()
                    sthread = SendingThread()
                    sthread.start()
                    #cv2.imwrite("test_123.jpg", frame)
                    #cv2.imwrite("test_1233.jpg", frame_raw)
                    isSending = True
                    timer = None
            else:
                timer = time.time()
        # 인증정보 전송중
        elif isSending:
            print('전송모드')
            if not sthread.is_alive():
                # 스레드 종료됨 - 1초간 통과 인지 실패인지 화면 표시
                if res_time == None:
                    res_time = time.time()
                else:
                    if time.time() - res_time < 1:
                        frame = wait
                        #playsound("./wait.mp3")
                    else:
                        res_time = None
                        isSending = False
                        if response:
                            isShowResult = True
                        else:
                            playsound("./Auth_fail.mp3")
                            isShowResult = False
                
            # elif res_time == None:
        # 인증결과에 따른 액션
        elif isShowResult != None:
            print('결과출력모드')
            if r_timer == None:
                r_timer = time.time()
            else:
                if time.time() - r_timer < 1:
                    if isShowResult == True:
                        frame = auth_true
                        # 말하기
                        print(emp_name)
                        engine.say('"{}".님 인증되었습니다'.format(emp_name)) 
                        engine.runAndWait() # 말 다할때까지 대기
                        engine.stop() # 끝
                        if not needCheckMask == False:
                            needCheckMask = True    
                    elif isShowResult == False:
                        frame = auth_false
                        playsound("./Auth_fail.mp3")
                else:
                    r_timer = None
                    isShowResult = None
                    response = None
        # 마스크 의무화 됬다는 것을 2초간 보여주고 마스크 썻는지 검사를 한다
        elif needCheckMask == True:
            print('마스크 안내모드')
            if m_timer == None:
                m_timer = time.time()
            else:
                if time.time() - m_timer < 1:
                    #frame = shouldMask
                    playsound("./Mask.mp3" )
                else:                   
                    needCheckMask = None
                    m_timer = None
                    detectMask = True
        # 마스크 감지
        elif detectMask != None:
            #print('마스크 감지모드')
            if mask_timer == None:
                mask_timer = time.time()
            else:
                if time.time() - mask_timer < 20:
                    fvs = get_fv.get_face_value(frame_raw)
                    if len(fvs) == 0:
                        pass
                    elif len(fvs) == 1:
                        reszie_result, faceImg = fr.face_resize(frame_raw, fvs[0]['bbox'], 112)
                        if reszie_result == True:
                            if isM.isMasked(faceImg) == True:
                                playsound("./Gate.mp3" )
                                mask_timer = None
                                detectMask = None
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                else:
                    mask_timer = None
                    detectMask = None

        
        cv2.imshow('frame', frame)
            
        key = cv2.waitKey(33)
        if key == 27:
            capture.release()
            cv2.destroyAllWindows()
            break
    return frame

capture_face_img()