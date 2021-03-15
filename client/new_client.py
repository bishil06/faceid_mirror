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
import pyttsx3
from functools import reduce

from modules.utils import (set_memory_growth, load_yaml, draw_bbox_landm,
                           pad_input_image, recover_pad_output)

# 부모모듈 경로 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util import face_resize as fr
import isMasked as isM
import get_face_value as get_fv

cert_server_address = 'http://{}:50000'.format('192.168.0.9')

# 클라이언트 실행전에 미리 설정해주세요
kind = '출근' # 출근 or 퇴근
LABEL = '2번 게이트'

camera_w = 640
camera_h = 480
camera_center = (camera_w//2, camera_h//2)

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

# 화면 안내메세지 이미지 로드
wait = cv2.imread('wait.jpg')
wait = cv2.resize(wait, dsize=(camera_w, camera_h), interpolation=cv2.INTER_AREA)
auth_false = cv2.imread('auth_false.png')
auth_false = cv2.resize(auth_false, dsize=(camera_w, camera_h), interpolation=cv2.INTER_AREA)
auth_true = cv2.imread('auth_true.jpg')
auth_true = cv2.resize(auth_true, dsize=(camera_w, camera_h), interpolation=cv2.INTER_AREA)
shouldMask = cv2.imread('shouldMask.jpg')
shouldMask = cv2.resize(shouldMask, dsize=(camera_w, camera_h), interpolation=cv2.INTER_AREA)

capture = cv2.VideoCapture(0) # 카메라 불러오기
capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_h)

'''
* 촬영모드 - 필터감지 후 서버로 전송
* 전송모드 - 인증결과 대기
* 마스크감지모드 - 마스크 를 다시써야 문이 열린다
* 소리모드 - 소리재생
'''

capture_mode = True # 촬영 모드, 필터를 체크하고 정해진 시간만큼 필터가 지속적으로 통과하면 전송모드 실행
serve_mode = None # 전송 모드, 서버로 인증정보를 전송하고 전송완료되면 결과출력 모드 실행
show_result_mode = None # 결과출력 모드, 인증성공은 True 실패는 False 모드종료는 None 성공했을경우 마스크 모드 실행
mask_mode = None # 마스크 모드, 일정시간동안만 실행이 되고 마스크를 썻다면 게이트가 열린다
sound_mode = None # 소리재생모드

response= None
emp_name = None

class SendingThread(threading.Thread):
    def __init__(self, args):
        if args == None:
            args=(frame_raw, fvalues, LABEL)
        threading.Thread.__init__(self, name='sending data', args=args)

    def run(self):
        global response, cert_server_address, emp_name

        fvalue = fvalues[0]
        landm = fvalue.landm

        # alignment
        # delta_x = landm['right_eye'][0] - landm['left_eye'][0] # 가로
        # delta_y = landm['right_eye'][1] - landm['left_eye'][1] # 세로
        # angle = np.arctan(delta_y / delta_x) 
        # angle = (angle * 180) / np.pi # 회전각 구하기
        
        # h, w, _ = frame_raw.shape
        # center = (w // 2, h // 2)
        # M = cv2.getRotationMatrix2D(center, (angle), 1)
        # rotated = cv2.warpAffine(frame_raw.copy(), M, (w,h))
        # alignment face value
        # rotated_fvs = get_fv.get_face_value(rotated)
        
        content_type = 'image/png'
        headers = {'content-type': content_type}
        myurl = '{}/certification'.format(cert_server_address)
        response = requests.post(myurl, data=pickle.dumps({'kind': kind, 'label': LABEL, 'img': frame_raw, 'fvalue':fvalue}), headers=headers)
        
        time.sleep(1) 

def check_size(fvalue, range_start, range_end):
    h = fvalue.get_bbox_h()
    print('size', h)
    return range_start < h < range_end

def check_center(fvalue, allow_range):
    bbox_center = fvalue.get_bbox_center()
    a = camera_center[0] - bbox_center[0]
    b = camera_center[1] - bbox_center[1]
    # 카메라 화면의 중앙으로부터 bbox의 중앙까지의 거리
    c_to_b_len = math.sqrt((a * a) + (b * b))
    print('center', c_to_b_len)
    if c_to_b_len < allow_range:
        return True
    else:
        return False

def check_align(fvalue, roll_range, pitch_range, yaw_range):
    bbox = fvalue.bbox
    roll = abs(fvalue.find_roll())
    pitch = abs(fvalue.find_pitch())
    yaw = abs(fvalue.find_yaw())
    print('align', roll, pitch, yaw)
    if (roll < roll_range) & (1< pitch < pitch_range) & (yaw < yaw_range):
        return True
    else:
        return False

def check_no_mask(fvalue, Threshold, frame_raw, size=112):
    # 얼굴 사이즈를 마스크 디텍션 모델의 학습크기와 맞춰준다
    reszie_result, face = fr.face_resize(frame_raw, fvalue.bbox, 112)
    print('mask')
    if reszie_result == True:
        # 마스크 디텍션 모델 실행
        check_maskv = isM.isMasked(face, Threshold)
        return not check_maskv
    

filter_time = None # 필터 가 측정되는 시간
sthread = None # 인증서버로 전송하는 thread 의 참조
mask_check_time = None # 마스크 검사 시간

while True:
    # 프레임단위로 이미지 read
    ret, frame = capture.read()
    # 좌우반전 
    frame = cv2.flip(frame, 1)
    frame_raw = frame.copy()

    # 카메라에서 프레임단위로 이미지 불러오기 성공
    if ret:
        # 촬영모드
        if (capture_mode == True) & (serve_mode != True) & (show_result_mode == None) & (mask_mode == None):
            print('캡처모드')
            # 얼굴 촬영 가이드 표시
            cv2.ellipse(frame, camera_center, (camera_w//4, camera_h//4), 90, 0, 360, (255, 255, 0), 2)
            fvalues = get_fv.get_face_value(frame_raw)

            # 크기필터
            fvalues = filter(lambda fvalue: check_size(fvalue,160, 190), fvalues)
            # 위치 필터
            fvalues = filter(lambda fvalue: check_center(fvalue, 100), fvalues)
            # 3축 필터
            fvalues = filter(lambda fvalue: check_align(fvalue, 7, 1.5, 10), fvalues)
            # no mask
            fvalues = filter(lambda fvalue: check_no_mask(fvalue, 0.5, frame_raw), fvalues)
            # 지연평가
            fvalues = list(fvalues)
            if len(fvalues) > 1:
                # 필터를 통과한 얼굴중 가장 큰 얼굴
                fvalues = reduce(lambda a,b: a if a.get_bbox_size() > b.get_bbox_size() else b, fvalues)
            # fvalues 의 길이를 반드시 0 아니면 1로 고정했다

            # 1초간 필터를 계속 통과할 경우 전송모드로 들어간다
            if len(fvalues) == 1:
                # 처음 필터를 통과한 경우
                if filter_time == None:
                    filter_time = time.time()
                # 이후로 필터를 통과한경우 시간을 측정한다
                else:
                    # 필터를 통과한 시간이 1초를 지났을 경우
                    if time.time() - filter_time > 1:
                        # thread 를 생성해서 서버로 전송
                        send_frame = frame_raw.copy()
                        sthread = SendingThread(args = (send_frame, fvalues, LABEL))
                        sthread.start()
                        filter_time = None
                        serve_mode = True
                        capture_mode = False
            # 필터 실패
            else:
                # 필터시간 초기화
                filter_time = None
        # 전송모드, 필터를 1초간 통과해서 전송모드에 들어갔다
        if serve_mode == True:
            print('전송모드')
            frame = wait
            # 쓰레드가 종료됨
            if not sthread.is_alive():
                serve_mode = False
                # 쓰레드 실행의 결과가 response에 담긴다
                
                if response.status_code == 200:
                    response = response.json()
                    print(response)
                    # 인증이 성공하면 1 아니면 0
                    if response['cert'] == 1:
                        engine.say('"{}".님 인증되었습니다'.format(emp_name))
                        engine.stop() # 끝
                        show_result_mode = True
                        playsound("./Mask.mp3" )
                    else:
                        playsound("./Auth_fail.mp3")
                        show_result_mode = False
                # 인증서버 에러 발생
                else:
                    playsound("./Auth_fail.mp3")
                    capture_mode = True


                # elif response.status_code == 500:
                #     show_result_mode = None
                # else:
                #     show_result_mode = None
        # 결과 출력 모드
        if show_result_mode != None:
            print('결과 출력 모드')
            # 인증됨
            if show_result_mode:
                # 마스크 탐지 모드
                print('인증성공')
                mask_mode = True
                show_result_mode = None
            # 인증실패
            else:
                print('인증실패')
                capture_mode = True
                show_result_mode = None
        # 마스크 디텍션 모드
        if mask_mode == True:
            print('마스크 디텍션 모드')
            if mask_check_time == None:
                mask_check_time = time.time()
            else:
                # 10초간 마스크 감지
                if time.time() - mask_check_time < 10:
                    fvalues = get_fv.get_face_value(frame_raw)
                    reszie_result, faceImg = fr.face_resize(frame_raw, fvalues[0].bbox, 112)
                    if faceImg is None:
                        continue
                    # 얼굴 resize, 마스크디텍션 모델 훈련했을때와 같이 전처리 해준다
                    if reszie_result == True:
                        # 마스크 감지됨
                        if isM.isMasked(faceImg, 0.95) == True:
                            # 문이 열린다
                            playsound("./Gate.mp3" )
                            mask_mode = None
                            mask_check_time = None
                            capture_mode = True
                    else:
                        pass


    # 카메라에서 프레임단위로 이미지 불러오기 실패
    else:
        frame = wait

    cv2.imshow('frame', frame)
    key = cv2.waitKey(33)
    if key == 27:
        capture.release()
        cv2.destroyAllWindows()
        break


    