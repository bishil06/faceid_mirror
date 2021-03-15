import cv2 ## openCV
import datetime ## 캡처파일 날짜 지정

def capture_face_img():
    capture = cv2.VideoCapture(0) # 카메라 불러오기

    while True:
        ret, frame = capture.read() # 카메리의 입력을 읽어와서
        frame = cv2.flip(frame, 1) ##출력영상 좌우반전
        cv2.imshow("Frame", frame) # 화면에 출력합니다
        
        key = cv2.waitKey(33)
        
        if key == 99: # c 키 의 ascii 코드가 99
            capture.release()
            cv2.destroyAllWindows()
            return frame
        # elif key == 27: break # esc 키를 누르면 카메라 종료
    #WebCam 종료

# capture_face_img('./captured_img')