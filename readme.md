# NMNP - No Mask No Pass

## 실행 환경
* arcface  
git clone https://github.com/peteryuX/arcface-tf2
* mtcnn  
pip install mtcnn
* opencv  
pip install opencv-python
* mariadb  
pip install mariadb  
그외에 여러가지 실행에 필요한 라이브러리 설치

# 체크포인트 파일 다운로드
https://drive.google.com/file/d/1kRdFCCLjWh5rZkVlc5OENALUdFDDU74-/view?usp=sharing

checkpoints 라는 폴더를 만들고 위 링크 파일을 다운로드 받아서 안에 압축파일을 풀어 주세요.

# 서버 실행
arcface_tf2 폴더안 server.py 실행

# 클라이언트 실행
client 폴더안 client.py 실행