# client
동작방식  
캠화면을 출력하고 사진을 찍어서 얼굴을 추출한뒤 서버로 전송  

# requirement
## checkpoints 
* retinaface
미리 체크포인트를 받아둬야 에러가 발생하지 않습니다   
https://drive.google.com/file/d/1tqdNqXosArC0X1F5AXT8ta3nIIL2P99b/view?usp=sharing  
위 파일을 받아서 server 폴더 안에 checkpoints 라는 폴더를 만들고 파일의 압축을 풀어주세요.  
그래야 [*] Cannot find ckpt from None. 라는 에러가 발생하지 않습니다.
## dependencies
* python 3.7 이상
* tensorflow 2.1 이상
* numpy
* opencv-python
* PyYAML
* tqdm
* Pillow
* Cython
* ipython

# 클라이언트 실행
client.py 안에 있는 myurl을 서버의 주소로 작성해주세요.  
py client.py   
정상적으로 실행이 되고 카메라 화면이 출력됨  
c 누르면 촬영 -> retinaface 모델로 얼굴 추출후 서버로 전송

# 에러 발생 대응
혹시 실행 에러가 발생할 경우 사용자 폴더 안에 있는 keras 폴더에  
https://drive.google.com/file/d/1Qf1xTvzDNX5aQFebhV9386wf8vGF2URY/view?usp=sharing  
이 파일을 받아서 넣어주면 된다
