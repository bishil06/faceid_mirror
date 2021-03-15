import os
import uuid
import cv2
import datetime
from flask import Flask, render_template, request, jsonify, make_response, Response
from flask import g
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import mariadb
import time
from numpy import dot
from numpy.linalg import norm
import numpy as np
import json
import sys
import codecs
import requests
import io
from PIL import Image
import jsonpickle
import numpy

from numpy import dot
from numpy.linalg import norm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util import generate_key as gk
from util import encrypt as enc
from util import decrypt as dec
from util import FaceValue

import vectorized_face_image as vect_face

app = Flask(__name__)

admin_address = '127.0.0.1:5002'
face_server_address = '127.0.0.1:5001'
db_user = 'root'
db_password = 'q1w2e3r4t5'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ENCRYPTED_FOLDER'] = 'encrypted/'
THRESHOLD = 0.65

def get_dbconn():
    # db 연결
    db_conn = getattr(g, '_dbconn', None)
    if db_conn is None:
        # user=db_user, password=db_pass, host=db_address, database=db_name, port=db_port
        db_conn = mariadb.connect(host='3.34.161.255', port=33061, database='NMNP', user=db_user, password=db_password)
        g._dbconn = db_conn
    return db_conn

@app.teardown_appcontext
def close_conn(exception):
    # db 연결 종료
    db_conn = getattr(g, '_dbconn', None)
    if db_conn is not None:
        db_conn.close()

def save_keyList(photo_name, keyList):
    # 사진 암호화 키 저장
    # 사진 이름과 키 리스트를 photo_key 테이블에 저장을 한다
    conn = get_dbconn()
    cur = conn.cursor()

    # 이미 key 가 저장되어 있는 지 확인
    cur.execute("select count(*) from NMNP.photo_key where photo_name = '{}'".format(photo_name))
    r = cur.fetchone()
    if r[0] >= 1:
        print('이미 데이터베이스 안에 key가 있다')
        # 이미 있는 경우 keyList 를 가져온다음 저장결과 False 와 같이 반환한다
        cur.execute("select keyList from NMNP.photo_key where photo_name = '{}'".format(photo_name))
        keyList = cur.fetchone()[0]
        return False, keyList

    idx = None
    created_at = None
    updated_at = None
    created_by = 'root'
    updated_by = 'root'
    photo_name = photo_name
    print('데이터베이스에 {} keyList 저장'.format(photo_name))
    cur.execute("INSERT INTO NMNP.photo_key(idx, created_at, updated_at, created_by, updated_by, photo_name, keyList) VALUES (?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, photo_name, pickle.dumps(keyList)))
    conn.commit()

    # 저장성공
    return True, None

@app.route('/reg_emp')
def get_reg_emp():
    return render_template('reg_emp.html')

# 직원 혹은 방문자 저장
@app.route('/reg_emp_result', methods = ['POST'])
def upload_emp_info():
    print(request.form) # 디버깅용 임시 코드
    # ImmutableMultiDict([('emp_num', ''), ('Name', ''), ('Age', ''), ('Team', ''), ('Pos', '')])
    fo = request.form 
    f = request.files['file']

    # 입력값 검증 코드 - 현재는 빈입력만을 체크하지만 설계로는 모든값이 제대로 입력되었는지 검증하는 단계
    for v in fo.values():
        if v == '':
            return 'fail'

    img_name = '{}_{}.png'.format(fo['emp_num'], os.path.basename(f.filename))
    # 전송받은 사진 저장 경로
    img_path = os.path.join(app.config['ENCRYPTED_FOLDER'], img_name)
    
    # post로 전달받은 파일을 steam으로 읽어와서
    image = f.read()
    image_stream = io.BytesIO(image)
    # numpy 로 변환한뒤
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    # opencv 로 디코딩해준다
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 암호화할 key 생성
    keyList = gk.gen_key(img, 1)
    # 암호화
    enc_img = enc.encrypt(img, keyList)
    # key 저장
    save_result, keyList = save_keyList(img_name, keyList)

    # 암호화된 이미지 저장
    cv2.imwrite(img_path, enc_img)

    conn = get_dbconn()
    cur = conn.cursor()

    # db 저장 시간 측정
    start = time.time()

    # 방문자 등록 인 경우와 사원등록을 구분해서 동작하도록 한다
    if fo.get('visitor_name') == None:
        # 방문자 등록이 아닌경우
        idx = None
        created_at = None
        updated_at = None
        created_by = db_user
        updated_by = db_user
        emp_num = fo['emp_num']
        emp_name = fo['Name']
        emp_age = fo['Age']
        emp_team = fo['Team']
        emp_pos = fo['Pos']
        extra = None
    else:
        # 방문자 등록인 경우
        idx = None
        created_at = None
        updated_at = None
        created_by = db_user
        updated_by = db_user
        emp_num = '{}'.format(uuid.uuid4())
        emp_name = fo['visitor_name']
        emp_age = 0
        emp_team = fo['visitor_phone']
        emp_pos = fo['visitor_address']
        extra = fo['visitor_why']
    try: 
        cur.execute("INSERT INTO NMNP.employee(idx, created_at, updated_at, created_by, updated_by, emp_num, emp_name, emp_age, emp_team, emp_pos, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (idx, created_at, updated_at, created_by, updated_by, emp_num, emp_name, emp_age, emp_team, emp_pos, extra))
    except mariadb.Error as e: 
        print(f"Error: {e}")
    print('db에 직원정보 입력하는데 걸리는 시간 ', time.time()-start)
    conn.autocommit = True

    return 'success'

def get_kor_now():
    utcnow = datetime.datetime.utcnow()
    time_gap = datetime.timedelta(hours=9)
    kor_time= utcnow + time_gap
    return kor_time

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def load_emp_photo():
    conn = get_dbconn()
    cur = conn.cursor()

    try:
        cur.execute('SELECT * FROM NMNP.emp_photo')
        emp_list = cur.fetchall()
    except mariadb.Error as e: 
        print(f"Error: {e}")
    
    def make_emp_photo(emp):
        idx, created_at, updated_at, created_by, updated_by, emp_num, emp_photo_path, emp_photo = emp
        result = {}
        result['idx'] = idx
        
        result['created_at'] = created_at
        result['updated_at'] = updated_at

        result['created_by'] = created_by
        result['updated_by'] = updated_by
        result['emp_num'] = emp_num
        result['emp_photo_path'] = emp_photo_path
        result['emp_photo'] = pickle.loads(emp_photo)
        # print(result)
        return result
    emp_photo_list = [make_emp_photo(emp) for emp in emp_list]
    return emp_photo_list

def save_transaction(req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error):
    conn = get_dbconn()
    cur = conn.cursor()

    idx = None
    created_at = None
    updated_at = None
    created_by = 'root'
    updated_by = 'root'
    high_sim = high_sim.__float__()
    error = None

    try:
        cur.execute("INSERT INTO NMNP.transaction(idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error))
        cur.execute('select emp_name from NMNP.employee where emp_num = "{}"'.format(highest_emp_num))
        out_name = cur.fetchall()
    except mariadb.Error as e: 
        print(f"Error: {e}")
        return ''
    conn.autocommit = True
    return out_name[0][0]

def cos_sim_from_emp(v):
    # db 에서 사원사진정보 불러오기
    emp_photo_list = load_emp_photo()
    
    # emp_num, emp_photo
    temp = map(lambda emp: (emp['emp_num'], emp['emp_photo']), emp_photo_list)
    # emp_num, cos_sim
    temp = map(lambda t: (t[0], cos_sim(v[0], t[1][0])), temp)
    result = list(temp)
    # sort
    result.sort(key=lambda tup: tup[1])
    return result

# 인증요청 처리
@app.route('/certification', methods = ['POST'])
def certification():
    req_time = get_kor_now().strftime('%Y-%m-%d %H:%M:%S.%f')

    print(request.data)
    data = pickle.loads(request.data)
    print(data)

    img_name = '{}.png'.format(uuid.uuid4())
    # 전송받은 사진 저장 경로
    img_path = os.path.join(app.config['ENCRYPTED_FOLDER'], img_name)

    # 전달받은 인증정보들 받아오기
    label = data['label']
    kind = data['kind']
    img = data['img']
    bbox = data['fvalue'].bbox

    # 얼굴사진 resize 시도 - 실패하면 다시 사진 보내달라고 요청해야한다
    face_server_res = requests.post('http://{}/photoUpload'.format(face_server_address), data=pickle.dumps(img))
    # 요청결과는 json 으로 성공여부와 이미지를 받는다
    resize_result = face_server_res.json()
    # print(resize_result)
    # 얼굴을 찾아 resize하는데 실패했다면 인증실패를 반환한다
    if resize_result['success'] == False:
        pass
        # return 'fail'
    faceImg = resize_result['face']
    faceImg = pickle.loads(codecs.decode(faceImg.encode(), "base64"))

    # 얼굴 resize가 되었다면 원본이미지를 암호화하고 저장한다
    # 암호화할 key 생성
    keyList = gk.gen_key(img, 1)
    # 암호화
    enc_img = enc.encrypt(img, keyList)
    # key 저장
    save_result, keyList = save_keyList(img_name, keyList)
    # 암호화된 이미지 저장
    cv2.imwrite(img_path, enc_img)

    # 얼굴사진 벡터화
    v = vect_face.get_vect_face_img(faceImg).numpy()

    # 인증이 되었다면 1 안되었다면 0
    certification = None
    
    cos_sim_list = cos_sim_from_emp(v)
    print(cos_sim_list)
    if cos_sim_list[-1][1] > THRESHOLD:
        certification = 1
    else:
        certification = 0

    req_end_time = get_kor_now().strftime('%Y-%m-%d %H:%M:%S.%f')
    
    req_type = kind
    req_time = req_time
    req_end_time = req_end_time
    req_from = label
    req_photo_path = img_path
    req_photo = pickle.dumps(v)
    high_sim = cos_sim_list[-1][1]
    highest_emp_num = cos_sim_list[-1][0]
    error = None

    # 트랜잭션 테이블에 요청정보 저장
    out_name = save_transaction(req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error)
    print(out_name)

    return make_response(jsonify({'cert': certification, 'high_sim_emp': highest_emp_num, 'high_sim': high_sim.__float__(), 'emp_name': out_name}), 200)

@app.route('/get_encrypt_photo/<filename>')
def get_encrypt_photo(filename):
    find = None

    # 파일이 encrypted 폴더에 있나 검색
    fileList = os.listdir('encrypted')
    for photo in fileList:
        if filename == photo:
            find = True
            break

    if find == True:
        img_path = os.path.join(app.config['ENCRYPTED_FOLDER'], filename)
        enc_img = cv2.imread(img_path)
        _, img_encoded = cv2.imencode('.png', enc_img)
        response = make_response(img_encoded.tostring())
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        print('{} 파일을 찾지 못했습니다.'.format(filename))
        # img = cv2.imread('pic3.png')
        # _, img_encoded = cv2.imencode('.png', img)
        # response = make_response(img_encoded.tostring())
        # response.headers['Content-Type'] = 'image/png'
        # return response
        return False
    
if __name__ == '__main__':
     #서버 실행
    app.run('0.0.0.0', port=50000)
