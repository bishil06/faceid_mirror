import os
import uuid
import cv2
import datetime
from flask import Flask, render_template, request, jsonify, make_response, Response
from flask import g
from werkzeug.utils import secure_filename # 업로드 된 파일의 이름이 안전한가 확인
import numpy as np
import pickle
import mariadb
import time
from datetime import datetime 
from numpy import dot
from numpy.linalg import norm
import numpy as np
import json
import sys
import codecs
import requests

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util import face_resize as fr

from util import generate_key as gk
from util import encrypt as enc
from util import decrypt as dec

import vectorized_face_image as vect_face

app = Flask(__name__)

admin_address = '127.0.0.1:5002'

UPLOAD_FOLDER = 'uploads/'
RESIZE_FOLDER = 'resized_face/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESIZE_FOLDER'] = RESIZE_FOLDER

# server_ip = '61.82.106.114'

def get_dbconn():
    db_conn = getattr(g, '_dbconn', None)
    if db_conn is None:
        # user=db_user, password=db_pass, host=db_address, database=db_name, port=db_port
        db_conn = mariadb.connect(host='3.34.161.255', port=33061, database='NMNP', user='root', password='q1w2e3r4t5')
        g._dbconn = db_conn
    return db_conn

@app.teardown_appcontext
def close_conn(exception):
    db_conn = getattr(g, '_dbconn', None)
    if db_conn is not None:
        db_conn.close()

@app.route('/reg_emp')
def get_reg_emp():
    return render_template('reg_emp.html')

@app.route('/reg_emp_result', methods = ['POST'])
def upload_emp_info():
    print(request.form)
    fo = request.form
    f = request.files['file']

    # 전송받은 사진 저장
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename) # 이름을 지정해서 저장해야함
    f.save(img_path)

    conn = get_dbconn()
    cur = conn.cursor()
    
    # ------------아직 코드 작성중------------------
    start = time.time()
    
    if fo.get('visitor_name') == None:
        # 방문자 등록이 아닌경우
        idx = None
        created_at = None
        updated_at = None
        created_by = 'root'
        updated_by = 'root'
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
        created_by = 'root'
        updated_by = 'root'
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

    # 랜드마크 값으로 얼굴추출해서 face alignment 수행하고 embeding
    # 랜드마크 값으로 얼굴추출해서 face alignment 수행하고 embeding
    img = cv2.imread(img_path)
    retina_server_address = '127.0.0.1:5001'
    retina_response = requests.post('http://{}/photoUpload'.format(retina_server_address), data=pickle.dumps(img))

    faceImg = retina_response.json()['face']
    faceImg = pickle.loads(codecs.decode(faceImg.encode(), "base64"))

    v = vect_face.get_vect_face_img(faceImg).numpy()
    print(v)

    try:
        cur.execute("INSERT INTO NMNP.emp_photo(idx, created_at, updated_at, created_by, updated_by, emp_num, emp_photo_path, emp_photo) VALUES (?,?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, emp_num, img_path, pickle.dumps(v)))
    except mariadb.Error as e: 
        print(f"Error: {e}")
    return 'good'

@app.route('/certification', methods = ['POST'])
def certification():
    req_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(request)
    data = pickle.loads(request.data)
    
    label = data['label']
    kind = data['kind']
    angle = data['angle']
    frame_raw = data['img']
    bbox = data['fvalue'][0]['bbox']

    img_path = os.path.join('cert_uploads/', '{}.jpg'.format(uuid.uuid4()))
    cv2.imwrite(img_path, frame_raw)

    h, w, _ = frame_raw.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, (angle), 1)
    rotated = cv2.warpAffine(frame_raw.copy(), M, (w,h))

    resize_result, faceImg = fr.face_resize(frame_raw, bbox, 112)
    # cv2.imwrite('haha.jpg', faceImg)
    v = vect_face.get_vect_face_img(faceImg).numpy()

    conn = get_dbconn()
    cur = conn.cursor()

    try:
        cur.execute('SELECT * FROM NMNP.emp_photo')
        emp_list = cur.fetchall()
    except mariadb.Error as e: 
        print(f"Error: {e}")
    
    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))
    
    # print(emp_list)
    def load_emp(emp):
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
    emp_list = [load_emp(emp) for emp in emp_list]
    
    message = ''

    certification = 0
    # 직원이 저장되어 있지 않은 경우
    if len(emp_list) == 0:
        message = '직원이 없다'
        return {'success':False, 'msg':message}
    else:
        high_sim_emp = emp_list[0]
        high_sim = cos_sim(high_sim_emp['emp_photo'], v[0])
    # 직원이 한명인 경우
    if len(emp_list) == 1:
        message = '직원이 한명'
    else:
        # 직원이 여러명인 경우
        for emp in emp_list[0:]:
            sim = cos_sim(emp['emp_photo'], v[0])
            print(emp['emp_num'], sim)
            if high_sim < sim:
                high_sim_emp = emp
                high_sim = sim

    if high_sim < 0.65:
        message = '같은 사람이 없습니다. ID: {}가 가장 유사한 사람 {}'.format(high_sim_emp['emp_num'], high_sim)
        print(message)
    else:
        certification = 1
        message = 'ID: {}가 가장 유사한 사람 {}'.format(high_sim_emp['emp_num'], high_sim)
        print(message)
        

    req_end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

    idx = None
    created_at = None
    updated_at = None
    created_by = 'root'
    updated_by = 'root'
    req_type = kind
    req_time = req_time
    req_end_time = req_end_time
    req_from = label
    req_photo_path = img_path
    req_photo = pickle.dumps(v)
    certification = certification
    high_sim = high_sim.__float__()
    print(high_sim, data['fvalue'][0]['bbox_size'])
    highest_emp_num = high_sim_emp['emp_num']
    error = None
    
    try:
        cur.execute("INSERT INTO NMNP.transaction(idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error))
        cur.execute('select emp_name from NMNP.employee where emp_num = "{}"'.format(highest_emp_num))
        out_name = cur.fetchall()
    except mariadb.Error as e: 
        print(f"Error: {e}")
    conn.autocommit = True
    # 반환해야할 값 - 인증이 성공했는지, 성공했다면 누구와 가장 유사했는지, 유사도는 어느정도 였는지, 
    
    return make_response(jsonify({'cert': certification, 'high_sim_emp': highest_emp_num, 'high_sim': high_sim, 'emp_name': out_name[0][0]}), 200)

def send_file(url, file_path):
    file = {'file': (file_path.split('/')[-1], open(file_path, 'rb'))}
#     data = {"id" : "2345AB"}
    response = requests.post(url, files=file, timeout=5)
    return response

@app.route('/get_emp_photo/<filename>')
def get_emp_photo(filename):
    conn = get_dbconn()
    cur = conn.cursor()

    print('요청된 파일', filename)
    if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        print('폴더에 파일이 존재함')

        # 이미 암호화된 파일이 있는지 검사
        if os.path.isfile(os.path.join('encrypted/', '{}.png'.format(filename))):
            print('이미 암호화됨')
            # upload 경로로 파일 전송
            r = send_file('http://127.0.0.1:5002/upload', os.path.join('encrypted/', '{}.png'.format(filename)))
            print(r)
            return make_response(jsonify({'success':True}), 200)
        # 암호화된 파일이 없다
        else:
            print('암호화 시작')
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = cv2.imread(img_path)
            
            # 암호화를 위한 키 생성
            keyList = gk.gen_key(img, 1) 
            
            # 키를 데이터베이스에 저장
            idx = None
            created_at = None
            updated_at = None
            created_by = 'root'
            updated_by = 'root'
            photo_name = filename
            cur.execute("INSERT INTO NMNP.photo_key(idx, created_at, updated_at, created_by, updated_by, photo_name, keyList) VALUES (?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, photo_name, pickle.dumps(keyList)))
            conn.autocommit = True

            print(keyList)

            # 암호화
            enc_img = enc.encrypt(img, keyList)

            # 암호화된 이미지 저장
            enc_img_path = os.path.join('encrypted/', '{}.png'.format(filename))
            cv2.imwrite(enc_img_path, enc_img)
            # upload 경로로 파일 전송
            r = send_file('http://127.0.0.1:5002/upload', enc_img_path)
            print(r)

            # 암호화된 이미지 전송
            return make_response(jsonify({'success':True}), 200)
    else:
        print('파일이 존재하지 않음', os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return make_response(jsonify({'success':False}), 200)

@app.route('/get_cert_photo/<filename>')
def get_cert_photo(filename):
    conn = get_dbconn()
    cur = conn.cursor()

    print('요청된 파일', filename)
    if os.path.isfile(os.path.join('cert_uploads', filename)):
        print('폴더에 파일이 존재함')

        # 이미 암호화된 파일이 있는지 검사
        if os.path.isfile(os.path.join('encrypted/', filename)):
            print('이미 암호화됨')
            # upload 경로로 파일 전송
            r = send_file('http://127.0.0.1:5002/upload', os.path.join('encrypted/', filename))
            print(r)
            return make_response(jsonify({'success':True}), 200)
        # 암호화된 파일이 없다
        else:
            print('암호화 시작')
            img_path = os.path.join('cert_uploads', filename)
            img = cv2.imread(img_path)
            
            # 암호화를 위한 키 생성
            keyList = gk.gen_key(img, 1) 
            print(keyList)
            
            # 키를 데이터베이스에 저장
            idx = None
            created_at = None
            updated_at = None
            created_by = 'root'
            updated_by = 'root'
            photo_name = filename
            
            cur.execute("INSERT INTO NMNP.photo_key(idx, created_at, updated_at, created_by, updated_by, photo_name, keyList) VALUES (?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, photo_name, pickle.dumps(keyList)))
            conn.commit()

            # 암호화
            enc_img = enc.encrypt(img, keyList)

            # 암호화된 이미지 저장
            enc_img_path = os.path.join('encrypted/', '{}.png'.format(filename))
            cv2.imwrite(enc_img_path, enc_img)
            # upload 경로로 파일 전송
            r = send_file('http://127.0.0.1:5002/upload', enc_img_path)
            print(r)

            # 암호화된 이미지 전송
            return make_response(jsonify({'success':True}), 200)
    else:
        print('파일이 존재하지 않음', os.path.join('encrypted/', filename))
        return make_response(jsonify({'success':False}), 200)

import jsonpickle

def save_keyList(photo_name, keyList):
    conn = get_dbconn()
    cur = conn.cursor()

    cur.execute("select count(*) from NMNP.photo_key where photo_name = '{}'".format(photo_name))
    r = cur.fetchone()

    if r[0] >= 1:
        print('이미 데이터베이스 안에 key가 있다')
        cur.execute("select keyList from NMNP.photo_key where photo_name = '{}'".format(photo_name))
        keyList = cur.fetchone()[0]
        return False, keyList
    else:
        print('데이터베이스 안에 key가 없다')

    idx = None
    created_at = None
    updated_at = None
    created_by = 'root'
    updated_by = 'root'
    photo_name = photo_name
    
    print('데이터베이스에 {} keyList 저장'.format(photo_name))
    cur.execute("INSERT INTO NMNP.photo_key(idx, created_at, updated_at, created_by, updated_by, photo_name, keyList) VALUES (?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, photo_name, pickle.dumps(keyList)))
    conn.commit()

    return True, None

@app.route('/get_encrypt_photo/<filename>')
def get_encrypt_photo(filename):
    find = None

    # 파일이 uploads 에 있나 검색
    uploads = os.listdir('uploads')
    for photo in uploads:
        if filename == photo:
            find = 'uploads'
            break
    
    # 파일이 cert_uploads 에 있나 검색
    if find == None:
        cert = os.listdir('cert_uploads')
        for photo in cert:
            if filename == photo:
                find = 'cert_uploads'
                break
    
    if find == None:
        print('{} 파일을 찾지 못했습니다.'.format(filename))
        img = cv2.imread('pic3.png')
        _, img_encoded = cv2.imencode('.png', img)
        response = make_response(img_encoded.tostring())
        response.headers['Content-Type'] = 'image/png'
        return response
    else:
        img = None
        img_path = os.path.join(find, filename)
        img = cv2.imread(img_path)

        keyList = gk.gen_key(img, 1)

        findKey, saved_keyList = save_keyList(filename, keyList)
        start = time.time()
        if findKey:
            enc_img = enc.encrypt(img, keyList)
            print('{} 암호화 시간 {}'.format(filename, time.time()-start))
            _, img_encoded = cv2.imencode('.png', enc_img)
            response = make_response(img_encoded.tostring())
            response.headers['Content-Type'] = 'image/png'
            return response
        else:
            keyList = pickle.loads(saved_keyList)
            enc_img = enc.encrypt(img, keyList)
            print('{} 암호화 시간 {}'.format(filename, time.time()-start))
            _, img_encoded = cv2.imencode('.png', enc_img)
            response = make_response(img_encoded.tostring())
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
     #서버 실행
    app.run(host='0.0.0.0', port=50000)


