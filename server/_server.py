import os
import uuid
import cv2
import datetime
from flask import Flask, render_template, request, jsonify, make_response
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

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# from client import get_face_value as get_fv

from util import face_resize as fr
# import capture as cap # capture.py 에서 함수를 불러온다
import vectorized_face_image as vect_face

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

server_ip = '61.82.106.114'

#업로드 HTML 렌더링
# @app.route('/upload')
# def render_file():
#    return render_template('upload.html')

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

# 테스트용 코드
# form 테이블 전송
@app.route('/test')
def test():
    # return render_template('test.html')
    return render_template('test.html')
# form 값을 post 로 전송받음
@app.route('/test_result', methods = ['POST'])
def test_result():
    r = request.form
    f = request.files['file']
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    return render_template('test_result.html', result = r, file=f)

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
    try: 
        cur.execute("INSERT INTO NMNP.employee(idx, created_at, updated_at, created_by, updated_by, emp_num, emp_name, emp_age, emp_team, emp_pos) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (idx, created_at, updated_at, created_by, updated_by, emp_num, emp_name, emp_age, emp_team, emp_pos))
    except mariadb.Error as e: 
        print(f"Error: {e}")
    print('db에 직원정보 입력하는데 걸리는 시간 ', time.time()-start)
    conn.autocommit = True

    # 랜드마크 값으로 얼굴추출해서 face alignment 수행하고 embeding
    img = cv2.imread(img_path)
    # fvalues = get_fv.get_face_value(img)
    # resize_result, faceImg = fr.face_resize(img, fvalues[0]['bbox'], 112)
    v = vect_face.get_vect_face_img(img).numpy()
    print(v)
    try:
        cur.execute("INSERT INTO NMNP.emp_photo(idx, created_at, updated_at, created_by, updated_by, emp_num, emp_photo_path, emp_photo) VALUES (?,?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, emp_num, img_path, pickle.dumps(v)))
    except mariadb.Error as e: 
        print(f"Error: {e}")
    return 'good'

@app.route('/req_cert')
def get_cert_page():
    return render_template('cert.html')

@app.route('/cert', methods = ['POST'])
def cert():
    req_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

    f = request.files['file']
    img_path = os.path.join('cert_uploads/', '{}.jpg'.format(uuid.uuid4()))
    cv2.imwrite(img_path)
    v = vect_face.get_vect_face_img(cv2.imread(img_path)).numpy()
    print(v)

    conn = get_dbconn()
    cur = conn.cursor()

    try:
        cur.execute('SELECT * FROM NMNP.emp_photo')
        emp_list = cur.fetchall()
    except mariadb.Error as e: 
        print(f"Error: {e}")

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))
    
    
    # print(lst) # 7 번 인덱스값이 벡터값이다
    # print(lst[:,7])

    # emp_vects = [emp_v[7] for emp_v in lst]
    # emp_vects = list(map(lambda v: pickle.loads(v), emp_vects))
    # print(v[0])
    # print(v.shape)
    # cos_sims = list(map(lambda emp_v: cos_sim(v[0], emp_v[0]), emp_vects))
    # if max(cos_sims)
    
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
        print(result)
        return result
    emp_list = [load_emp(emp) for emp in emp_list]

    message = ''

    certification = 0
    # 직원이 저장되어 있지 않은 경우
    if len(emp_list) == 0:
        message = '직원이 없다'
        return {'success':False}

    high_sim_emp = emp_list[0]
    high_sim = cos_sim(high_sim_emp['emp_photo'], v[0])
    # 직원이 한명인 경우
    if len(emp_list) == 1:
        message = '직원이 한명'
    else:
        # 직원이 여러명인 경우
        for emp in emp_list[1:]:
            sim = cos_sim(emp['emp_photo'], v[0])
            if high_sim < sim:
                high_sim_emp = emp
                high_sim = sim

    if high_sim < 0.7:
        message = '같은 사람이 없습니다. {}가 가장 유사한 사람 {}'.format(high_sim_emp['emp_num'], high_sim)
    else:
        certification = 1
        message = '{}가 가장 유사한 사람 {}'.format(high_sim_emp['emp_num'], high_sim)

    req_end_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

    idx = None
    created_at = None
    updated_at = None
    created_by = 'root'
    updated_by = 'root'
    req_type = '출근'
    req_time = req_time
    req_end_time = req_end_time
    req_from = 'web'
    req_photo_path = img_path
    req_photo = pickle.dumps(v)
    certification = certification
    high_sim = high_sim.__float__()
    print(high_sim)
    highest_emp_num = high_sim_emp['emp_num']
    error = None

    try:
        cur.execute("INSERT INTO NMNP.transaction(idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error))
    except mariadb.Error as e: 
        print(f"Error: {e}")
    conn.autocommit = True

    return message

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

    resize_result, faceImg = fr.face_resize(rotated, bbox, 112)
    #cv2.imwrite('./test123.jpg', faceImg)
    v = vect_face.get_vect_face_img(faceImg).numpy()

    # rotated_face = rotated[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    # key = cv2.waitKey(0)

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
        print(result)
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
        for emp in emp_list[1:]:
            sim = cos_sim(emp['emp_photo'], v[0])
            if high_sim < sim:
                high_sim_emp = emp
                high_sim = sim

    if high_sim < 0.65:
        message = '같은 사람이 없습니다. {}가 가장 유사한 사람 {}'.format(high_sim_emp['emp_num'], high_sim)
    else:
        certification = 1
        message = '{}가 가장 유사한 사람 {}'.format(high_sim_emp['emp_num'], high_sim)

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
    print(high_sim)
    highest_emp_num = high_sim_emp['emp_num']
    error = None

    try:
        cur.execute("INSERT INTO NMNP.transaction(idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (idx, created_at, updated_at, created_by, updated_by, req_type, req_time, req_end_time, req_from, req_photo_path, req_photo, certification, high_sim, highest_emp_num, error))
    except mariadb.Error as e: 
        print(f"Error: {e}")
    conn.autocommit = True
    # 반환해야할 값 - 인증이 성공했는지, 성공했다면 누구와 가장 유사했는지, 유사도는 어느정도 였는지, 
    
    return make_response(jsonify({'cert': certification, 'high_sim_emp': highest_emp_num, 'high_sim': high_sim}), 200)

@app.route('/')
def inc_test():
    conn = get_dbconn()
    cur = conn.cursor()

    start = time.time()
    count = 0
    cur.execute("select count(*) as count from NMNP.inc_test")
    for c in cur:
            print(c)
            count = c[0]+1
    print('db 에서 inc_test 값 개수 가져오는데 걸리는 시간 ', time.time()-start)
    
    start = time.time()
    cur.execute("INSERT INTO NMNP.inc_test(idx, num) VALUES (?)",(count, 1))
    conn.autocommit = True
    print('db에 숫자값 하나 입력하는데 걸리는 시간 ', time.time()-start)
    return render_template('inc_test.html')

#파일 업로드 처리
@app.route('/fileUpload', methods = ['POST']) # post 요청 처리
def upload_file():
    r = request
    if r.method == 'POST':
        data = pickle.loads(r.data)
        img = data['img']
        # 데이터를 잘 받았나 디버깅용 코드
        print('label', data['label'])
        # cv2.imshow(data['label'], img)
        # cv2.waitKey(0)
        # f = request.files['file']
        # print(request.files)

        #저장할 경로 + 파일명
        # filename = secure_filename(f.filename)
        file_path = os.path.join('uploads', "{}_{}.jpg".format(data['label'], uuid.uuid4()))
        cv2.imwrite(file_path, img)

        v = vect_face.get_vect_face_img(img).numpy() # arcface 벡터화
        print(v)

        conn = get_dbconn()
        cur = conn.cursor()

        
        count = 0
        cur.execute("select count(*) as count from NMNP.test")
        for c in cur:
            print(c)
            count = c[0]+1
        
        cur.execute("INSERT INTO NMNP.test(idx, label, photo) VALUES (?, ?, ?)",
                (count, data['label'], pickle.dumps(v)))
        conn.autocommit = True

        # f.save(file_path) 
        # print(vect_face.get_vect_face_img(cv2.imread(file_path)).numpy())
        return '전송 성공 ^^ {}'.format(datetime.datetime.now())

if __name__ == '__main__':
    #서버 실행
    app.run(host='0.0.0.0')

