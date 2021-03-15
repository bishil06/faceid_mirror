import pickle
import uuid
from flask import Flask, request, jsonify, make_response
import cv2
import os
import codecs

import get_face_value as get_fv
import face_resize as fr

save_path = 'uploads'
resized_save_path = 'resized_faces'

app = Flask(__name__)

@app.route('/photoUpload', methods=['POST'])
def photoUpload():
    r = request
    if r.method == 'POST':
        img = pickle.loads(r.data)
        print(img)

        # 얼굴 찾기 -> 얼굴 resize -> 얼굴사진 반환
        fvalues = get_fv.get_face_value(img)
        print(fvalues)
        if (len(fvalues) == 1):
            result, resized = fr.face_resize(img, fvalues[0]['bbox'], 112)
            if result == False:
                # 사이즈 문제로 실패
                return make_response(jsonify({'success':False, 'face':None}))
            elif result == True:
                # 성공
                try:
                    face_save_path = os.path.join(resized_save_path, 'resized_{}.png'.format(uuid.uuid4()))
                    cv2.imwrite(face_save_path, resized)
                except cv2.error as e:
                    print(e)
                    print('write error path =', face_save_path, 'img = ', result)
            return make_response(jsonify({'success':True, 'face': codecs.encode(pickle.dumps(resized), "base64").decode()}), 200)
        else:
            # 얼굴이 여러명이라 실패
            return make_response(jsonify({'success':False, 'face':None}))
    else:
        return make_response(jsonify({'success':False, 'face':None}))

if __name__ == '__main__':
     #서버 실행
    app.run(host='0.0.0.0', port=5001)