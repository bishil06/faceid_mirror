import os
import cv2
import sys
# 부모모듈 경로 가져오기
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from util import get_file_list as getFList
from util import face_resize as fr

import get_face_value as get_fv

def resized_img_dataset(path, dest_path):
    files = getFList.get_file_list(path)
    # print(files)

    for filename in files:
        img_path = os.path.join(path, filename)
        print(img_path)
        img = cv2.imread(img_path)
        fvalues = get_fv.get_face_value(img)
        # print(len(fvalues))
        if len(fvalues) == 1:
            result, resized = fr.face_resize(img, fvalues[0]['bbox'], 112)
            if result == False:
                print(img_path, resized)
            elif result == True:
                try:
                    save_path = os.path.join(dest_path, 'resized_{}'.format(filename))
                    cv2.imwrite(save_path, resized)
                except cv2.error as e:
                    print(e)
                    print('write error path =', save_path, 'img = ', result)

# path = r'C:\Users\user\Desktop\no_mask_no_pass\FaceID_Project\server\cert_uploads'
# dest_path = r'C:\Users\user\Desktop\dest'     

# resized_img_dataset(path, dest_path)