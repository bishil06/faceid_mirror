import cv2

# import get_face_value as get_fv

def face_resize(img, bbox, size):
    img_h, img_w, _ = img.shape
    x1, y1, x2, y2 = bbox 

    bb_h = y2 - y1
    bb_w = x2 - x1
    
    bb_w_center = x1 + ((x2 - x1)/2)
    bb_h_center = y1 + ((y2 - y1)/2)
    
    if bb_h > bb_w:
        pad = int((bb_h - bb_w)/2)
        x1 = x1-pad
        x2 = x2+pad
        y1 -= 15
    if bb_h < bb_w:
        pad = int((bb_w - bb_h)/2)
        y1 -= pad
        y2 += pad
    
    if (x1 >= 0) & (y1 >= 0) & (x2 <= img_w) & (y2 <= img_h) & (min(bbox) >= 0):
        try:
            resized = cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation = cv2.INTER_AREA)
            #cv2.imwrite('./resizes.jpg', resized)
            return (True, resized)
        except cv2.error as e:
            print(e)
            print('resize error img size = ', img_h, img_w, 'bbox = ', bbox, 'x1y1 x2y2', x1, y1, x2, y2, 'pad', pad)
    else:
        return (False, None)


# img = cv2.imread('data/iu.jpg')
# fv = get_fv.get_face_value(img)

# resized = face_resize(img, fv[0]['bbox'], 112)
# cv2.imshow('test', resized)
# cv2.waitKey(0)