class FaceValue:
    def __init__(self, out, w, h):
        self.w = w
        self.h = h

        left = int(out[0]*w)
        top = int(out[1]*h)
        right = int(out[2]*w)
        bottom = int(out[3]*h)
        self.bbox = [left, top, right, bottom]
        
        temp = {}
        temp['left_eye'] = (int(out[4] * w), int(out[5] * h))
        temp['right_eye'] = (int(out[6] * w), int(out[7] * h))
        temp['nose'] = (int(out[8] * w), int(out[9] * h))
        temp['mouse_left'] = (int(out[10] * w), int(out[11] * h))
        temp['mouse_right'] = (int(out[12] * w), int(out[13] * h))
        self.landm = temp
    def get_bbox_h(self):
        bbox = self.bbox
        return bbox[3] - bbox[1]
    
    def get_bbox_w(self):
        bbox = self.bbox
        return bbox[2] - bbox[0]

    def get_bbox_size(self):
        """
        get bbox size
        """
        return self.get_bbox_h() * self.get_bbox_w()
    
    def get_bbox_center(self):
        bbox = self.bbox
        return (bbox[0] + self.get_bbox_w()//2, bbox[1] + self.get_bbox_h()//2)

    def find_roll(self):
        landm = self.landm
        return landm['left_eye'][1] - landm['right_eye'][1]
    
    def find_yaw(self):
        landm = self.landm
        le2n = landm['nose'][0] - landm['left_eye'][0]
        re2n = landm['right_eye'][0] - landm['nose'][0]
        return le2n - re2n
    
    def find_pitch(self):
        landm = self.landm
        eye_y = (landm['left_eye'][1] + landm['right_eye'][1]) / 2
        mou_y = (landm['mouse_left'][1] + landm['mouse_right'][1]) / 2
        e2n = eye_y - landm['nose'][1]
        n2m = landm['nose'][1] - mou_y
        return e2n/n2m