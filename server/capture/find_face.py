from mtcnn import MTCNN # mtcnn 불러오기

def find_face(img):
    detector = MTCNN()
    face_values = detector.detect_faces(img)
    return face_values
    