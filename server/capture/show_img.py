import cv2

def show_img(annotation, img):
    cv2.imshow(annotation, img)
    while True:
        key = cv2.waitKey(0)
        if key == 27: break
    cv2.destroyAllWindows()