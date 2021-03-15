from random import randint

def gen_key(img, iteration):
    h, w, ch = img.shape
    alpha = 8

    result = []

    for i in range(iteration):
        Kr = [randint(0,pow(2,alpha)-1) for i in range(h)] # h 개수 2의 8 제곱 -1 즉  0 ~ 255 까지의 랜덤숫자배열
        Kc = [randint(0,pow(2,alpha)-1) for i in range(w)]

        key = {'Kr': Kr, 'Kc':Kc}
        # key.Kr = Kr
        # key.Kc = Kc
        result.append(key)
    return result