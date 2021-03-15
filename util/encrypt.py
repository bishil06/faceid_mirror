import numpy as np

def one_stage_cube_encrypt(img, Kr, Kc):
	h, w, ch = img.shape
	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]

	for i in range(h):
		row_R_sum = np.sum(r[i])
		row_G_sum = np.sum(g[i])
		row_B_sum = np.sum(b[i])

		if row_R_sum % 2 == 0:
			r[i] = np.roll(r[i], Kr[i])
		else:
			r[i] = np.roll(r[i], -Kr[i])
		if row_G_sum % 2 == 0:
			g[i] = np.roll(g[i], Kr[i])
		else:
			g[i] = np.roll(g[i], -Kr[i])
		if row_B_sum % 2 == 0:
			b[i] = np.roll(b[i], Kr[i])
		else:
			b[i] = np.roll(b[i], -Kr[i])

	img[:,:,0] = b
	img[:,:,1] = g
	img[:,:,2] = r
	return img

def two_stage_cube_encrypt(img, Kr, Kc):
	h, w, ch = img.shape
	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]

	b = np.transpose(b)
	g = np.transpose(g)
	r = np.transpose(r)

	for i in range(w):
		column_R_sum = np.sum(b[i])
		column_G_sum = np.sum(g[i])
		column_B_sum = np.sum(r[i])

		if column_R_sum  % 2 == 0:
			r[i] = np.roll(r[i], Kc[i])
		else:
			r[i] = np.roll(r[i], -Kc[i])
		if column_G_sum % 2 == 0:
			g[i] = np.roll(g[i], Kc[i])
		else:
			g[i] = np.roll(g[i], -Kc[i])
		if column_B_sum % 2 == 0:
			b[i] = np.roll(b[i], Kc[i])
		else:
			b[i] = np.roll(b[i], -Kc[i])
	
	b = np.transpose(b)
	g = np.transpose(g)
	r = np.transpose(r)
	
	img[:,:,0] = b
	img[:,:,1] = g
	img[:,:,2] = r
	return img

def three_stage_cube_encrypt(img, Kr, Kc):
	h, w, ch = img.shape
	b = img[:,:,0]
	g = img[:,:,1]
	r = img[:,:,2]

	Kr = np.array(Kr)
	Kc = np.array(Kc)

	for i in range(h):
		if i%2 == 1:
			r[i] = r[i]^Kc
			g[i] = g[i]^Kc
			b[i] = b[i]^Kc 
		else:
			r[i] = r[i]^Kc[::-1]
			g[i] = g[i]^Kc[::-1]
			b[i] = b[i]^Kc[::-1]
	
	b = np.transpose(b)
	g = np.transpose(g)
	r = np.transpose(r)

	for i in range(w):
		if i%2 == 0:
			r[i] = r[i]^Kr
			g[i] = g[i]^Kr
			b[i] = b[i]^Kr
		else:
			r[i] = r[i]^Kr[::-1]
			g[i] = g[i]^Kr[::-1]
			b[i] = b[i]^Kr[::-1]
	b = np.transpose(b)
	g = np.transpose(g)
	r = np.transpose(r)

	img[:,:,0] = b
	img[:,:,1] = g
	img[:,:,2] = r
	return img

def encrypt(img, keyList):
    temp = img.copy()

    for key in keyList:
        Kr = key.get('Kr')
        Kc = key.get('Kc')

        temp = one_stage_cube_encrypt(temp, Kr, Kc)
        temp = two_stage_cube_encrypt(temp, Kr, Kc)
        temp = three_stage_cube_encrypt(temp, Kr, Kc)
            
    return temp