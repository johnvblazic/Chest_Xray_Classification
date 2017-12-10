from scipy import misc
import os
import glob
import numpy as np

filesD = []
files = sorted(glob.glob("images/*.png"),key=os.path.abspath)
imgArray = np.zeros(1048576,)
for i in range(len(files)):
	filesD.append(files[i])
	image = misc.imread(files[i])
	#print(image.shape)
	if image.ndim > 2:
		#print(image)
		#print(image.ndim)
		image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
		print(files[i])
	'''print(image.shape)
	print(os.path.basename(files[i])[:-4])
	if i == 12 :
		#print(image)
		#print(image.ndim)
		image = np.dot(image[...,:3], [0.299, 0.587, 0.114]) 
		print(image)
	if i == 11:
		print(image)'''