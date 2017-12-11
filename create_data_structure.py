import glob
import numpy as np
import os
import csv
from shutil import copyfile


writeOut = True
#pathogens = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',Hernia','Infiltration','Mass','Nodule','No Finding','Pleural_Thickening','Pneumonia','Pneumothorax']
pathogens = ['Cardiomegaly','Pneumothorax','Mass','Effusion','No Finding']

file_dict = {}
count_dict = {}
i = 0
with open("Data_Entry_2017.csv") as a:
	reader = csv.reader(a)
	next(reader,None)
	for row in reader:
		#if i < 2200:
			matching = [s for s in pathogens if s in row[1]]
			for match in matching:
				if match in file_dict:
					file_dict[match].append(row[0])
					count_dict[match] += 1
				else:
					file_dict[match] = [row[0]]
					count_dict[match] = 1
		#i += 1

print(count_dict)

if writeOut == True:
	for directory in file_dict:
		if not os.path.exists("data/test/%s"%(directory)):
			os.makedirs("data/train/%s"%(directory))
			os.makedirs("data/validation/%s"%(directory))
			os.makedirs("data/test/%s"%(directory))
			for file in file_dict[directory]:
				if os.path.isfile("images/%s"%(file)):
					val =  np.random.uniform(0,1,1)
					out_dir = "train"
					if val < 0.1:
						out_dir = "validation"
					elif val < 0.2:
						out_dir = "test"
					copyfile("images/%s"%(file), "data/%s/%s/%s"%(out_dir,directory,file))
		else:
			#do stuff
			for file in file_dict[directory]:
				if os.path.isfile("images/%s"%(file)):
					val =  np.random.uniform(0,1,1)
					out_dir = "train"
					if val < 0.1:
						out_dir = "validation"
					elif val < 0.2:
						out_dir = "test"
					copyfile("images/%s"%(file), "data/%s/%s/%s"%(out_dir,directory,file))

