import os
import cv2
import numpy as np
from PIL import Image
import pickle

imgpath = './Data/img'
txtpath = './Data/txt'
data =[]
def getImages():
	for image_Name in os.listdir(imgpath):
		image_path = os.path.join(imgpath,image_Name)
		#print(image_path)
		text_Name = image_Name.replace('.png','.txt')
		text_path = os.path.join(txtpath,text_Name)
		text_path_new = text_path.replace('\\','/')
		with open(text_path_new,mode = 'r', encoding = 'utf8') as f:
			text = f.read()

		image = cv2.imread(image_path)

		# cv2.imshow("imagetest",image)
		# cv2.waitKey()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#image = cv2.resize(image,(224,224))
		iamge = np.array(image, dtype = np.float32)
		data.append([image,text])

	pik = open('data.pickle','wb')
	pickle.dump(data,pik)
	pik.close()

#getImages()
def load_data():
	pick = open('data.pickle','rb')
	data = pickle.load(pick)
	pick.close()

	np.random.shuffle(data)

	feature = []
	texts = []
	for img,text in data:
		feature.append(img)
		texts.append(text)
	feature = np.array(feature)
	texts = np.array(texts)
	feature = feature/255.0

	return [feature,texts]

#getImages()
#(feature,text) = load_data()
#print(feature,text)