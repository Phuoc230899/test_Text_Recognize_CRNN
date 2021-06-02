import os
import cv2
import numpy as np
from PIL import Image
import pickle
from pathlib import Path
import string
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
import keras.backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.activations import relu, sigmoid, softmax
from keras.utils import to_categorical

imgpath = './Data/img/'
txtpath = './Data/txt/'
vi = 'áàảãạắằẳẵặấầẩẫậớờởỡợếềểễệéèẻẽẹốồổỗộđúùủũụọóỏõòứừửữựíìỉĩịÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬỚỜỞỠỢẾỀỂỄỆÉÈẺẼẸỐỒỔỖỘĐÚÙỦŨỤỌÓỎÕÒỨỪỬỮỰÍÌỈĨỊưâƯÂăĂơƠêÊôÔýỳỷỹỵÝỲỶỸỴ'
char_list = string.ascii_letters + string.digits + string.punctuation + string.whitespace + vi
max_label_len = 0
# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

#lists for validation dataset
valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst



def getImages():
	i = 1
	filetext = list(Path(txtpath).rglob('*.[txt]*'))
	names = [str(path.name).split('.')[0] for path in filetext]
	for name in names:
		image_path = imgpath+name+'.png'
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		with open(txtpath+name+'.txt','r',encoding = 'utf8') as f:
			txt = f.read()
			
		if i%10 == 0:
			valid_orig_txt.append(txt)   
			valid_label_length.append(len(txt))
			valid_input_length.append(31)
			valid_img.append(image)
			valid_txt.append(encode_to_labels(txt))
		else:
			orig_txt.append(txt)
			train_label_length.append(len(txt))
			train_input_length.append(31)
			training_img.append(image)
			training_txt.append(encode_to_labels(txt))

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))

getImages()



def decode_to_labels(dig_lst):
	txt = []
	output = ""
	for i in dig_lst:
		txt.append(char_list[i])
	for text in txt :
		output = output+text
	return output


inputs = Input(shape=(32,128,1))
 
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)
 
act_model = Model(inputs, outputs)


labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
 
 
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
  
 
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length]) 

#model to be used at training time
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)


model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
 
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)

batch_size = 256
epochs = 10
batch_size = 256
epochs = 10
model.fit(x=[training_img, train_padded_txt, train_input_length, 
	train_label_length], y=np.zeros(len(training_img)), 
	batch_size=batch_size, epochs = epochs,
	 validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
	  [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)