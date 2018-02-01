import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

from skimage.io import imread, imsave
from skimage import transform
from skimage.filters import threshold_otsu, threshold_local
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

import os

class DocumentsClassificator:
	'''Class for documents classification which should pe presented in "data" folder.'''

	def __init__(self):
		self.PATH_TO_DATA = 'data/'
		self.PATH_TO_TEPLATES = self.PATH_TO_DATA + 'templates/'
		self.NUM_CLASSES = len(open(self.PATH_TO_TEPLATES + 'types.txt').readlines())
		self.TARGET_TITLES = []
		self.TARGET_IDS = {}
		self.IMG_ROWS, self.IMG_COLS, self.DEPTH = 150, 150, 1

		self.EPOCHS = 300
		self.INPUT_SHAPE = ()
		self.classifier = None

		self._prepare_classifier()

	def document_class(self, path_to_image):
		img = imread(path_to_image)
		img = self._prepare_image(img)
		doc_class = self.classifier.predict(img)
		return doc_class[0].argmax()


	def _prepare_image(self, img):
		prepared = transform.resize(img, (150, 150))
		prepared -= 0.5
		prepared = np.array([prepared])

		if K.image_data_format() == 'channels_first':
			prepared = prepared.reshape(prepared.shape[0], self.DEPTH, self.IMG_ROWS, self.IMG_COLS)
		else:
			prepared = prepared.reshape(prepared.shape[0], self.IMG_ROWS, self.IMG_COLS, self.DEPTH)

		return prepared

	def _prepare_classifier(self):
		if os.path.isfile(self.PATH_TO_DATA + 'internals/model.h5'):
			self.classifier = load_model(self.PATH_TO_DATA + 'internals/model.h5')
		else:
			self._train_classifier()


	def _train_classifier(self):
		images, targets = self._load_train_data()

		model = Sequential()

		model.add(Conv2D(filters = 20, kernel_size = (7, 7), 
		                 activation ='relu', input_shape = self.INPUT_SHAPE))
		model.add(MaxPool2D(pool_size=(4, 4)))
		model.add(Conv2D(filters = 50, kernel_size = (5, 5), activation ='relu'))
		model.add(MaxPool2D(pool_size=(4, 4)))
		model.add(Flatten())
		model.add(Dense(1000, activation = "relu"))
		model.add(Dense(1000, activation = "relu"))
		model.add(Dropout(0.5))
		model.add(Dense(self.NUM_CLASSES, activation = "softmax"))

		model.compile(loss=keras.losses.categorical_crossentropy,
		            optimizer='sgd',
		            metrics=['accuracy'])
		history = model.fit(images, targets, epochs = self.EPOCHS)


		model.save(self.PATH_TO_DATA + 'internals/model.h5')
		self.classifier = model

	def _load_train_data(self):
	    images = np.array([])
	    targets = []
	    path_to_images = self.PATH_TO_TEPLATES + 'templates_images/'
	    samples_number = self._file_number(path_to_images)
	    
	    for i in range(samples_number):
	        path_to_sample = path_to_images + str(i) + '.jpg'
	            
	        new_img = transform.resize(imread(path_to_sample), (150, 150))
	            
	        if len(images) == 0:
	            images = np.array([new_img])
	        else:
	            images = np.append(images, [new_img], axis=0)
	    
	    fl = open(self.PATH_TO_TEPLATES + 'types.txt', 'r')
	    for line in fl:
	        targets.append(int(line))
	    fl.close()

	    if K.image_data_format() == 'channels_first':
	        images = images.reshape(images.shape[0], self.DEPTH, self.IMG_ROWS, self.IMG_COLS)
	        self.INPUT_SHAPE = (self.DEPTH, self.IMG_ROWS, self.IMG_COLS)
	    else:
	        images = images.reshape(images.shape[0], self.IMG_ROWS, self.IMG_COLS, self.DEPTH)
	        self.INPUT_SHAPE = (self.IMG_ROWS, self.IMG_COLS, self.DEPTH)
	    
	    images = images.astype('float32')
	    images -= 0.5
	    
	    targets = keras.utils.to_categorical(targets, self.NUM_CLASSES)
	    return images, targets

	def _file_number(self, path):
		return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])


if __name__ == '__main__':
	path_to_image = '/home/fergy/Projects/nsu-winter/nsu/7.jpg'
	classifier = DocumentsClassificator()
	res = classifier.document_class(path_to_image)
	print(res)