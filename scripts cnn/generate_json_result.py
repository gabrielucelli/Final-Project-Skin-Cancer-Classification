from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
from keras import backend as K

import pickle
import sys
import os
import json
import tensorflow as tf

K.set_image_data_format('channels_first')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

IMG_SIZE = 128
BATCH_SIZE = 64

folder = "results/modelo1/"

def get_validation_generators():

	validation_data_dir = 'images/validation'
	validation_datagen = ImageDataGenerator(rescale=1./255)
	validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False)

	return validation_generator

def load_model_from_json():

	# load json and create model
	json_file = open(folder + "model.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	return loaded_model

if __name__ == "__main__":

	#model = load_model()
	model = load_model(sys.argv[1])

	validation_generator = get_validation_generators()
	predict = model.predict_generator(validation_generator, len(validation_generator.filenames))

	indices = validation_generator.class_indices

	label_melanoma = indices['melanoma']
	label_outros = indices['outros']

	y_true = []

	for i in range(len(validation_generator.filenames)):
		if "melanoma" in validation_generator.filenames[i]:
			y_true.append(label_melanoma)
		else:
			y_true.append(label_outros)

	y_pred = []

	for i in range(predict.size):
		if (predict[i] >= 0.5):
			y_pred.append(1)
		elif (predict[i] < 0.5):
			y_pred.append(0)

	data = {}

	data['label_outros'] = label_outros
	data['label_melanoma'] = label_melanoma
	data['y_true'] = y_true
	data['y_pred'] = y_pred

	data_json = json.dumps(data)

	with open(folder + sys.argv[2], "w") as json_file:
	    json_file.write(data_json)