from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
from keras.models import model_from_json

import pickle
import os
import tensorflow as tf

K.set_image_data_format('channels_first')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 50

FOLDER_RESULTS = 'model1/'

def get_generators():

	train_data_dir = 'images/train'
	validation_data_dir = 'images/validation'

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.1,
		zoom_range=0.1,
		width_shift_range=0.05,
        height_shift_range=0.05,
        rotation_range=5,
		horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	validation_datagen = ImageDataGenerator(rescale=1./255)

	# automagically retrieve images and their classes for train and validation sets
	train_generator = train_datagen.flow_from_directory(
			train_data_dir,
			target_size=(IMG_SIZE, IMG_SIZE),
			batch_size=BATCH_SIZE,
			class_mode='binary')

	validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary')

	return train_generator, validation_generator

def get_model1():

	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(3, IMG_SIZE, IMG_SIZE)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	return model


def get_model2():

	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(3, IMG_SIZE, IMG_SIZE)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dense(512))
	model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	return model

def get_model3():

	model = Sequential()

	model.add(Conv2D(32, (3, 3), input_shape=(3, IMG_SIZE, IMG_SIZE)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(256))
	model.add(Activation('relu'))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	return model

def train_cnn():
	
	train_generator, validation_generator = get_generators()

	train_samples = 2033
	validation_samples = 200

	for i in range(15):

		model = get_model1()

		model.compile(loss='binary_crossentropy',
	              optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
	              metrics=['accuracy'])

		# serialize model to JSON
		model_json = model.to_json()

		with open(FOLDER_RESULTS + "model.json", "w") as json_file:
		    json_file.write(model_json)

		filepath=FOLDER_RESULTS + "weights-" + str(i) + "-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		callbacks_list = [checkpoint]

		history = model.fit_generator(
			train_generator,
			steps_per_epoch = train_samples // BATCH_SIZE,
			epochs = EPOCHS,
			validation_data= validation_generator,
			validation_steps = validation_samples // BATCH_SIZE,
			callbacks=callbacks_list)

		# save the weights
		model.save_weights(FOLDER_RESULTS + "model_weights" + str(i) +".h5")		

		# save the history
		with open(FOLDER_RESULTS + "train_hist" + str(i), 'wb') as file_pi:
			pickle.dump(history.history, file_pi)

# main
#print(get_model2().count_params())
train_cnn()


