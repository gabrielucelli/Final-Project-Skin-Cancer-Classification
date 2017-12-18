from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
K.set_image_data_format('channels_first')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 1


def get_generators():

	train_data_dir = 'images/train'
	validation_data_dir = 'images/validation'

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
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


def cnn_model():

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(3, IMG_SIZE, IMG_SIZE)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	return model
	

model = cnn_model()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_generator, validation_generator = get_generators()

# outros - 3315
# melanoma - 1071

# indices = train_generator.class_indices
# class_weight = {indices['outros']:1, indices['melanoma']:3.094304388}

train_samples = 2033
validation_samples = 200

model.fit_generator(
        train_generator,
        steps_per_epoch = train_samples // BATCH_SIZE,
		epochs=EPOCHS,
		validation_data=validation_generator,
		validation_steps = validation_samples // BATCH_SIZE)

model.save_weights('first_try.h5')