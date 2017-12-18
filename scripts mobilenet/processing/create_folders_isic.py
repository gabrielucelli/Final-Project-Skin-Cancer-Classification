import pandas as pd
import numpy as np
import os
import shutil

DIR_IMAGES = "../data/isic_images224/"
DIR_CSV = "../data/"
TARGET_DIR = "../images_isic/"
FILENAME_CSV = DIR_CSV + "isic_ground_truth.csv"
FOLDERS_NAME = ['melanoma', 'seborrheic_keratosis', 'nevus']

images = {}
data = pd.read_csv(FILENAME_CSV)

melanoma_images = []
seborrheic_keratosis_images = []
nevus_images = []

for i in range(data['image'].size):

	image_id = data['image'][i]

	if data['keratosis'][i] == 1:
		seborrheic_keratosis_images.append(image_id)
	elif data['melanoma'][i] == 1:
		melanoma_images.append(image_id)
	else:
		nevus_images.append(image_id)

images['melanoma'] = melanoma_images
images['seborrheic_keratosis'] = seborrheic_keratosis_images
images['nevus'] = nevus_images

''' CREATING THE FOLDERS '''
for key in images.keys():
	if not os.path.exists(TARGET_DIR + key):
		os.makedirs(TARGET_DIR + key)

''' COPYING FILES '''
for key, value in images.items():
	for i in value:
		try:
			shutil.copy2(DIR_IMAGES + i + ".jpg", TARGET_DIR + key + "/" + i + ".jpg")
		except Exception as e:
			print("miss: " + key + "/" + i)
		