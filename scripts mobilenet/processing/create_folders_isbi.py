import pandas as pd
import numpy as np
import os
import shutil

DIR_IMAGES = "../data/isbi_images224/"
DIR_CSV = "../data/"
TARGET_DIR = "../images/"
FILENAME_CSV = DIR_CSV + "ISIC-2017_Training_Part3_GroundTruth.csv"
FOLDERS_NAME = ['melanoma', 'seborrheic_keratosis', 'nevus']
HEADERS = ['image_id', 'melanoma', 'seborrheic_keratosis']

images = {}
data = pd.read_csv(FILENAME_CSV)

melanoma_images = []
seborrheic_keratosis_images = []
nevus_images = []

for i in range(data['image_id'].size):

	image_id = data['image_id'][i]

	if data['melanoma'][i] == 1:
		melanoma_images.append(image_id)
	elif data['seborrheic_keratosis'][i] == 1:
		seborrheic_keratosis_images.append(image_id)
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
		shutil.copy2(DIR_IMAGES + i + ".jpg", TARGET_DIR + key + "/" + i + ".jpg")




	


