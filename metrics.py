from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import metrics

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json

def plot_conf_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()

def print_conf_matrix(y_true, y_pred, label_melanoma, label_others):

	confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

	print('Confusion Matrix: ')
	print(confusion_matrix)

	labels = ["" for x in range(2)]
	labels[label_melanoma] = 'Non-Melanoma'
	labels[label_others] = 'Melanoma'

	if label_melanoma == 1 :
		cm = np.zeros(shape=(2,2))
		cm[0, 0] = confusion_matrix[1, 1]
		cm[0, 1] = confusion_matrix[1, 0]
		cm[1, 0] = confusion_matrix[0, 1]
		cm[1, 1] = confusion_matrix[0, 0]
		plot_conf_matrix (cm, labels, normalize=True)
	else:
		plot_conf_matrix (confusion_matrix, labels, normalize=True)

def print_metrics(y_true, y_pred, label_melanoma, label_others):

	print("\nPrinting metrics\n---------- ")

	accuracy_score = metrics.accuracy_score(y_true, y_pred)
	print('Accuracy: %g' % accuracy_score)

	precision_score = metrics.precision_score(y_true, y_pred)
	print('Precision: %g' % precision_score)

	recall_score =  metrics.recall_score(y_true, y_pred)
	print('Recall: %g' % recall_score)

	roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
	print('ROC AUC: %g' % roc_auc_score)

	f1_score = metrics.f1_score(y_true, y_pred)
	print('F1: %g' % f1_score)

	confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

	sensitivity = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])
	print('Sensitivity:', sensitivity )

	specificity = confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1])
	print('Specificity:', specificity)

	print('---------- \n')

def print_metrics_as_csv(y_true, y_pred, label_melanoma, label_others):

	accuracy_score = metrics.accuracy_score(y_true, y_pred)
	precision_score = metrics.precision_score(y_true, y_pred)
	recall_score =  metrics.recall_score(y_true, y_pred)
	roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
	f1_score = metrics.f1_score(y_true, y_pred)

	confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
	sensitivity = confusion_matrix[0,0] / (confusion_matrix[0,0] + confusion_matrix[0,1])
	specificity = confusion_matrix[1,1] / (confusion_matrix[1,0] + confusion_matrix[1,1])


	print(accuracy_score, ",",
	 precision_score, ",",
	 recall_score, ",",
	 f1_score, ",",
	 roc_auc_score, ",",
	 sensitivity, ",",
	 specificity)

def get_data_from_json(json_file):

	json_file = open(json_file, 'r')
	json_string = json_file.read()
	data = json.loads(json_string)
	json_file.close()

	return data

if __name__ == "__main__":

	data = get_data_from_json(*sys.argv[1:])

	#print_metrics_as_csv(data['y_true'], data['y_pred'], data['label_melanoma'], data['label_outros'])
	print_conf_matrix(data['y_true'], data['y_pred'], data['label_melanoma'], data['label_outros'])
