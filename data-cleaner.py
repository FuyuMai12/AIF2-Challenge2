import os, cv2
import numpy as np

true_samples = []
true_classes = []
true_weights = []
train_samples = []
train_files = []
train_classes = []

for filename in sorted(os.listdir('../Class')):
	avg_difference = []
	class_id = int(filename.replace('.png', ''))

	base_img = cv2.imread('../Class/{}'.format(filename))
	base_img = (cv2.resize(base_img, (64, 64))).reshape(12288)
	for _ in range(25):
		true_samples.append(base_img)
		true_classes.append(class_id)
	# true_weights.append(25)
	
	for trainname in os.listdir('../realtrain/{}'.format(class_id)):
		target_img = cv2.imread('../realtrain/{}/{}'.format(class_id, trainname))
		target_img = (cv2.resize(target_img, (64, 64))).reshape(12288)

		true_samples.append(target_img)
		true_classes.append(class_id)
		# true_weights.append(1)
	
	for trainname in os.listdir('../train/{}'.format(class_id)):
		target_img = cv2.imread('../train/{}/{}'.format(class_id, trainname))
		target_img = (cv2.resize(target_img, (64, 64))).reshape(12288)

		train_samples.append(target_img)
		train_files.append(trainname)
		train_classes.append(class_id)

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
models = ['knn']
classifiers = [
	KNeighborsClassifier(3)
]

correct_match = [0 for _ in range(len(train_files))]

for index in range(len(models)):
	print('Training', models[index], '...')

	classifiers[index].fit(true_samples, true_classes)

	print('Predicting by', models[index], '...')

	predicted_classes = classifiers[index].predict(train_samples)
	for i in range(len(train_files)):
		correct_match[i] += (predicted_classes[i] == train_classes[i])

print('Writing result...')
for i in range(len(train_files)):
	if correct_match[i] < len(models): continue
	class_id, trainname = train_classes[i], train_files[i]
	img = cv2.imread('../train/{}/{}'.format(class_id, trainname))
	cv2.imwrite('../cleanedtrain/{}/{}'.format(class_id, trainname), img)