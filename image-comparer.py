import os, cv2
import numpy as np

for filename in sorted(os.listdir('../Class')):
	avg_difference = []
	class_id = int(filename.replace('.png', ''))

	# if class_id != 1: continue

	base_img = cv2.imread('../Class/{}'.format(filename))
	base_img = cv2.resize(base_img, (64, 64))

	train_imgs = sorted(os.listdir('../train/{}'.format(class_id)))
	
	for trainname in train_imgs:
		target_img = cv2.imread('../train/{}/{}'.format(class_id, trainname))
		target_img = cv2.resize(target_img, (64, 64))

		different_vectors = []
		sum_weight = 0
		for i in range(64):
			for j in range(64):
				weight = 1.0 / (((i - 31.5) ** 2 + (j - 31.5) ** 2) ** (1.0 / 2))
				sum_weight += weight
				diff = sorted([(int(target_img[i][j][k]) - int(base_img[i][j][k])) ** 2 for k in range(3)], reverse=True)
				subweight = [1.0 - 1.0 / 3 * i for i in range(3)]
				different_vectors.append(weight / 2 * sum([diff[i] + subweight[i] for i in range(3)]))
		average = np.average(different_vectors) / sum_weight
		avg_difference.append(average)
	
	avg_difference = np.array(avg_difference)

	tokens = sorted([(avg_difference[i], train_imgs[i]) for i in range(len(train_imgs))])
	max_allowed = min(tokens[0][0] * 2.0, tokens[0][0] + 2.0)
	for diff, trainname in tokens:
		if diff > max_allowed: break
		# print('Difference = ', diff)
		img = cv2.imread('../train/{}/{}'.format(class_id, trainname))
		# cv2.imshow('Image', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		cv2.imwrite('../realtrain/{}/{}'.format(class_id, trainname), img)

	print(class_id, avg_difference)

# print(os.listdir('../train/0'))