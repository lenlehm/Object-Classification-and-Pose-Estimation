# Additional package to install: pillow, scipy
# You also need to create a folder in your directory with the name models
import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import sys
from lenet_model import leNet
from utils import *
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier

# TODO: (line 314)
# get feature space of CNN and then turn test image into feature space
# TRANSFORM EVERY DB IMAGE INTO DESCRIPTOR SPACE AS WELL AND THEN COMPARE

iterations = 200
# Change these two paths accordingly!
model_path = 'C:\\Users\\Lenny\\Documents\\Studium_Robotics (M.Sc.)\\Semester 1\\Tracking and Detection in CV\\Project 3 Pose Estimation & Classification\\models'
base_folder = 'C:\\Users\\Lenny\\Documents\\Studium_Robotics (M.Sc.)\\Semester 1\\Tracking and Detection in CV\\Project 3 Pose Estimation & Classification\\dataset'
#base_folder2 = 'C:/Users/Michelle/Desktop/TUM/WS1819_RCI1/W_IN2210-Tracking and Detection in CV/practicals/Project_3/code/dataset'

dataset_folders = ['coarse', 'fine', 'real']
NUM_CLASS = 5
class_folders = ['ape', 'benchvise', 'cam', 'cat', 'duck']

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1) # normalize quaternions
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # same output like given formula: return np.arccos(np.abs(np.dot(v1, v2))) - more stable tho

def batch_generator(train_norm, db_norm, train_poses, db_poses, batch_size = 285):
	triplet = []
	for cnt in range(batch_size):
		closest = 1.0
		idx = 0
		randomClass = np.random.randint(0, train_norm.shape[0])
		randomImg = np.random.randint(0, train_norm.shape[1])
		anchor = train_norm[randomClass][randomImg] # randomly from training set
		anchor_pose = train_poses[randomClass][randomImg]
		for i in range(len(db_norm[1])): # loop over db images - for pusher
			# get closest quaternion angle
			dist = angle_between(anchor_pose, db_poses[randomClass][i])
			if (dist < closest and dist != 0.0): # not identical pose & smallest
				closest = dist
				idx = i
		#print("Closest distance {}".format(closest))
		puller = db_norm[randomClass][idx] # most similiar one quaternion wise
		try: # in case our random variable is 0 -- throws error
			pusher = db_norm[randomClass - 1][idx] # either different pose or different object
		except: # then increase the class instead
			pusher = db_norm[randomClass + 1][idx]
		triplet.append((anchor, puller, pusher))
	# shall we only return the x values of anchor, puller and pusher??
	#triplet = (anchor, puller, pusher) # returning one image for each triplet entry
	return triplet

# def normalize(arr): # Currently not in use -- nromalization in line 207
#	normalization = (data - data.mean()) / data.std()
#     """
#     Linear normalization
#     http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
#     """
#     arr = arr.astype('float')
#     # Do not touch the alpha channel
#     for i in range(3):
#         minval = arr[...,i].min()
#         maxval = arr[...,i].max()
#         if minval != maxval:
#             arr[...,i] -= minval
#             arr[...,i] *= (255.0/(maxval-minval))
#     return arr

# ## ############################## RUN THIS CODE ONCE - WILL STORE EVERYTHING SEPERATELY #####################################
#
# ##   get all folders
# coarse_folders = []
# fine_folders = []
# real_folders = []
#
# dataDir = os.path.join(base_folder, dataset_folders[0])
# for folder in os.listdir(dataDir):
#     coarse_folders.append(os.path.join(dataDir, folder))
#
# dataDir = os.path.join(base_folder, dataset_folders[1])
# for folder in os.listdir(dataDir):
#     fine_folders.append(os.path.join(dataDir, folder))
#
# dataDir = os.path.join(base_folder, dataset_folders[2])
# for folder in os.listdir(dataDir):
#     if not (folder.endswith('.txt')):
#         real_folders.append(os.path.join(dataDir, folder))
#
# #   read in img paths
# #   images are named according to dataset folders e.g. real0.png
# #   list separated by class
# #   list = [NUM_CLASS]['FULL_PATH', img_index]
# #   e.g. list[0] = ['blablabla\ape\\real1.png', 1]
# coarse_imgs_list = []
# fine_imgs_list = []
# real_imgs_list = []
#
# for class_f in coarse_folders:
#     filelist = []
#     for file in os.listdir(class_f):
#         if file.endswith('.png'):
#             filepath = [os.path.join(class_f, file) , [int(c) for c in re.split('([0-9]+)', file) if c.isdigit()][0]]
#             filelist.append(filepath)
#     if (len(filelist) != 0):
#         coarse_imgs_list.append(filelist)
#
# for class_f in fine_folders:
#     filelist = []
#     for file in os.listdir(class_f):
#         if file.endswith('.png'):
#             filepath = [os.path.join(class_f, file) , [int(c) for c in re.split('([0-9]+)', file) if c.isdigit()][0]]
#             filelist.append(filepath)
#     if (len(filelist) != 0):
#         fine_imgs_list.append(filelist)
#
# for class_f in real_folders:
#     filelist = []
#     for file in os.listdir(class_f):
#         if file.endswith('.png'):
#             filepath = [os.path.join(class_f, file) , [int(c) for c in re.split('([0-9]+)', file) if c.isdigit()][0]]
#             filelist.append(filepath)
#     if (len(filelist) != 0):
#         real_imgs_list.append(filelist)
#
# #   Sort all file paths
# for i in range(0,5):
#     coarse_imgs_list[i].sort(key=lambda x : x[1])
#     fine_imgs_list[i].sort(key=lambda x : x[1])
#     real_imgs_list[i].sort(key=lambda x : x[1])
#
# #   get all poses
# #   e.g. [-0.28184579021235323, -0.6032481990846498, 0.6534595646367771, -0.3600627142949052]
# coarse_poses_list = []
# for class_f in coarse_folders:
#     pose_txt = class_f + '/poses.txt'
#     coarse_poses_list.append(getposes(pose_txt))
#
# real_poses_list = []
# for class_f in real_folders:
#     pose_txt = class_f + '/poses.txt'
#     real_poses_list.append(getposes(pose_txt))
#
# fine_poses_list = []
# for class_f in fine_folders:
#     pose_txt = class_f + '/poses.txt'
#     fine_poses_list.append(getposes(pose_txt))
#
# #   get training split
# train_txt = os.path.join(base_folder, (dataset_folders[2] + '/training_split.txt'))
# delim = ','
# train_data_list = readtxt(train_txt, delim)
#
# real_train_split_imgs_list = []
# real_train_split_pose_list = []
# for i in range(0, NUM_CLASS):
#     filelist = []
#     tmp_pose_list = []
#     for img_path in real_imgs_list[i]:
#         if (any((img_path[1] == s) for s in train_data_list)):
#             filelist.append(img_path)
#             tmp_pose_list.append(real_poses_list[i][img_path[1]])
#     real_train_split_imgs_list.append(filelist)
#     real_train_split_pose_list.append(tmp_pose_list)
#
# # Full train images and poses
# train_imgs_list = []
# train_poses_list = []
# for i in range(0, NUM_CLASS):
#     train_imgs_list.append(fine_imgs_list[i] + real_train_split_imgs_list[i])
#     train_poses_list.append(fine_poses_list[i] + real_train_split_pose_list[i])
#
# # rest to test set
# # get imgs_list and corresponding poses
# real_test_split_imgs_list = []
# real_test_split_pose_list = []
# for i in range(0, NUM_CLASS):
#     filelist = []
#     tmp_pose_list = []
#     for img_path in real_imgs_list[i]:
#         if (any((img_path[1] == s) for s in train_data_list)):
#             pass
#         else:
#             filelist.append(img_path)
#             tmp_pose_list.append(real_poses_list[i][img_path[1]])
#     real_test_split_imgs_list.append(filelist)
#     real_test_split_pose_list.append(tmp_pose_list)
#
# # 707
# # print(len(real_test_split_imgs_list[0]))
# # print(len(real_test_split_pose_list[0]))
#
# # normalized dataset
# train_imgs_normalized = []
# test_imgs_normalized = []
# db_imgs_normalized = []
#
# #  load imgs to list
# train_imgs = []
# test_imgs = []
# db_imgs = []
#
# print("Loading images takes a while...")
# for i in range(0, NUM_CLASS):
#     train_imgs.append([np.array(Image.open(fname[0])) for fname in train_imgs_list[i]])
#     test_imgs.append([np.array(Image.open(fname[0])) for fname in real_test_split_imgs_list[i]])
#     db_imgs.append([np.array(Image.open(fname[0])) for fname in coarse_imgs_list[i]])
#
#     print("\n" + "Done class: {}".format(class_folders[i]))
#     print("num_train_imgs: {}".format(len(train_imgs_list[i])))
#     print("num_test_imgs: {}".format(len(real_test_split_imgs_list[i])))
#     print("num_coarse_imgs: {}".format(len(coarse_imgs_list[i])))
#
#     # perform zero mean normalization and unit variance
#     train_imgs_normalized.append([ (np.array(Image.open(fname[0])) - np.array(Image.open(fname[0])).mean() ) / np.array(Image.open(fname[0])).std() for fname in train_imgs_list[i]])
#     test_imgs_normalized.append([ (np.array(Image.open(fname[0])) - np.array(Image.open(fname[0])).mean() ) / np.array(Image.open(fname[0])).std() for fname in real_test_split_imgs_list[i]])
#     db_imgs_normalized.append([ (np.array(Image.open(fname[0])) - np.array(Image.open(fname[0])).mean() ) / np.array(Image.open(fname[0])).std() for fname in coarse_imgs_list[i]])
#
#     print("num_train_imgs: {}".format(len(train_imgs_normalized[i])))
#     print("num_test_imgs: {}".format(len(test_imgs_normalized[i])))
#     print("num_coarse_imgs: {}".format(len(db_imgs_normalized[i])))

# # save all data to npy files so it's faster
# np.save(os.path.join(base_folder, "train_imgs"), train_imgs)
# np.save(os.path.join(base_folder, "test_imgs"), test_imgs)
# np.save(os.path.join(base_folder, "db_imgs"), db_imgs)

# np.save(os.path.join(base_folder, "train_imgs_normalized"), train_imgs_normalized)
# np.save(os.path.join(base_folder, "test_imgs_normalized"), test_imgs_normalized)
# np.save(os.path.join(base_folder, "db_imgs_normalized"), db_imgs_normalized)

# np.save(os.path.join(base_folder, "train_imgs_list"), train_imgs_list)
# np.save(os.path.join(base_folder, "test_imgs_list"), real_test_split_imgs_list)
# np.save(os.path.join(base_folder, "db_imgs_list"), coarse_imgs_list)

# np.save(os.path.join(base_folder, "train_poses"), train_poses_list)
# np.save(os.path.join(base_folder, "test_poses"), real_test_split_imgs_list)
# np.save(os.path.join(base_folder, "db_poses"), coarse_poses_list)

## ---------------------------------- CONTINUE FROM BELOW THIS COMMENT ----------------------

########################  load npy files ##############################
# ALL FORMAT: list = [NUM_CLASS][CONTENTS ACCORDING TO DATA]

# e.g. train_imgs[0][0] = [image_array(64x64)]
print("Loading images...")
train_imgs = np.load(os.path.join(base_folder, "train_imgs.npy"))
test_imgs = np.load(os.path.join(base_folder, "test_imgs.npy"))
db_imgs = np.load(os.path.join(base_folder, "db_imgs.npy"))
print("Done loading.")

print("Loading normalized images...")
train_imgs_normalized = np.load(os.path.join(base_folder, "train_imgs_normalized.npy"))
test_imgs_normalized = np.load(os.path.join(base_folder, "test_imgs_normalized.npy"))
db_imgs_normalized = np.load(os.path.join(base_folder, "db_imgs_normalized.npy"))
print("Done loading.")

# fig = plt.figure()
# plt.imshow(Image.fromarray(np.uint8(train_imgs_normalized[3][200] * 255))) // multiply with 255 to retrieve image
# plt.show()

# e.g. train_poses_list[0][0] = [-0.28184579021235323, -0.6032481990846498, 0.6534595646367771, -0.3600627142949052]
print("Loading poses...")
train_poses_list = np.load(os.path.join(base_folder, "train_poses.npy"))
test_poses_list = np.load(os.path.join(base_folder, "test_poses.npy"))
db_poses_list = np.load(os.path.join(base_folder, "db_poses.npy"))
print("Done loading.")

# list = [NUM_CLASS]['FULL_PATH', img_index]
# e.g. list[0] = ['blablabla\ape\\real1.png', 1]
print("Loading lists...")
train_imgs_list = np.load(os.path.join(base_folder, "train_imgs_list.npy"))
test_imgs_list = np.load(os.path.join(base_folder, "test_imgs_list.npy"))
db_imgs_list = np.load(os.path.join(base_folder, "db_imgs_list.npy"))
print("Done loading.")
print(" ------------------- LET'S GET IT STARTED -------------")

# test load - train_imgs_list[class][pictureNumber][filename, value]
#print(train_imgs_list[3][0]) # prints the location
#print(train_poses_list[3][0]) # [-0.21679692 -0.58559097  0.76315701 -0.16635413]

#  show image
# fig = plt.figure()
# plt.imshow(Image.fromarray(np.uint8(db_imgs[3][200])))
# plt.show()

triplets = batch_generator(train_imgs_normalized, db_imgs_normalized, train_poses_list, db_poses_list)
# print(np.shape(triplets[0]))
# print(triplets[0][0].dtype) # float64

triplets_f32 = [np.float32(t) for t in triplets]
labels = None
input_data = iter(triplets_f32[:])

# cnn = tf.estimator.Estimator(model_fn=lenet_model.leNet)
# Create the Estimator
total_loss = []
ang_difference = []
histo = []
cnn = tf.estimator.Estimator(model_fn=leNet, model_dir=model_path)
for i in range(iterations):
	# don't know whether I get this as a return
	mode, loss, train_op = cnn.train(input_fn=lambda: (next(input_data), labels), steps=10)
	if i % 10: # store loss every tenth iteration
		total_loss.append(loss)
		print("Iteration: {}} || Triplet Loss: {}".format(i, loss))
	#cnn.train(input_fn=lambda: (next(input_data), labels), steps=10)

#https://github.com/jireh-father/tensorflow-cnn-visualization  -- CNN Visualization

# loop over every image and every class in our test dataset
for clas in range(NUM_CLASS):
	for img in range(len(test_imgs_normalized)):
		# feed images into CNN and obtain features
		features = cnn.predict(test_imgs_normalized[clas][img])
		knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
		# use the db_imgs_features - need to be transformed first as well
		knn.fit(features, db_imgs_normalized) # match it with our Database Images S_db
		pred = knn.predict(features) # predict the descriptors which we obtained from CNN earlier.
		## check for the same class
		if pred == clas: # correct prediction
			## calculate the angular difference of it
			ang_difference.append(angle_between(db_poses_list[clas][img], test_poses_list[clas][img]))

		if iteration % 1000: # each 1000th iteration print histogram 
			degree_10 	= sum(i < 10 for i in ang_difference) # count values in our angles which are smaller than 10
			degree_20 	= sum(i < 20 for i in ang_difference)
			degree_40 	= sum(i < 40 for i in ang_difference)
			degree_180 	= sum(i < 180 for i in ang_difference)
			# histogram values should be stored in %
			degree_10 	/= len(ang_difference)
			degree_20 	/= len(ang_difference)
			degree_40 	/= len(ang_difference)
			degree_180 	/= len(ang_difference)

			histo.append(degree_10, degree_20, degree_40, degree_180)
			print("{} degrees don't fall into any of the bins -- error! ".format( len(ang_difference) - (degree_10+degree_180+degree_20+degree_40) ))
			#plt.bar(uniq_Vals, histo)
			#plt.show()