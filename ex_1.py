import numpy as np
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import sys
from lenet_model import leNet
from utils import *
import tensorflow as tf
import cv2 # --------------------- need to have OpenCV installed for the descriptor matcher
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

#base_folder = 'C:/Users/Michelle/Desktop/TUM/WS1819_RCI1/W_IN2210-Tracking and Detection in CV/practicals/Project_3/code/dataset'
#LOGDIR = 'C:/Users/Michelle/Desktop/TUM/WS1819_RCI1/W_IN2210-Tracking and Detection in CV/practicals/Project_3/code/models'

LOGDIR = 'C:\\Users\\Lenny\\Documents\\Studium_Robotics (M.Sc.)\\Semester 1\\Tracking and Detection in CV\\Project 3 Pose Estimation & Classification\\models'
base_folder = 'C:\\Users\\Lenny\\Documents\\Studium_Robotics (M.Sc.)\\Semester 1\\Tracking and Detection in CV\\Project 3 Pose Estimation & Classification\\dataset'

batchSize = 1 # take each single image in at once
numEpochs = 3535 // batchSize # for batch repeat in line 80

trainBatch = 285
LogIter = 30
NUM_EPOCHS = int((LogIter * 1000)/trainBatch)  + ((LogIter * 1000) % trainBatch > 0)
LRATE = 1e-5

dataset_folders = ['coarse', 'fine', 'real']
NUM_CLASS = 5
class_folders = ['ape', 'benchvise', 'cam', 'cat', 'duck']

## -------------- GET THE TRIPLETS ALONG WITH THE QUATERNION ANGLE DIFFERENCES ----------------------
def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1) # normalize vectors first
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) # calculate angle among them
    # same output like given formula: return np.arccos(np.abs(np.dot(v1, v2)))

def batch_generator(train_norm, db_norm, train_poses, db_poses, batch_size = 285, plot=False):
    triplet = []
    maxdiff = 0.
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
            diff = 2*np.arccos(np.abs(np.dot(anchor_pose, db_poses[randomClass][i])))
            if (maxdiff<diff):
                maxdiff = diff
        #print("Closest distance {}".format(closest))
        puller = db_norm[randomClass][idx] # most similiar one quaternion wise
        # All pushers are different class
        pusher = db_norm[randomClass - 1][idx] # either different pose or different object
        triplet.append((anchor, puller, pusher))

        ## Plot the Anchor, Puller pusher if wanted
        if plot:
            fig = plt.figure()
            for i in range(3):
                fig.add_subplot(1, 3, i + 1)
                img = plt.imread(triplet[i])
                plt.imshow(img)
            plt.show()
    #triplet = (anchor, puller, pusher) # returning one image for each triplet entry
    print("Maximum diff:" + str(diff))
    # input()
    return (triplet, diff+0.01)
# -------------------------------------------------------------------------------

# --------------------------- LOAD DATASETS FOR TENSORFLOW ----------------------
# LOAD TRAIN DATASET
data_dataset_train = tf.placeholder(tf.float32, [None, 3, 64, 64, 3])
# dataset_train = dataset_train.repeat(2)
dataset = tf.data.Dataset.from_tensor_slices(data_dataset_train)
# # Shuffle, repeat, and batch the examples.
# dataset = dataset.shuffle(1000).repeat().batch(batch_size)
batched_dataset_train = dataset.repeat().shuffle(100).batch(1)
batched_dataset_train = batched_dataset_train.prefetch(1)

iterator_train = batched_dataset_train.make_initializable_iterator()
next_element_train = iterator_train.get_next()

# LOAD TEST DATASET
test_db = tf.placeholder(tf.float32, [None, 64, 64, 3])
dataset_test = tf.data.Dataset.from_tensor_slices(test_db)
batched_dataset_testing = dataset_test.batch(batchSize).repeat(numEpochs)
batched_dataset_testing = batched_dataset_testing.prefetch(1)
iterator_test = batched_dataset_testing.make_initializable_iterator()
next_element_test = iterator_test.get_next()

# LOAD DATABASE
data_db = tf.placeholder(tf.float32, [None, 64, 64, 3])
dataset_db = tf.data.Dataset.from_tensor_slices(data_db)
batched_dataset_db = dataset_db.batch(batchSize).repeat(numEpochs)
batched_dataset_db = batched_dataset_db.prefetch(1)
iterator_db = batched_dataset_db.make_initializable_iterator()
next_element_db = iterator_db.get_next()
# ------------------------------------------------------------------------

# ------------ SET UP THE CNN NETWORK ---------------------------------------
inputs_ = tf.placeholder(tf.float32, [None, 64, 64, 3], name='inputs')
m = tf.placeholder(tf.float32, shape=(), name="margin")

with tf.name_scope("LeNet"):

    # Convolutional Layer #1
    end_point = 'conv1_57x57x16'
    net = tf.layers.conv2d(
      inputs=inputs_,
      filters=16,
      kernel_size=[8, 8],
      activation=tf.nn.relu,
      name=end_point)

    # Pooling Layer #1
    end_point = 'pool1_28x28x16'
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name=end_point)

    # Convolutional Layer #2 and Pooling Layer #2
    end_point = 'conv2_24x24x7'
    net = tf.layers.conv2d(
      inputs=net,
      filters=7,
      kernel_size=[5, 5],
      activation=tf.nn.relu, name=end_point)

    # Pooling Layer #2
    end_point = 'poo2_12x12x7'
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2, name=end_point)

    # Dense Layer
    net = tf.reshape(net, [-1, 1008])
    end_point = 'dense_256'
    net = tf.layers.dense(inputs=net, units=256, activation=None, name=end_point)

    # Logits Layer
    end_point = 'logits_16'
    logits = tf.layers.dense(inputs=net, units=16, activation=None, name=end_point)

# Calculate Loss (for both TRAIN and EVAL modes)
batch_size = tf.shape(inputs_)[0]

diff_pos = logits[0:batch_size:3] - logits[1:batch_size:3]
diff_neg = logits[0:batch_size:3] - logits[2:batch_size:3]
l2_loss_diff_pos = tf.nn.l2_loss(diff_pos) * 2
l2_loss_diff_neg = tf.nn.l2_loss(diff_neg) * 2

# m = 0.305 # 0.01, changed to max angular difference between anchor and puller
loss_triplets = tf.reduce_sum(tf.maximum(0., (1.-(l2_loss_diff_neg/(l2_loss_diff_pos+m)))))

loss_pairs = tf.reduce_sum(l2_loss_diff_pos)

loss = loss_triplets + loss_pairs

opt = tf.train.AdamOptimizer(learning_rate=LRATE).minimize(loss=loss, global_step=tf.train.get_global_step())

with tf.name_scope("variables"):
    # Variable to keep track of how many times the graph has been run
    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="global_step")

with tf.name_scope("update"):
    # Increments the above `global_step` Variable, should be run whenever the graph is run
    increment_step = global_step.assign_add(1)

# Creates summaries for output node
with tf.name_scope("summaries"):
    tf.summary.image(name="input_img", tensor=inputs_)
    tf.summary.scalar(family='Training loss', tensor=loss, name="total_loss")
    tf.summary.scalar(family='Training loss', tensor=loss_triplets, name="loss_triplets")
    tf.summary.scalar(family='Training loss', tensor=loss_pairs, name="loss_pairs")

with tf.name_scope("global_ops"):
    # Initialization Op
    init_op = tf.global_variables_initializer()
    # Merge all summaries into one Operation
    merged_summaries = tf.summary.merge_all()
# --------------------------------------------------------------- DONE WITH THE CNN PART ----------------

########################  load npy files ##############################
# ALL FORMAT: list = [NUM_CLASS][CONTENTS ACCORDING TO DATA]

# # e.g. train_imgs[0][0] = [image_array(64x64)]
print("Loading images...")
train_imgs = np.load(os.path.join(base_folder, "train_imgs.npy"))
test_imgs_ori = np.load(os.path.join(base_folder, "test_imgs.npy"))
db_imgs_ori = np.load(os.path.join(base_folder, "db_imgs.npy"))
print("Done loading.")
# fig = plt.figure()
# plt.imshow(Image.fromarray(np.uint8(train_imgs_normalized[3][200] * 255))) // multiply with 255 to retrieve image
# plt.show()

# list = [NUM_CLASS]['FULL_PATH', img_index]
# e.g. list[0] = ['blablabla\ape\\real1.png', 1]
# print("Loading lists...")
# train_imgs_list = np.load(os.path.join(base_folder, "train_imgs_list.npy"))
# test_imgs_list = np.load(os.path.join(base_folder, "test_imgs_list.npy"))
# db_imgs_list = np.load(os.path.join(base_folder, "db_imgs_list.npy"))
# print("Done loading.")

# e.g. train_poses_list[0][0] = [-0.28184579021235323, -0.6032481990846498, 0.6534595646367771, -0.3600627142949052]
print("Loading poses...")
train_poses_list = np.load(os.path.join(base_folder, "train_poses.npy"))
test_poses_list = np.load(os.path.join(base_folder, "test_poses.npy"))
db_poses_list = np.load(os.path.join(base_folder, "db_poses.npy"))
print("Done loading.")

print("Loading normalized images...")
train_imgs_normalized = np.load(os.path.join(base_folder, "train_imgs_normalized.npy"))
test_imgs_normalized = np.load(os.path.join(base_folder, "test_imgs_normalized.npy"))
db_imgs_normalized = np.load(os.path.join(base_folder, "db_imgs_normalized.npy"))
print("Done loading.")

print("Loading database descriptors...")
database_desc = np.load(os.path.join(base_folder, "database_descriptors.npy"))
print("Done loading.")

print(" ------------------- LET'S GET IT STARTED -------------")

# test load - train_imgs_list[class][pictureNumber][filename, value]
#print(train_imgs_list[3][0]) # prints the location
#print(train_poses_list[3][0]) # [-0.21679692 -0.58559097  0.76315701 -0.16635413]
##  show image
# fig = plt.figure()
# plt.imshow(Image.fromarray(np.uint8(db_imgs[3][200])))
# plt.show()

# ------------------------------ GET THE INPUT DATA FOR THE NETWORK -------------------
triplets, mar = batch_generator(train_imgs_normalized, db_imgs_normalized, train_poses_list, db_poses_list)
# print(np.shape(triplets)) # 285
# print(triplets[0][0].dtype) # float64

triplets_f32 = [np.float32(t) for t in triplets]

# DB and Test images all classes stacked together - get the right shape for feeding it into network
db_imgs = db_imgs_normalized[0]
test_imgs = test_imgs_normalized[0]
for i in range(1,NUM_CLASS):
    db_imgs = np.vstack((db_imgs, db_imgs_normalized[i]))
    test_imgs = np.vstack((test_imgs, test_imgs_normalized[i]))

trueClass = [] # get the ground truth labels
for truClass in range(NUM_CLASS):
    for truth in range(test_imgs_normalized.shape[1]):
        trueClass.append(truClass)

#print("Database shape {}".format(db_imgs.shape))
#print("Test DB shape {}; and Database Descriptors {}".format(test_imgs.shape, database_desc.shape)) # (3535, 64, 64, 3) and (1335, 16)
#print("DB Poses List: {}".format(db_poses_list.shape)) # (5, 267, 4)

#print(database_desc[0], database_desc[1], database_desc[-1])
# print(np.shape(db_imgs)) # (1335, 64, 64, 3)
# print(np.shape(db_imgs_normalized)) # (5, 267, 64, 64, 3)
# print(np.shape(triplets_f32)) # (285, 3, 64, 64, 3)
# ang_difference = [] # get the angular difference of test vs. db images
histo = [] # store the histogram in 4 bins (10, 20, 40 or 180 degrees difference)
descriptorMatcher = cv2.BFMatcher() # setup Descriptormatcher in OpenCV
#arrays for the confusion matrix
predLabels = []

sess = tf.Session() # start Tensorflow Session
sess.run(init_op)
saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOGDIR + '/summary', sess.graph)
TOTAL_ITER = 0

# ---------- Let the CNN magic happen ----------------------
# logging
loss_log = open("Log_loss.txt", "w")

for e in range(NUM_EPOCHS):

    sess.run(iterator_train.initializer, feed_dict={data_dataset_train: triplets_f32})
    #sess.run(iterator_db.initializer, feed_dict= {data_db: db_imgs})
    # sess.run(iterator_test.initializer, feed_dict={test_db: test_imgs})
    loss_mean = []
    loss_itermediate = []
    for i in range(len(triplets_f32)):#len(triplets_f32) // batchSize):
        TOTAL_ITER = TOTAL_ITER + 1

        train_input = sess.run(next_element_train)
        input_dict = {inputs_: train_input[0], m: mar}

        batch_loss, _, step, summary = sess.run(
            [loss, opt, increment_step, merged_summaries],
            feed_dict=input_dict)

        writer.add_summary(summary, global_step=step)
        loss_mean.append(batch_loss)
        loss_itermediate.append(batch_loss)
        # print("Iter: {} of {}".format(TOTAL_ITER, NUM_EPOCHS * len(triplets_f32)),
        #         "Training loss: {:.4f}".format(batch_loss))

        if TOTAL_ITER % 10 == 0:
            # saver.save(sess, LOGDIR + '/checkpoints/model',global_step=step)
            # print("Model saved successfully")
            loss_log.write("Iter: {} Training loss: {:.4f} \n".format(TOTAL_ITER,np.mean(loss_itermediate)))
            loss_itermediate = []
        else:
            pass
        # every 1000th iteration: calculate the database descriptors and get the nearest Neighbors of the test images with the db
        if TOTAL_ITER % 1000 == 0:
            database_desc = np.array([], dtype=np.float32).reshape(0,16)
            sess.run(iterator_db.initializer, feed_dict={data_db: db_imgs})
            for i in range(len(db_imgs)): # get database descriptors
                db_input = sess.run(next_element_db)
                input_dict_db = {inputs_: db_input}
                descriptors = sess.run([logits], feed_dict=input_dict_db)
                database_desc = np.vstack((database_desc,descriptors[0]))
            desc_name = "db_16_" + str(TOTAL_ITER) +".npy"
            np.save(os.path.join(base_folder, desc_name), database_desc)

            predLabels = [] # only get one prediction for each image - clear it here
            ang_difference = [] # get the angular difference of test vs. db images
            sess.run(iterator_test.initializer, feed_dict={test_db: test_imgs})
            test_dscrp = np.array([], dtype=np.float32).reshape(0,16)
            for i in range(len(test_imgs)):
                ## Predict Test image Descriptors ...
                test_input = sess.run(next_element_test)
                input_dict_test = {inputs_: test_input}
                features = sess.run([logits], feed_dict=input_dict_test)
                test_dscrp = np.vstack((test_dscrp,features[0]))
                #print("Test Image Descriptors: {}".format(*features[0])) # get rid of outer list with "*"

                ## Perform Descriptor matching on feature and Database descriptors
                matches = descriptorMatcher.match(database_desc, np.reshape(*features[0], [-1, 16]))
                matches = sorted(matches, key=lambda x: x.distance) # sort matches according to distance
                # matches[0].queryIdx = closest image index [0 ... 1334] => 1335 = 5 (classes) * 267 (images per class in db)
                predClass = int(matches[0].queryIdx/ db_imgs_normalized.shape[1]) # rounds down

                predLabels.append(predClass)
                # print(" -----> Pred Class: {} - True Class: {} <-----".format(predClass, trueClass[i]))

                if predClass == trueClass[i]: # correct prediction
                # Retrieve index of test image [0:706]
                    if i < 707:
                        properTestImg = i
                    elif i >= 707:
                        looped = int(i/ 707.) # rounds down
                        properTestImg = i - (707*looped)

                    ang_difference.append(angle_between(db_poses_list[trueClass[i]][matches[0].queryIdx - (trueClass[i]*db_imgs_normalized.shape[1])], test_poses_list[trueClass[i]][properTestImg])*360/np.pi)
                    #print("Angle Difference: {}".format(ang_difference[-1]))

            # After looping through the test_images
            # SAVE TEST DESCRIPTORS
            desc_name = "test_16_" + str(TOTAL_ITER) +".npy"
            np.save(os.path.join(base_folder, desc_name), test_dscrp)

            curr_hist = []
            degree_10   = sum(i <= 10 for i in ang_difference) # count values in our angles which are smaller than 10
            degree_20   = sum( i <= 20 for i in ang_difference)
            degree_40   = sum( i <= 40 for i in ang_difference)
            degree_180  = sum( i <= 180 for i in ang_difference)

            # histogram values should be stored in %
            try: # else ZeroDivisionError
                degree_10   /= len(test_imgs)
                degree_20   /= len(test_imgs)
                degree_40   /= len(test_imgs)
                degree_180  /= len(test_imgs)
                curr_hist.append(degree_10*100.)
                curr_hist.append(degree_20*100.)
                curr_hist.append(degree_40*100.)
                curr_hist.append(degree_180*100.)

                #print("{} degrees don't fall into any of the bins -- error! ".format( len(ang_difference) - (degree_10+degree_180+degree_20+degree_40) ))
                #print(curr_hist)

                #bin = ['10', '20', '40', '180']
                #x_pos = [i for i, _ in enumerate(bin)]
                #plt.style.use('ggplot')
                #plt.bar(x_pos, curr_hist, color='green')
                #plt.xlabel("Angles, $^\circ$")
                #plt.ylabel("Percentage, %")
                #plt.title("Angle histogram")
                #plt.xticks(x_pos, ('<10$^\circ$', '<20$^\circ$', '<40$^\circ$', '<180$^\circ$'))
                #plt.yticks(np.arange(0, max(curr_hist)+1, 5.))
                # plt.show()
                #plt.savefig('hist_' + str(TOTAL_ITER) + '_loss_' + str(np.mean(loss_mean)) + '.png')

                histo.append(curr_hist)

            except ZeroDivisionError:
                pass

            #Check for same size of the lists
            if(len(predLabels) != len(trueClass)):
                print("Unequal shape for Confusion Matrix! Predictions: {}, Truth: {}".format(len(predClass), len(trueClass)))
            else: # print the confusion matrix
                cm = confusion_matrix(trueClass, predLabels)
                cm = (cm / cm.astype(np.float).sum(axis=1)) * 100 # normalize to get % of the confusion matrix
                df_cm = pd.DataFrame(cm, index = [i for i in class_folders], columns = [i for i in class_folders])
                print(df_cm)
                plt.figure(figsize = (10, 7))
                sn.heatmap(df_cm, annot=True, cmap="Blues")
                # print("Saving the image in follwing path: {}".format(base_folder))
                plt.savefig('confusion_m_' + str(TOTAL_ITER) + '.png', bbox_inches='tight')
                #plt.show()
                #plt.close('all')
            #classificationAccuracy = (len(ang_difference) / len(predLabels)) * 100

            ##GET DESCRIPTORS of every database image -- batchsize 1!!
            #if TOTAL_ITER % 10 == 0:# 1400:
            #db_input = sess.run(next_element_db)
            #input_dict_db = {inputs_: db_input}
            #descriptors = sess.run([logits], feed_dict=input_dict_db)
            #print(*descriptors[0].shape) # np.ndarray(1 by 16)
            #db_desc.append(*descriptors[0]) # * - to remove outer list [ [val1, ... valN]]

    print("Epoch: {}/{}...".format(e + 1, NUM_EPOCHS),
            "Training loss: {:.4f}".format(np.mean(loss_mean)))

np.save(os.path.join(base_folder, "histo.npy"), histo)
loss_log.close()
sess.close()
writer.flush()
writer.close()
#np.save(os.path.join(base_folder, "database_descriptors"), db_desc)
#print("Successfully saved the Database Descriptors")
print('DONE')
