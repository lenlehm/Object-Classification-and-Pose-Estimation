# Object Classification and Pose Estimation in Python

In this project the classification and pose estimation of an object in a room is required. 
Therefore we have a big dataset of multiple classes, where we constructed a CNN but instead of predicting the label of this image, we take the feature descriptors and perform a k-NN matching with our database to check which pose along with the label is the closest to our extracted descriptor. 
The Setup is as follows: 
![architectural setup](/Setup.PNG)

For the Loss function we are using triplets. There is an achor, puller and a pusher who are tuning the loss function (see below):

![Loss function](/triplets.PNG)

Over time as you can see in the confusion matrix below, the system is learning and can classify the images better than better.
 ![Confusion Matrix](/confusion_matrix.gif)

 Furthermore the angle histogram (=how far away our pose guesses were) of the correct classified images also turned better than at the very first steps

 ![angle histogram](/Histogram_29000.png)

 Additionally we visualized the embeddings using the great tool in tensoflow which looked as follows: 

 ![Classification Embeddings](/PCA.PNG) 

