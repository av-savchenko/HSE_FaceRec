Implementation of face identification system with convolutional neural network (CNN).
Pre-requisites:
- Qt 5
- opencv 3
- caffe and auxiliary libraries

We use the Lightened CNN (https://github.com/AlfredXiangWu/face_verification_experiment). See paper Wu, X., He, R., & Sun, Z. (2015). A Lightened CNN for Deep Face Representation. arXiv preprint arXiv:1511.02683.
We manually trained this CNN with the Casia WebFaces dataset (http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). We have not do any preprocessing except converting all images to grayscale and 128x128 cropping.
In our experiments our models showed superior accuracy of the 1-NN classifier with Euclidean distances between L2 Normed 256 features over either VGGNet or Lightened CNN A and B models. Our model is also the smallest one (22 MB). Further task - to check C model
