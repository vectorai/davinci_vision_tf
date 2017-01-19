# davinci_vision_tf
This repository contains TensorFlow (TF) implementations of various Deep Learning models for common computer vision tasks (Object detection and Segmentation)
around the lab. For object detection we use a TF-based caffe rapper to perform Faster-RCNN. For segmentation we leverage a SegNet-style Deep Autoencoder.
# Depndencies
python2.7: numpy, tensorflow or tensorflow-gpu, Keras, easydict, rospy, opencv, PIL, matplotlib, h5py

# Segmentation
Our Segmentation Network is located in the `segmentation` folder.
##Files
# Object detection
Our Object detection network (along with an in-depth README) can be found in the appropriate github submodule. It is forked off of the smallcorgi implementation.
# Ros_tf.py
This is a file to publish a RosNode that is serving a TF model.
