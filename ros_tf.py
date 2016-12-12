import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.models.image.imagenet import classify_image
from keras import backend as K
import sys
from vanilla_segnet import model

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb
class RosTensorFlow():
    def __init__(self,model=None,img_shape=None, input_tensor=None):
        classify_image.maybe_download_and_extract()
        self._session = tf.Session()
        self.tensor_2_run=model
        classify_image.create_graph()
        self._cv_bridge = CvBridge()
        self.input_tensor=input_tensor
        self.img_shape=img_shape

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('result', Image, queue_size=1)
        self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        self.use_top_k = rospy.get_param('~use_top_k', 5)

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        if sys.argv[1]=='segmentation':
            cv_image = np.array(np.rollaxis(cv2.resize(cv_image,(self.img_shape[2],self.img_shape[1])),2))
        # copy from
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
        #image_data = cv2.imencode('.jpg', cv_image)[1].tostring()
        # Creates graph from saved GraphDef.
        predictions = self._session.run(
            self.tensor_2_run, {self.input_tensor: cv_image})
        if sys.argv[1]=='segmentation':
        # Creates node ID --> English string lookup.
            predictions=visualize(np.argmax(predictions[0],axis=1).reshape((self.img_shape[1],self.img_shape[2])), False)

        self._pub.publish(self._cv_bridge.cv2_to_imgmsg(predictions,'bgr8'))

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor=None
    if sys.argv[1]=='segmentation':
        the_model=model((3,360,480))
        x=K.placeholder(shape=(None,3,360,480),dtype='float32')
        outputs=the_model(x)
        tensor=RosTensorFlow(outputs,(3,360,480),x)
    elif sys.argv[1]=='detection':
        pass
    tensor = RosTensorFlow()
    tensor.main()
