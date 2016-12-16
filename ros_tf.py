import rospy
import sys
#from rospy.numpy_msg import numpy_msg
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
def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
def demo(sess, net, im,vis=False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)

    # Detect all object classes and regress object bounds
    scores, boxes = im_detect(sess, net, im)
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    det_list=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        if vis:
            vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)
class RosTensorFlow():
    def __init__(self,model=None,type_='segmentation',img_shape=None, input_tensor=None):
        classify_image.maybe_download_and_extract()
        self._session = tf.Session()
        self.tensor_2_run=model
        init = tf.initialize_all_variables()
        self._session.run(init)
        #classify_image.create_graph()
        self._cv_bridge = CvBridge()
        self.input_tensor=input_tensor
        self.img_shape=img_shape
        self.type_=type_

        self._sub = rospy.Subscriber('image', Image, self.callback, queue_size=1)
        print('found camera')
        self._pub = rospy.Publisher('result', String, queue_size=1)
        print('setup publishing')
        self.count=0

    def callback(self, image_msg):
        print('getting image')
        img=np.fromstring(image_msg.data,np.uint8).reshape(image_msg.height, image_msg.width, 3)
        #cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        print('got image')
        if self.type_=='segmentation':
            #cv_image = np.array([np.rollaxis(cv2.resize(cv_image,(self.img_shape[2],self.img_shape[1])),2)])
            cv_image = np.array([np.rollaxis(cv2.resize(img,(self.img_shape[2],self.img_shape[1])),2)])
            cv_image[:,:,[0,1,2]] = cv_image[:,:,[2,1,0]] 
            print('reshaped image')
        # copy from
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/imagenet/classify_image.py
        #image_data = cv2.imencode('.jpg', cv_image)[1].tostring()
        # Creates graph from saved GraphDef.
            predictions = self._session.run(
                    self.tensor_2_run, {self.input_tensor: cv_image})#,K.learning_phase():0})
            print('ran tensor')
        # Creates node ID --> English string lookup.
            #predictions=visualize(np.argmax(predictions[0],axis=1).reshape((self.img_shape[1],self.img_shape[2])), False)
            predictions=np.argmax(predictions[0],axis=1).reshape((self.img_shape[1],self.img_shape[2]))
            print('reformatted results')
            cv2.imwrite('output_vids/vid1_%5d.png'%self.count,predictions)
            #self._pub.publish(self._cv_bridge.cv2_to_imgmsg(predictions,'bgr8'))
            self._pub.publish('segmented img # '+str(self.count))
            self.count+=1
            print('published results')
        if self.type_=='detection':
            self.tensor_2_run(self._session,cv_image)
            #TODO: Figure out how to publishmultiple arrays at once
    def main(self):
        rospy.spin()

if __name__ == '__main__':
    K.set_learning_phase(0)
    rospy.init_node('rostensorflow')
    tensor=None
    if sys.argv[1]=='segmentation':
        the_model=model((3,360,480))
        the_model.load_weights('model_weight_ep500.hdf5')
        x=K.placeholder(shape=(None,3,360,480),dtype='float32')
        outputs=the_model(x)
        tensor=RosTensorFlow(outputs,'segmentation',(3,360,480),x)
    elif sys.argv[1]=='detection':
        sys.path.insert(0,'Faster-RCNN_TF/tools/')
        import _init_paths
        from fast_rcnn.config import cfg
        from fast_rcnn.test import im_detect
        from fast_rcnn.nms_wrapper import nms
        from networks.factory import get_network
        net=get_network('VGGnet_test')
        tensor = RosTensorFlow(lambda x,y:demo(x,net,y),'detection')
    tensor.main()
