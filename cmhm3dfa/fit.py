from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from menpo.shape import PointCloud
from .networks import DNFaceMultiView
from .utils import crop_image_bounding_box, tf_heatmap_to_lms

slim = tf.contrib.slim
this_file_dir = os.path.dirname(os.path.realpath(__file__))

class CMHMFitter(object):
    def __init__(self, sess, model_path, n_landmarks):
        self.images_input = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='input_images')
        net_model = DNFaceMultiView('')
        with tf.variable_scope('net'):
            lms_heatmap_prediction,states = net_model._build_network(self.images_input, datas=None, is_training=False, n_channels=n_landmarks)
            self.pts_predictions = tf_heatmap_to_lms(lms_heatmap_prediction)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, model_path)
        
        self.sess = sess

    def __call__(self, image, bounding_boxes):
        image_path = image.path
        image_ID_name = image_path.stem

        batch_pixels = []
        batch_trans = []
        box_names = []

        for box_index,bbox in enumerate(bounding_boxes):
            crop_img, crop_trans = crop_image_bounding_box(image, bbox, [256., 256.], base=256./256., order=1)
            input_pixels = crop_img.pixels_with_channels_at_back()
            batch_pixels.append(input_pixels)
            batch_trans.append(crop_trans)
            box_names.append('%s__%02d' % (image_ID_name, box_index))

        pts_pred = self.sess.run(
            self.pts_predictions,
            feed_dict={self.images_input: np.stack(batch_pixels, axis=0)})

        aligned_pts_list = []
        for btrans, pts, box_name in zip(batch_trans, pts_pred, box_names):
            orig_pts = btrans.apply(PointCloud(pts))
            image.landmarks['SAVEPTS_%s' % box_name] = orig_pts
            aligned_pts_list.append(orig_pts)

        return aligned_pts_list