from __future__ import print_function
import sys
import os
import unittest
import numpy as np
import tensorflow as tf
import menpo.io as mio
from pathlib import Path
# Add the project folder to first to sys.path to test the local cmhm3dfa package
this_file_dir = Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,str(this_file_dir.parent))
from cmhm3dfa.detect_face import MTCNNFaceDetector
from cmhm3dfa.fit import CMHMFitter

class TestDetection(unittest.TestCase):
    def test_detection(self):
        test_img = mio.import_image(this_file_dir.parent / 'image' / 'test.jpg')
        self.assertIsNotNone(test_img)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Graph().as_default():
            with tf.Session(config=config) as sess:
                face_detector = MTCNNFaceDetector(sess)
                face_detector.face_classification_threshold = 0.8
                results = face_detector(test_img)
                self.assertIsNotNone(results)
                # 32 faces in the images
                self.assertEqual(len(results),32) 

class TestAlignment84(unittest.TestCase):
    def test_detection(self):
        # must download and install
        model_path = os.path.expanduser('~/ckpt/3D84/model.ckpt-277538')        
        self.assertTrue(os.path.exists(model_path+'.index'))
        n_landmarks = 84
        test_img = mio.import_image(this_file_dir.parent / 'image' / 'test.jpg')
        self.assertIsNotNone(test_img)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Graph().as_default():
            with tf.Session(config=config) as detect_sess:
                face_detector = MTCNNFaceDetector(detect_sess)
                face_detector.face_classification_threshold = 0.8
                with tf.Graph().as_default():
                    with tf.Session(config=config) as align_sess:
                        fit_face = CMHMFitter(align_sess,model_path,n_landmarks)
                        results = fit_face(test_img, face_detector(test_img))
                        self.assertIsNotNone(results)
                        # 32 faces in the images
                        self.assertEqual(len(results),32)

if __name__ == '__main__':
    unittest.main(verbosity=2)

