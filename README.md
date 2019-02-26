# Face_Detection_Alignment
Face Detection and Alignment Tool
3D projection landmarks (84) and 2D multi-view landmarks(39/68)

Forked from https://github.com/jiankangdeng/Face_Detection_Alignment

Original author: Jiankang Deng https://ibug.doc.ic.ac.uk/people/jdeng

Related paper: 
Deng J, Zhou Y, Cheng S, Zaferiou S. Cascade multi-view hourglass model for robust 3D face alignment. In2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018) 2018 May 15 (pp. 399-403). IEEE.

Environment:
Tensorflow 1.12, menpo 0.8.1, python 3.6

Train:
CUDA_VISIBLE_DEVICES="1" python train.py --train_dir=ckpt/3D84 --batch_size=8 --initial_learning_rate=0.0001 --dataset_dir=3D84/300W.tfrecords,3D84/afw.tfrecords,3D84/helen_testset.tfrecords,3D84/helen_trainset.tfrecords,3D84/lfpw_testset.tfrecords,3D84/lfpw_trainset.tfrecords,3D84/ibug.tfrecords,3D84/menpo_trainset.tfrecords --n_landmarks=84

Test:
3D model: 84
2D model: frontal68/Union68/Union86(better)

Pretrained Models:
https://drive.google.com/open?id=1DKTeRlJjyo_tD1EluDjYLhtKFPJ9vIVd
