# Face_Detection_Alignment
Face Detection and Alignment Tool
3D projection landmarks (84) and 2D multi-view landmarks(39/68)

Forked from https://github.com/jiankangdeng/Face_Detection_Alignment

Original author: Jiankang Deng https://ibug.doc.ic.ac.uk/people/jdeng

Related paper: 
Deng J, Zhou Y, Cheng S, Zaferiou S. Cascade multi-view hourglass model for robust 3D face alignment. In2018 13th IEEE International Conference on Automatic Face & Gesture Recognition (FG 2018) 2018 May 15 (pp. 399-403). IEEE.

## Environment:

### Running tests CPU

```
docker build --network=host -t cmhm3dfa-cpu:latest ./docker_cpu
docker run --network=host -v $PWD:/root/ --rm -ti cmhm3dfa-cpu:latest /bin/bash -c "source /opt/conda/bin/activate cmhm && cd ~ && python tests/test_cmhm3dfa.py"
```

### Building whl file

```
docker run --network=host -v $PWD:/root/ --rm -ti cmhm3dfa-cpu:latest /bin/bash -c "source /opt/conda/bin/activate cmhm && cd ~ && python setup.py bdist_wheel"
```

Same whl file will work for CPU and GPU, only tensorflow differs.
The whl file will be owned by root, use `--user` and `-v $PWD:/home/appuser/` to make a whl file owned by current user.

### Running tests  GPU

Requires nvidia-docker2
```
docker build --network=host -t cmhm3dfa-gpu:latest ./docker_gpu
nvidia-docker run --network=host -v $PWD:/root/ --rm -ti cmhm3dfa-gpu:latest /bin/bash -c "source /opt/conda/bin/activate cmhm && cd ~ && python tests/test_cmhm3dfa.py"
```

### Jupyter Lab (GPU)

Requires `--user` to work.

```
nvidia-docker run --user 1000 --network=host -v $PWD:/home/appuser/ --rm -ti cmhm3dfa-gpu:latest /bin/bash -c "source /opt/conda/bin/activate cmhm && cd ~ && jupyter lab"
```

## Train:
CUDA_VISIBLE_DEVICES="1" python train.py --train_dir=ckpt/3D84 --batch_size=8 --initial_learning_rate=0.0001 --dataset_dir=3D84/300W.tfrecords,3D84/afw.tfrecords,3D84/helen_testset.tfrecords,3D84/helen_trainset.tfrecords,3D84/lfpw_testset.tfrecords,3D84/lfpw_trainset.tfrecords,3D84/ibug.tfrecords,3D84/menpo_trainset.tfrecords --n_landmarks=84

## Test:
3D model: 84
2D model: frontal68/Union68/Union86(better)

## Pretrained Models:
https://drive.google.com/open?id=1DKTeRlJjyo_tD1EluDjYLhtKFPJ9vIVd
