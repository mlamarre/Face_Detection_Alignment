import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "cmhm3dfa",
    version = "0.1.0",
    author = "Jiankang Deng",
    author_email = "j.deng16@imperial.ac.uk",
    description = ("Multi-view Hourglass Model for Robust 3D Face Alignment and Joint face detection and alignment using multitask cascaded convolutional network."),
    license = "MIT",
    keywords = "face detection alignement",
    url = "https://github.com/jiankangdeng/Face_Detection_Alignment",
    packages=['cmhm3dfa'],
    package_data = {'cmhm3dfa':['pretrained_mtcnn/*.npy']},
    install_requires=[
        'numpy',
        'menpo'
    ],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: C++',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
)
