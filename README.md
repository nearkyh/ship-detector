# Ship Detection App

A Real-time ship recognition app using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).


## Requirements

- Ubuntu 16.04
- Python 3.5
- [Tensorflow 1.4](http://yongyong-e.tistory.com/10)
- [OpenCV 3.2](http://yongyong-e.tistory.com/41)


## Getting Started

Creating virtualenv
1. virtualenv env/ship-detector --python=python3.5
2. source env/ship-detector/bin/activate

Binding OpenCV
1. cd env/ship-detector/lib
2. cp /usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so ~/env/ship-detector/lib/python3.5/site-packages

Install Dependencies
- pip3 install -r requirements.txt

Run Demo:
- python3 ship_detection_demo.py {filename.mp4}

<div align='center'>
  <img src='object_detection/g3doc/img/demo20171018_093153.gif' width='600px'>
</div>
<div align='center'>
  <img src='object_detection/g3doc/img/demo20171018_093059.gif' width='600px'>
</div>

Run App:
- python3 ship_detection_app.py


## Extras

- [Creating your own dataset](http://yongyong-e.tistory.com/31)
- [Training your own dataset](http://yongyong-e.tistory.com/32)
- [Testing your own models](http://yongyong-e.tistory.com/35)
