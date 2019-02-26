# Ship Detector
A Real-time ship detector using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).


## Requirements
- Ubuntu 16.04
- Python 3.5
- [TensorFlow 1.9](http://yongyong-e.tistory.com/10)
- [OpenCV 3.2](http://yongyong-e.tistory.com/41)


## Getting Started
Creating virtualenv
```bash
$ cd Ship-Detector
$ virtualenv env --python=python3.5
$ source env/bin/activate
```

Install Dependencies
```bash
$ pip install -r requirements.txt
```

Binding OpenCV
```bash
$ cp /usr/local/lib/python3.5/dist-packages/cv2.cpython-35m-x86_64-linux-gnu.so \
    ~/Ship-Detector/env/lib/python3.5/site-packages
```

Download the frozen inference graph (ssd_mobilenet_v1_ship_15000) from the [Google Drive](https://drive.google.com/open?id=1HQxJMlF7Iaho4kuXOSjv08eDe2xRlKod).

Run
```bash
$ python detector.py
```

Optional arguments (default value):
* Video to test `--video=test.mp4`
* Device index of the camera `--camera=None`
* Saved model name `--model=ssd_inception_v2_ship_15000`

<div align='center'>
  <img src='object_detection/g3doc/img/demo20171018_093153.gif' width='600px'>
</div>
<div align='center'>
  <img src='object_detection/g3doc/img/demo20171018_093059.gif' width='600px'>
</div>


## Using GPU
Requirements [(Install reference)](http://yongyong-e.tistory.com/10)
- [Tensorflow-GPU 1.9](https://www.tensorflow.org/install)
- [CUDA 9.0](https://developer.nvidia.com/cuda-downloads)
- [cuDNN 7.1.4](https://developer.nvidia.com/cudnn)


## Extras
- [Creating your own dataset](http://yongyong-e.tistory.com/31)
- [Training your own dataset](http://yongyong-e.tistory.com/32)
- [Testing your own models](http://yongyong-e.tistory.com/35)
