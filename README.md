# Ship Detector
A Real-time ship detector using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) and [OpenCV](http://opencv.org/).


## Requirements
- Ubuntu 16.04
- Python 3.5
- [Tensorflow 1.8](http://yongyong-e.tistory.com/10)
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

Run Demo
```bash
$ python detector_demo.py \
    --input_video=video_to_test \
    --export_model=your_trained_model
````

<div align='center'>
  <img src='object_detection/g3doc/img/demo20171018_093153.gif' width='600px'>
</div>
<div align='center'>
  <img src='object_detection/g3doc/img/demo20171018_093059.gif' width='600px'>
</div>

Run App
```bash
$ python detector_app.py \
    --input_cam=your_cam_num \
    --export_model=your_trained_model
````


## Extras
- [Creating your own dataset](http://yongyong-e.tistory.com/31)
- [Training your own dataset](http://yongyong-e.tistory.com/32)
- [Testing your own models](http://yongyong-e.tistory.com/35)
