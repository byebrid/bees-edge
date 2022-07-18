## Issue: tried to run Polytrack.py. `pandas` not installed!

### Solution: install pandas, add it to requirements file.

## Issue:
```
[ERROR:0@8.423] global /home/conda/feedstock_root/build_artifacts/libopencv_1656375765172/work/modules/videoio/src/cap.cpp (595) open VIDEOIO(CV_IMAGES): raised OpenCV exception:

OpenCV(4.5.5) /home/conda/feedstock_root/build_artifacts/libopencv_1656375765172/work/modules/videoio/src/cap_images.cpp:253: error: (-5:Bad argument) CAP_IMAGES: can't find starting number (in the name of file): ./data/video/output/video.avi in function 'icvExtractPattern'


Traceback (most recent call last):
  File "/home/lex/Development/Polytrack_v1/PolyTrack.py", line 18, in <module>
    processing_details= open(str(pt_cfg.POLYTRACK.OUTPUT)+ "videoprocessing_details.txt","w+")
FileNotFoundError: [Errno 2] No such file or directory: './data/video/output/videoprocessing_details.txt'
```

### Solution: Make that directory. But this should be done automatically ideally.

## Issue: 