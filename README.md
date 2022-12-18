# bees-edge
## Authorship
- Author: Lex Gallon
- Adapted from code by: Malika Nisal Ratnayake
- Supervised by: Alan Dorin, Adel Nadjaran Toosi (Monash University)


## Aim
To provide a computationally cheap way of identifying bees based on camera input. We use this identification to black out regions of the video where there are no bees, with the aim of reducing the filesize of the output video, which can then be sent on to an accurate machine learning model to track the possible bees. 

Ideally, we want to minimise the transmission of pixels that aren't part of a bee (or other insect of interest), but want to be very sure that we are not accidentally omitting any bee pixels. I.e. we lean towards false positives to be safe.

Although we could use something like [YOLOv4-Tiny](https://models.roboflow.com/object-detection/yolov4-tiny-darknet), we've found that simply detecting changing pixels seems to recall almost all bees in a video. That is, even though it will also detect moving flowers and other regions we don't care so much about, we can be sure that _almost all bees are detected_. 

Using such a simple method also has the benefit of not requiring training on a particular dataset. There are a few hyperparameters that need to be tuned, but these can be done manually with little effort.

## Where this came from
This code originally sat inside of [Polytrack_v1](https://github.com/malikaratnayake/Polytrack_v1), but I realised that I wasn't actually using any of the code from that repository so I've broken this out into its own repo.

## Setup
Clone this repo to your local computer:
```
git clone https://github.com/byebrid/bees-edge.git
```

Make sure you have some form of [conda](https://docs.conda.io/en/latest/) installed. Create the virtual environment for this repo:
```
conda env create -f environment.yml
```

Make sure to activate this environment everytime you want to run any code in this repo! E.g.
```
conda activate bees-edge
```

Finally, copy the provided example config file [eg_config.json](eg_config.json) into a new file called [config.json](config.json). Leave [eg_config.json](eg_config.json) as is and only edit [config.json](config.json) in the future when you want to adjust parameters.

Running [app.py](src/app.py) for the first time should also create an [out/](out/) directory. This is where each run's output videos and metadata will be stored. Be aware that running the main script lots of times can generate lots of video output, so make sure you don't overwhelm your storage!

## Quick note on how to provide video input
You can either use a live webcam or an existing video file as input. Simply change `video_source` in your config to either an integer (i.e. for live camera input, starts from 0) or string (for filepath). For filepaths, you can either specify a single video file, OR a directory of video files, in which case the script will run on each video file found in that directory.

### Webcam
To use a webcam, you must use its "index". This is probably 0 for an inbuilt webcam (that's what I use by default in [eg_config.json](eg_config.json)). For me, the in-built webcam actually registers itself as two devices (see [this stackoverflow post](https://unix.stackexchange.com/a/539573) for why), so if I want to use my USB webcam, I actually need to provide an index of 2 (not 1).

### Files
Files should be easier, just pass in a filepath relative to your current working
directory (which should always be the root of this repo!) or an absolute filepath.