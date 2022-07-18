# bees-edge
## Aim
To provide a computationally cheap way of identifying bees based on camera input. We use this identification to black out regions of the video where there are no bees, with the aim to greatly reduce the filesize of the output video, which can then be sent on to an accurate machine learning model to track the possible bees. We're aiming to lean heavily towards false positives, as the ML model can filter these out, but we don't want so many false positives that we don't reduce the filesize very much.

Although we could use something like [YOLOv4-Tiny](https://models.roboflow.com/object-detection/yolov4-tiny-darknet), we've found that simply detecting changing pixels seems to
recall almost all bees in a video. That is, even though it will also detect 
moving flowers and other regions we don't care so much about, we can be sure that
_almost all bees are detected_. 

Using such a simple method also has the benefit of not requiring training on a 
particular dataset. There are a few hyperparameters that need to be tuned, but
these can be done manually with little effort.

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

Make sure to activate this environment everytime you want to run any code in this repo!

The final step is to make your [config.json](config.json) file. Just run [app.py](src/app.py) once, and it'll create it for you!
```
python src/app.py
```
This will copy the [eg_config.json](eg_config.json) file provided in this repo. Leave [eg_config.json](eg_config.json) as is and only edit [config.json](config.json) in the future when you want to adjust parameters.

Running [app.py](src/app.py) for the first time will also likely create an [out/](out/) directory. This is where each run's output videos and metadata will be stored. Feel free to delete these runs whenever you want to regain some storage space!

## Quick note on how to provide video input
You can either use a live webcam or an existing video file as input. Simply change the INPUT in your config to either an integer (for webcam) or string (for filepath).
### Webcam
To use a webcam, you must use its "index". This is probably 0 for an inbuilt webcam (that's what I use by default in [eg_config.json](eg_config.json)). For me, the in-built webcam actually registers itself as two devices (see [this stackoverflow post](https://unix.stackexchange.com/a/539573) for why), so if I want to use my USB webcam, I actually need to provide an index of 2 (not 1).

### Files
Files should be easier, just pass in a filepath relative to your current working
directory (which should always be the root of this repo!) or an absolute filepath.