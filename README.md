# bees-edge
## Aim
To provide a computationally cheap way of identifying bees based on camera input. We use this identification to black out regions of the video where there are no bees, with the aim to greatly reduce the filesize of the output video, which can then be sent on to an accurate machine learning model to track the possible bees. We're aiming to lean heavily towards false positives, as the ML model can filter these out, but we don't want so many false positives that we don't reduce the filesize very much.

Although we could use something like (YOLOv4-Tiny)[https://models.roboflow.com/object-detection/yolov4-tiny-darknet], we've found that simply detecting changing pixels seems to
recall almost all bees in a video. That is, even though it will also detect 
moving flowers and other regions we don't care so much about, we can be sure that
_almost all bees are detected_. 

Using such a simple method also has the benefit of not requiring training on a 
particular dataset. There are a few hyperparameters that need to be tuned, but
these can be done manually with little effort.

## Where this came from
This code originally sat inside of [Polytrack_v1](https://github.com/malikaratnayake/Polytrack_v1), but I realised that I wasn't actually using any of the code from that repository so I've broken this out into its own repo.