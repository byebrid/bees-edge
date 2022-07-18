# bees-edge
## Aim
To provide a computationally cheap way of identifying bees based on camera input. We use this identification to black out regions of the video where there are no bees, with the aim to greatly reduce the filesize of the output video, which can then be sent on to an accurate machine learning model to track the possible bees. We're aiming to lean heavily towards false positives, as the ML model can filter these out, but we don't want so many false positives that we don't reduce the filesize very much.

## Where this came from
This code originally sat inside of [Polytrack_v1](https://github.com/malikaratnayake/Polytrack_v1), but I realised that I wasn't actually using any of the code from that repository so I've broken this out into its own repo.