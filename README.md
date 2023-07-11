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

# Setup
Most of this section is dedicated how to get this running *on a raspberry pi*. If you just want to run this on your personal computer, don't be intimidated!

This is how I personally setup the [raspberry pi](https://www.raspberrypi.com/). You may need to make small changes here and there.

## Hardware
All of this was provided by Malika:
- Raspberry Pi 4
- Camera attached to the Pi
- Sandisk ultra microSD card (128 GB)
- Sandisk microSD adapter (since my laptop only has normal-sized SD slot)
- Cygnett 20,000 mAh power bank (ChargeUp Quad)
- Charging cables (USB-A to USB-C for powering the Pi, my own USB-A to USB-A for charging power bank)

## Steps
> Most of these steps are only for setting up the Raspberry Pi for the first time. If you don't need to setup a raspberry pi and just want to run this on a normal computer, just follow the steps to
> - clone this repo (17),
> - create the conda environment (22),
> - create the config file (23).
> 
> I'd recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) if you haven't installed conda before.
>
> If you're on Windows, you probably want to use [git bash](https://gitforwindows.org/).
>
> For simplicity, I'll assume you're using the Lite version of the Raspberry Pi OS, which then requires you to ssh to connect to the Pi. Some steps can be omitted if you are using the desktop version, mainly any steps related to ssh and wifi connection.

1. Download and install [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
2. Open Raspberry Pi Imager.
3. Select the OS. I used [Raspberry Pi OS Lite](https://www.raspberrypi.com/software/operating-systems/) (64-bit), specifically version 6.1.21-v8+. "Lite" means no desktop GUI, so select the normal OS if you want to use a desktop on the Pi.
4. Plug in the SD card and select it ("Choose Storage") in Imager.
5. Ensure ssh is enabled.
6. Set a password for the Pi.
7. Optionally change the hostname of the Pi. By default, it's `raspberrypi.local`.
8. Select "Write" to write the OS to the SD card. Wait for this to finish.
9. Create a new empty file called `ssh` to the `bootfs` partition of the SD card.
10. Create a new file called `wpa_supplicant.conf` to the `bootfs` partition. Refer to the section below to see what the contents of this file should be.
11. Edit `cmdline.txt` in the `bootfs` partition. Add `systemd.restore_state=0 rfkill.default_state=1` to the end of this file (make sure you add a space!).
12. Eject the SD card from your computer, and plug it into the Pi.
13. Connect the Pi to power. Give it a minute to ensure it has booted.
14. `ssh pi@raspberrypi.local` to connect via ssh (if you changed the Pi's hostname earlier, make sure it matches here). Accept any prompts and enter the password you set earlier when asked. You should now be logged in as the user `pi` on the Pi!
15. `sudo apt-get update && sudo apt-get upgrade` to update any existing software.
16. `sudo apt-get install git` to install git.
17. `git clone https://github.com/byebrid/bees-edge.git` to clone this repo. You may need to enter your GitHub username and password (use an [access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)!).
18. `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh` to download the [miniforge3](https://github.com/conda-forge/miniforge#miniforge3) installer. *Note you may need to select a different installer if you chose to install a different OS.*
19. `chmod u+x Miniforge*` to make the installer exectuable.
20. `./Miniforge*` to run the installer. Follow all of its prompts. One prompt will ask if you want `conda` to be initialised whenever you login. Select yes, but note that this makes your shell a little slower to start up.
21. `sudo raspi-config` to ensure the camera plays nicely with `opencv`. Go to "Interface Options" and enable "Legacy Camera". Reboot as required.
22. `cd` into this repo and enter `conda env create -f environment.yml` to create the conda environment. Enter yes if a confirmation prompt appears.
23. `cp eg_config.json config.json` to create a config file. Edit `config.json` to change any parameters for the program.

### wpa_supplicant.conf contents
> This is only needed for a headless OS. If using a desktop Pi OS, you can skip this. Instead, connect your keyboard and monitor to the Pi and connect to Wifi like a normal person. 

Replace the `ssid` and `psk` values with your wifi network name and password (without the "<" and ">", but keep the quotation marks).

You may want to change the `country` value too. You must use a [ISO 3166-1 alpha-2 country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).

This worked for me, but you may require a more complicated `wpa_supplicant.conf`. I can only point you to this [Arch Linux](https://wiki.archlinux.org/title/wpa_supplicant) page, this [man page](https://linux.die.net/man/5/wpa_supplicant.conf), and wish you good luck!

```
country=AU
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1

network={
    ssid="<my (5GHz) WIFI SSID>"
    psk="<my WIFI password>"
}
```

### 🛜 Connecting the raspberry pi to other Wifi networks
Read this if you're using a headless OS on the raspberry pi and want to connect to multiple wifi networks.

One option I didn't test is simply adding multiple networks to the `wpa_supplicant.conf` file (see [this stackoverflow post](https://raspberrypi.stackexchange.com/a/40144)). I didn't do this myself because this file is removed from the boot partition upon booting the Pi, and I wasn't sure if I could edit it in the main filesystem after the fact or create a new copy in the boot partition.

I relied on this [ArchLinux wiki page](https://wiki.archlinux.org/title/wpa_supplicant) for a lot of this.

Instead, I connected to my Pi via ssh. I then ran
```shell
$ wpa_cli -i wlan0
```
I found that if I didn't specify the wlan0 interface, `wpa_cli` would choose the wrong interface and so couldn't detect any wifi networks (see [this ArchLinux thread](https://bbs.archlinux.org/viewtopic.php?id=234877)).

Start your mobile phone's hotspot.

At this point, scan for all available wifi networks using
```shell
> scan
OK
```

Wait for the message confirming that the scan finished. Then list all networks using
```shell
> scan_results
```

Search for your phone's hotspot's SSID. If you notice encoded bytes in the SSID (e.g. "\xC3\xA9"), I would recommend just changing your hotspot's name to use safe/boring characters, and do the scan again. Otherwise, you'll want to [encode your SSID into hex](https://raspberrypi.stackexchange.com/a/103637).

Once you've verified your hotspot is available, do
```shell
> add_network
1
> set_network 1 ssid "<hotspot's SSID>"
> set_network 1 psk "<hotspot's password>"
> enable_network 1
```

`add_network` creates a new *empty* network configuration. You then edit this configuration using those `set_network` commands. Note that the ID following `set_network` should match that returned by `add_network`!

I didn't actually confirm if you need to do `enable_network` when already connected to a wifi network, but better to be safe than sorry! This *may* be needed to automatically connect to your hotspot if your normal wifi network is unavailable.

To finish up,
```shell
> save_config
OK
> quit
```

Now test that everything by disabling your normal wifi network, connecting your personal computer to your mobile hotspot, and verifying that you can still ssh into the Pi. Another way of verifying the Pi has connected is by checking your phone to see how many devices are connected to your hotspot.

## Running the program
Ensure your conda environment is activated:
```shell
(base) $ conda activate bees-edge
```

Then just run `app.py`:
```shell
(bees-edge) $ python bees_edge/app.py
```

You should soon see the program tell you  where any output will go, amongst other information. If you don't see any error messages, then things are *probably* working! To make sure things are working, you can wait for confirmation that 'n' frames have been read, and perhaps check the output directory to verify that a video file has been created and has some data (e.g. filesize > 0).

If using a camera as input, **enter Ctrl-C to stop at any time. Give it a minute to gracefully exit**, else the output video

## Notes on setup
1. I've used the *lite* OS because it was just easier for me to SSH into the raspberry pi rather than rely on an external keyboard and monitor. The lite version presumably runs a little quicker too.
2. I've used the *64*-bit OS so I have the option of using more than 3 GB memory for a single process (see [Wikipedia - 3 GB barrier](https://en.wikipedia.org/wiki/3_GB_barrier)).
3. You don't *need* to use [conda](https://docs.conda.io/en/latest/). You could just use [pip](https://pip.pypa.io/en/stable/getting-started/), preferably with a virtual environment (e.g. [venv](https://docs.python.org/3/library/venv.html)). conda is just easy for me to document.



## Quick note on how to provide video input
You can either use a live webcam or an existing video file as input. Simply change `video_source` in your config to either an integer (i.e. for live camera input, starts from 0) or string (for filepath). For filepaths, you can either specify a single video file, OR a directory of video files, in which case the script will run on each video file found in that directory.

### Webcam
To use a webcam, you must use its "index". This is probably 0 for an inbuilt webcam (that's what I use by default in [eg_config.json](eg_config.json)). For me, the in-built webcam actually registers itself as two devices (see [this stackoverflow post](https://unix.stackexchange.com/a/539573) for why), so if I want to use my USB webcam, I actually need to provide an index of 2 (not 1).

### Files
Files should be easier, just pass in a filepath relative to your current working
directory (which should always be the root of this repo!) or an absolute filepath.