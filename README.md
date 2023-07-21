# The Blinking Game

## Description

This Python script uses OpenCV, dlib, and imutils libraries for face and blink detection, as well as Pygame for the timer. It captures video streams from the webcam and detects faces. It also tracks the blinking of each person in the video. A timer is shown at the bottom of the screen. The person who blinks first is labeled 'LOSER', and others are labeled as 'WINNER'. The frames where someone is labeled as 'LOSER' for the first time and where others are labeled 'WINNER' are saved as 'loser.png' and 'winner.png', respectively.

## Requirements

* Python 3.6+
* OpenCV
* dlib
* imutils
* Pygame

## Instructions

1. Install the required Python packages if not already installed:

```bash
pip install opencv-python
pip install dlib
pip install imutils
pip install pygame
```

2. Download the dlib's face landmarks model [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), and ensure the path to it is correct in the script.

3. Run the Python script:

```bash
python face_blink_detection.py
```

4. To terminate the program, press 'q' while the webcam window is in focus.

## Notes

* Adjust the blink threshold (`EYE_AR_THRESH`) and the consecutive frames threshold (`EYE_AR_CONSEC_FRAMES`) according to your requirements and environment to increase/decrease the sensitivity of blink detection.
* The script does not handle cases where more than one person blinks at the same time.
* The 'winner.png' file is continuously overwritten with the current frame where others are detected until the program is terminated.

## Credits

This code leverages OpenCV for webcam stream capture and image processing, dlib for face detection, and Pygame for timer functionality.
