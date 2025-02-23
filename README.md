# Rock-Paper-Scissors Hand Gesture Game

## Overview

This project implements a Rock-Paper-Scissors game using hand gestures, leveraging OpenCV and MediaPipe's Hand Tracking module. Users can play against the computer by showing a hand gesture, which is detected in real-time through a webcam.

## Features

- Real-time hand gesture recognition
- Automatic determination of Rock, Paper, or Scissors based on finger positions
- Randomized computer selection of a gesture
- Displays results on the screen

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- MediaPipe

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rock-paper-scissors-hand-gesture.git
   cd rock-paper-scissors-hand-gesture
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

## Usage

Run the script:

```bash
python rps_gesture.py
```

### Controls

- Show your hand with one of the following gestures:
  - **Rock** (Fist)
  - **Paper** (All fingers extended)
  - **Scissors** (Index and Middle fingers extended)
- The computer will randomly choose a move.
- The game will display the result on the screen.
- Press 'q' to exit.

## License

This project is open-source and licensed under the MIT License.

## Acknowledgements

- OpenCV for computer vision processing
- MediaPipe for hand tracking

