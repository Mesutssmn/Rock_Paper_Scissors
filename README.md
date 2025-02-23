# Rock-Paper-Scissors Hand Gesture Game

A simple Streamlit-based application that allows you to play Rock-Paper-Scissors using hand gestures captured via your webcam. The application uses the MediaPipe Hands module to detect hand landmarks and classify gestures into "Rock," "Paper," or "Scissors."

## Features
- Capture a static image from your webcam using Streamlit's `st.camera_input`.
- Detect hand gestures ("Rock," "Paper," "Scissors") with MediaPipe.
- Play against the computer, which randomly selects its gesture.
- Display the processed image with hand landmarks and game results.

## Requirements
- Python 3.6+
- Required Python libraries:
  - `streamlit`
  - `opencv-python`
  - `mediapipe`

## Installation
1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd rock-paper-scissors-game
2. Install the required dependencies:
   pip install streamlit opencv-python mediapipe

## Usage
1. Run the application:
   streamlit run rps_game_streamlit.py
2. Open your browser and go to the URL provided by Streamlit (typically http://localhost:8501).
3. Allow webcam access when prompted by the browser.
4. Click "Capture Image from Webcam" to take a snapshot of your hand gesture:
   Rock: Closed fist
   Paper: Open hand with all fingers extended
   Scissors: Index and middle fingers extended, others closed
5. The app will display your gesture, the computer's gesture, and the result.

## How It Works
* Hand Detection: Uses MediaPipe Hands to detect hand landmarks in the captured image.
* Gesture Recognition:
   * Rock: No open fingers detected.
   * Paper: All five fingers (including thumb) detected as open.
   * Scissors: Only index and middle fingers detected as open.
* Game Logic: Compares your gesture with a randomly chosen computer gesture to determine the winner.

## Troubleshooting
* Webcam Not Working:
   * Ensure your webcam is connected and accessible.
   * Check browser permissions for webcam access.
   * Click "Clear Cache and Restart" to reset the app state.
* Gesture Not Recognized:
   * Hold your hand steady and ensure it is fully visible in the camera frame.
   * Adjust lighting conditions (avoid very dark or overly bright environments).
* Errors:
   * Check the terminal for logged error messages (prefixed with ERROR).
   * Ensure all dependencies are installed correctly.

## Credits
* Built with Streamlit, OpenCV, and MediaPipe.
* Inspired by classic Rock-Paper-Scissors gameplay with a modern hand gesture twist.

## License
This project is open-source and available under the MIT License.
