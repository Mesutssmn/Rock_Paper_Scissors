import av
import cv2
import numpy as np
import math
import mediapipe as mp
import random
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
        a: A point with attributes x and y.
        b: Another point with attributes x and y.
        
    Returns:
        float: The Euclidean distance between a and b.
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def determine_winner(user_gesture, computer_gesture):
    """
    Determine the game outcome based on user and computer gestures.
    
    Parameters:
        user_gesture (str): The gesture detected from the user's hand.
        computer_gesture (str): The gesture chosen by the computer.
    
    Returns:
        str: "Tie!", "You Win!", or "Computer Wins!" based on the comparison.
    """
    if user_gesture == computer_gesture:
        return "Tie!"
    elif (user_gesture == "Rock" and computer_gesture == "Scissors") or \
         (user_gesture == "Scissors" and computer_gesture == "Paper") or \
         (user_gesture == "Paper" and computer_gesture == "Rock"):
        return "You Win!"
    else:
        return "Computer Wins!"

def get_gesture(hand_landmarks, handedness):
    """
    Detect the hand gesture based on finger positions using dynamic thresholding.
    
    This function calculates the hand size as the distance from the wrist (landmark 0)
    to the middle finger MCP (landmark 9) and uses it as a reference to determine if
    a finger is extended. The thumb is checked by comparing the distance between its tip
    (landmark 4) and IP joint (landmark 3). For other fingers, the distance from the tip 
    to the MCP is used.
    
    Parameters:
        hand_landmarks (list): List of 21 hand landmarks.
        handedness (str): "Right" or "Left", indicating the orientation of the hand.
    
    Returns:
        str or None: The detected gesture ("Rock", "Paper", "Scissors") or None if ambiguous.
    """
    open_fingers = []
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Calculate hand size using distance from wrist (landmark 0) to middle finger MCP (landmark 9)
    hand_size = euclidean_distance(hand_landmarks[0], hand_landmarks[9])
    threshold_ratio = 0.6  # Adjust based on testing

    # Thumb check: Use distance between tip (landmark 4) and IP joint (landmark 3)
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_distance = euclidean_distance(thumb_tip, thumb_ip)
    if thumb_distance > threshold_ratio * hand_size:
        open_fingers.append(finger_names[0])

    # Check other fingers using tip to MCP distance
    for tip, mcp, name in [(8, 5, 1), (12, 9, 2), (16, 13, 3), (20, 17, 4)]:
        distance = euclidean_distance(hand_landmarks[tip], hand_landmarks[mcp])
        if distance > threshold_ratio * hand_size:
            open_fingers.append(finger_names[name])

    # Determine gesture based on the number of open fingers
    if len(open_fingers) == 0:
        return "Rock"
    elif len(open_fingers) >= 4:  # Allow slight thumb closure for "Paper"
        return "Paper"
    elif len(open_fingers) == 2 and "Index" in open_fingers and "Middle" in open_fingers:
        return "Scissors"
    else:
        return None

class RPSVideoTransformer(VideoTransformerBase):
    """
    A video transformer that processes each video frame, detects hand gestures,
    and overlays game information on the frame.
    """
    def __init__(self):
        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.computer_choice = None
        self.result_text = ""
        self.game_active = False

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a numpy array
        image = frame.to_ndarray(format="bgr24")
        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                [h.classification[0].label for h in results.multi_handedness]
            ):
                gesture = get_gesture(hand_landmarks.landmark, handedness)
                if gesture:
                    if not self.game_active:
                        self.computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                        self.result_text = determine_winner(gesture, self.computer_choice)
                        self.game_active = True

                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    y_start = 50
                    cv2.putText(image, f"Your Gesture: {gesture}", (50, y_start),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Computer: {self.computer_choice}", (50, y_start + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, self.result_text, (50, y_start + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        else:
            self.game_active = False

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Streamlit interface
st.title("Rock-Paper-Scissors Hand Gesture Game")
st.write("This app uses your webcam to play Rock-Paper-Scissors using hand gestures.")

# Start the video stream with our custom transformer
webrtc_streamer(key="rps", video_transformer_factory=RPSVideoTransformer)
