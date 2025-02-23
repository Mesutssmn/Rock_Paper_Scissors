import streamlit as st
import cv2
import mediapipe as mp
import random
import math
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def determine_winner(user_gesture, computer_gesture):
    """Determine the winner of the Rock-Paper-Scissors game."""
    if user_gesture == computer_gesture:
        return "Tie!"
    elif (user_gesture == "Rock" and computer_gesture == "Scissors") or \
         (user_gesture == "Scissors" and computer_gesture == "Paper") or \
         (user_gesture == "Paper" and computer_gesture == "Rock"):
        return "You Win!"
    else:
        return "Computer Wins!"

def get_gesture(hand_landmarks, handedness):
    """Detect the hand gesture."""
    open_fingers = []
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Thumb check
    thumb_tip = hand_landmarks[4]
    thumb_mcp = hand_landmarks[2]
    if (handedness == "Right" and thumb_tip.x < thumb_mcp.x) or \
       (handedness == "Left" and thumb_tip.x > thumb_mcp.x):
        open_fingers.append(finger_names[0])

    # Check other fingers
    for tip, pip, name in [(8, 6, 1), (12, 10, 2), (16, 14, 3), (20, 18, 4)]:
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            open_fingers.append(finger_names[name])

    # Determine gesture
    if len(open_fingers) == 0:
        return "Rock"
    elif len(open_fingers) == 5:
        return "Paper"
    elif len(open_fingers) == 2 and "Index" in open_fingers and "Middle" in open_fingers:
        return "Scissors"
    else:
        return None

def process_image(image):
    """Process the image and detect the hand gesture."""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    [h.classification[0].label for h in results.multi_handedness]
                ):
                    gesture = get_gesture(hand_landmarks.landmark, handedness)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    return gesture, image
            return None, image
    except Exception as e:
        logging.error(f"process_image error: {e}")
        return None, image

def main():
    st.title("Rock-Paper-Scissors Game (Static Image)")
    st.write("Capture an image from your webcam to play Rock-Paper-Scissors with hand gestures!")

    # Initialize session state for score
    if "user_score" not in st.session_state:
        st.session_state.user_score = 0
    if "computer_score" not in st.session_state:
        st.session_state.computer_score = 0
    if "ties" not in st.session_state:
        st.session_state.ties = 0

    # Display scores
    st.sidebar.header("Scoreboard")
    st.sidebar.write(f"**You:** {st.session_state.user_score}")
    st.sidebar.write(f"**Computer:** {st.session_state.computer_score}")
    st.sidebar.write(f"**Ties:** {st.session_state.ties}")

    # Cache clear button
    if st.button("Clear Cache and Restart"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.user_score = 0
        st.session_state.computer_score = 0
        st.session_state.ties = 0
        st.success("Cache cleared and scores reset!")

    # Capture image from webcam
    img_file_buffer = st.camera_input("Capture Image from Webcam")
    if img_file_buffer is not None:
        try:
            # Convert image to OpenCV format
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            cv2_img = cv2.flip(cv2_img, 1)  # Flip for mirror effect
            
            # Process the image
            gesture, processed_img = process_image(cv2_img)
            
            st.image(processed_img, channels="BGR", caption="Processed Image")
            
            if gesture:
                computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                result_text = determine_winner(gesture, computer_choice)
                st.write(f"**Your Gesture:** {gesture}")
                st.write(f"**Computer's Gesture:** {computer_choice}")
                st.write(f"**Result:** {result_text}")

                # Update scores
                if result_text == "You Win!":
                    st.session_state.user_score += 1
                elif result_text == "Computer Wins!":
                    st.session_state.computer_score += 1
                else:  # Tie
                    st.session_state.ties += 1
            else:
                st.write("No valid gesture detected. Please show your hand clearly.")
        except Exception as e:
            logging.error(f"Error: {e}")
            st.error(f"An error occurred: {e}. Please try again.")

if __name__ == "__main__":
    main()
