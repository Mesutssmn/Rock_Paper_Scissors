import cv2
import numpy as np
import math
import mediapipe as mp
import random

def euclidean_distance(a, b):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
        a (mediapipe.framework.formats.landmark_pb2.NormalizedLandmark): The first point.
        b (mediapipe.framework.formats.landmark_pb2.NormalizedLandmark): The second point.

    Returns:
        float: The Euclidean distance between points `a` and `b`.
    """
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

def determine_winner(user_gesture, computer_gesture):
    """
    Determine the winner of the Rock-Paper-Scissors game.

    Parameters:
        user_gesture (str): The gesture chosen by the user ("Rock", "Paper", or "Scissors").
        computer_gesture (str): The gesture chosen by the computer ("Rock", "Paper", or "Scissors").

    Returns:
        str: The result of the game ("Tie!", "You Win!", or "Computer Wins!").
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
    Detect the hand gesture based on the positions of hand landmarks.

    Parameters:
        hand_landmarks (list): A list of 21 landmarks representing the detected hand.
        handedness (str): The handedness of the detected hand ("Right" or "Left").

    Returns:
        str: The detected gesture ("Rock", "Paper", or "Scissors"). Returns `None` if the gesture is ambiguous.
    """
    open_fingers = []
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    # Calculate hand size as distance from wrist (0) to middle finger MCP (9)
    hand_size = euclidean_distance(hand_landmarks[0], hand_landmarks[9])
    threshold_ratio = 0.6  # Threshold for detecting open fingers

    # Thumb check: Use distance between tip (4) and IP joint (3)
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_distance = euclidean_distance(thumb_tip, thumb_ip)
    if thumb_distance > threshold_ratio * hand_size:
        open_fingers.append(finger_names[0])

    # Check other fingers: Tip to MCP distance (more stable)
    for tip, mcp, name in [(8, 5, 1), (12, 9, 2), (16, 13, 3), (20, 17, 4)]:
        distance = euclidean_distance(hand_landmarks[tip], hand_landmarks[mcp])
        if distance > threshold_ratio * hand_size:
            open_fingers.append(finger_names[name])

    # Determine gesture
    if len(open_fingers) == 0:
        return "Rock"
    elif len(open_fingers) >= 4:  # Allow slight thumb closure for "Paper"
        return "Paper"
    elif len(open_fingers) == 2 and "Index" in open_fingers and "Middle" in open_fingers:
        return "Scissors"
    else:
        return None

# Main game loop
cap = cv2.VideoCapture(0)
computer_choice = None
result_text = ""
game_active = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                            [h.classification[0].label for h in results.multi_handedness]):
            # Get gesture
            gesture = get_gesture(hand_landmarks.landmark, handedness)
            
            if gesture:  # Valid gesture detected
                if not game_active:
                    computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                    result_text = determine_winner(gesture, computer_choice)
                    game_active = True

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display texts
                y_start = 50
                cv2.putText(frame, f"Your Gesture: {gesture}", (50, y_start), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Computer: {computer_choice}", (50, y_start + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, result_text, (50, y_start + 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
    else:
        game_active = False

    cv2.imshow('Rock-Paper-Scissors', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
