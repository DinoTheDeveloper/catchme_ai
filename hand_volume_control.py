import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def set_volume(volume_percentage):
    """Set system volume using osascript (macOS only)"""
    volume_percentage = max(0, min(100, volume_percentage))  # Ensure volume is between 0 and 100
    subprocess.run(['osascript', '-e', f'set volume output volume {volume_percentage}'])

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, img = cap.read()
    if not success:
        continue
        
    # Convert to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Initialize hand landmarks list
    landmark_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Store landmark coordinates
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
                
            # If landmarks detected, calculate volume
            if len(landmark_list) >= 9:
                # Get coordinates for thumb and index finger
                thumb_tip = (landmark_list[4][1], landmark_list[4][2])
                index_tip = (landmark_list[8][1], landmark_list[8][2])
                
                # Draw points and line between fingers
                cv2.circle(img, thumb_tip, 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, index_tip, 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, thumb_tip, index_tip, (255, 0, 255), 3)
                
                # Calculate distance and map to volume
                length = calculate_distance(thumb_tip, index_tip)
                
                # Convert length to volume percentage (map length 50-300 to volume 0-100)
                volPercentage = np.interp(length, [50, 300], [0, 100])
                
                # Set system volume (only update every few frames to avoid overwhelming the system)
                if int(volPercentage) % 5 == 0:  # Update every 5% change
                    set_volume(int(volPercentage))
                
                # Visual feedback
                volBar = np.interp(length, [50, 300], [400, 150])
                
                # Draw volume bar
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPercentage)}%', (40, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display title
    cv2.putText(img, "Hand Volume Control", (10, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                
    # Show image
    cv2.imshow("Hand Volume Control", img)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()