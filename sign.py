import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from datetime import datetime
import pickle
from collections import deque

class SASignLanguageTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.typed_text = ""
        self.last_letter = ""
        self.letter_countdown = 0
        self.recording = False
        self.training_data = {}
        self.motion_buffer = deque(maxlen=30)  # Store last 30 frames of hand positions
        self.word_suggestions = []
        self.common_words = self.load_common_words()
        
        # Load existing training data if available
        self.load_training_data()
        
        # Initialize feature thresholds
        self.thresholds = {
            'distance_threshold': 0.1,
            'angle_threshold': 0.5,
            'motion_threshold': 0.03,
            'confidence_threshold': 0.7
        }

    def load_common_words(self):
        # Add common English words for suggestions
        return ['hello', 'thank', 'you', 'please', 'how', 'are', 'good', 'bye', 'morning', 'night']

    def load_training_data(self):
        if os.path.exists('sasl_training_data.pkl'):
            try:
                with open('sasl_training_data.pkl', 'rb') as f:
                    self.training_data = pickle.load(f)
            except:
                print("Could not load training data, starting fresh")
                self.training_data = {}

    def save_training_data(self):
        with open('sasl_training_data.pkl', 'wb') as f:
            pickle.dump(self.training_data, f)

    def extract_hand_features(self, landmarks):
        """Extract comprehensive hand features including angles, distances, and relative positions"""
        features = {}
        
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Calculate angles between all fingers
        for i in range(5):  # For each finger
            base = i * 4 + 1
            angles = self.calculate_finger_angles(points[base:base+4])
            features[f'finger_{i}_angles'] = angles
        
        # Calculate distances between fingertips and palm
        palm = points[0]
        for i, tip in enumerate([4, 8, 12, 16, 20]):  # Finger tip indices
            dist = np.linalg.norm(points[tip] - palm)
            features[f'finger_{i}_palm_distance'] = dist
        
        # Calculate relative positions between fingertips
        for i in range(5):
            for j in range(i+1, 5):
                tip1 = points[[4, 8, 12, 16, 20][i]]
                tip2 = points[[4, 8, 12, 16, 20][j]]
                features[f'tips_{i}_{j}_distance'] = np.linalg.norm(tip1 - tip2)
        
        return features

    def calculate_finger_angles(self, points):
        """Calculate angles between finger segments"""
        angles = []
        for i in range(len(points)-2):
            v1 = points[i] - points[i+1]
            v2 = points[i+2] - points[i+1]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angles.append(angle)
        return angles

    def detect_motion(self):
        """Detect hand motion patterns from the motion buffer"""
        if len(self.motion_buffer) < 2:
            return None
            
        # Calculate movement pattern
        movements = []
        for i in range(1, len(self.motion_buffer)):
            prev = self.motion_buffer[i-1]
            curr = self.motion_buffer[i]
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            movements.append((dx, dy))
        
        # Analyze movement pattern
        total_dx = sum(m[0] for m in movements)
        total_dy = sum(m[1] for m in movements)
        
        # Detect specific motion patterns
        if abs(total_dx) > self.thresholds['motion_threshold']:
            if total_dx > 0:
                return 'right'
            return 'left'
        elif abs(total_dy) > self.thresholds['motion_threshold']:
            if total_dy > 0:
                return 'down'
            return 'up'
        elif self.detect_circular_motion(movements):
            return 'circular'
        return None

    def detect_circular_motion(self, movements):
        """Detect if movements form a circular pattern"""
        if len(movements) < 8:
            return False
            
        # Calculate direction changes
        direction_changes = 0
        prev_dx, prev_dy = movements[0]
        for dx, dy in movements[1:]:
            if (prev_dx * dx < 0) or (prev_dy * dy < 0):
                direction_changes += 1
            prev_dx, prev_dy = dx, dy
            
        return direction_changes >= 6  # Arbitrary threshold for circular motion

    def train_letter(self, letter, landmarks, motion=None):
        """Record training data for a letter"""
        if letter not in self.training_data:
            self.training_data[letter] = {'static_features': [], 'motion': None}
        
        features = self.extract_hand_features(landmarks)
        self.training_data[letter]['static_features'].append(features)
        if motion:
            self.training_data[letter]['motion'] = motion
        
        print(f"Recorded training data for letter {letter}")
        self.save_training_data()

    def match_letter(self, landmarks):
        """Match current hand position to trained letters"""
        if not self.training_data:
            return None, 0
            
        current_features = self.extract_hand_features(landmarks)
        best_match = None
        best_score = 0
        
        for letter, data in self.training_data.items():
            if not data['static_features']:
                continue
                
            # Compare features with training data
            scores = []
            for training_features in data['static_features']:
                score = self.compare_features(current_features, training_features)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            # Check if motion is required
            if data['motion']:
                detected_motion = self.detect_motion()
                if detected_motion != data['motion']:
                    continue
            
            if avg_score > best_score and avg_score > self.thresholds['confidence_threshold']:
                best_score = avg_score
                best_match = letter
        
        return best_match, best_score

    def compare_features(self, features1, features2):
        """Compare two sets of features and return similarity score"""
        score = 0
        count = 0
        
        for key in features1:
            if key in features2:
                if isinstance(features1[key], (list, np.ndarray)):
                    similarity = 1 - np.mean(np.abs(np.array(features1[key]) - np.array(features2[key])))
                else:
                    similarity = 1 - abs(features1[key] - features2[key])
                score += similarity
                count += 1
        
        return score / count if count > 0 else 0

    def update_word_suggestions(self):
        """Update word suggestions based on current typed text"""
        if not self.typed_text:
            self.word_suggestions = []
            return
            
        current_word = self.typed_text.split()[-1] if self.typed_text else ""
        self.word_suggestions = [word for word in self.common_words 
                               if word.startswith(current_word.lower())][:3]

    def process_frame(self, frame, training_mode=False):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        # Create text display area with word suggestions
        text_area = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
        
        # Display typed text
        cv2.putText(text_area, self.typed_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display word suggestions
        suggestions_text = " | ".join(self.word_suggestions)
        cv2.putText(text_area, suggestions_text, (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Store hand position for motion detection
            palm_pos = hand_landmarks.landmark[0]
            self.motion_buffer.append((palm_pos.x, palm_pos.y))

            if training_mode:
                cv2.putText(frame, "Training Mode", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Detect letter
                letter, confidence = self.match_letter(hand_landmarks)
                
                if letter and confidence > self.thresholds['confidence_threshold']:
                    if self.last_letter != letter:
                        self.letter_countdown = 10
                        self.last_letter = letter
                    elif self.letter_countdown == 0:
                        self.typed_text += letter
                        self.last_letter = ""
                        self.update_word_suggestions()
                
                if self.letter_countdown > 0:
                    self.letter_countdown -= 1

                # Display current detection
                if letter:
                    cv2.putText(frame, f"{letter} ({confidence:.2f})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Combine frame with text area
        combined_frame = np.vstack([frame, text_area])
        return combined_frame

def main():
    trainer = SASignLanguageTrainer()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    training_mode = False
    training_letter = None
    recording_countdown = 0
    
    print("\nControls:")
    print("\\ - Toggle training mode") 
    print("A-Z - Select letter to train in training mode")
    print("0-9 - Select number to train in training mode")
    print("Space - Add space")
    print("Backspace - Delete last character")
    print("Enter - Clear text")
    print("Tab - Accept word suggestion")
    print("= - Quit")
    print("\nTraining Status: OFF")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        
        # Get hand detection results
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = trainer.hands.process(frame_rgb)
        
        # Add training mode status to frame
        if training_mode:
            cv2.putText(frame, "Training Mode ON", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if training_letter:
                cv2.putText(frame, f"Recording '{training_letter}' ({recording_countdown})", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        output_frame = trainer.process_frame(frame, training_mode)
        cv2.imshow('SASL Trainer', output_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Handle key controls
        if key == ord('='):  # Quit
            break
        elif key == ord('\\'):  # Backslash to toggle training mode
            training_mode = not training_mode
            training_letter = None  # Reset training letter when toggling mode
            recording_countdown = 0
            print("\nTraining Mode:", "ON" if training_mode else "OFF")
        elif key == ord('\t') and trainer.word_suggestions:  # Tab for word suggestions
            words = trainer.typed_text.split()
            if words:
                words[-1] = trainer.word_suggestions[0]
                trainer.typed_text = " ".join(words) + " "
                trainer.update_word_suggestions()
        elif key == ord(' '):  # Space
            trainer.typed_text += " "
            trainer.update_word_suggestions()
        elif key == ord('\r'):  # Enter
            trainer.typed_text = ""
            trainer.update_word_suggestions()
        elif key == ord('\b'):  # Backspace
            trainer.typed_text = trainer.typed_text[:-1]
            trainer.update_word_suggestions()
        elif training_mode and (
            (ord('a') <= key <= ord('z')) or 
            (ord('A') <= key <= ord('Z')) or 
            (ord('0') <= key <= ord('9'))
        ):
            # Start recording training data for the selected letter
            training_letter = chr(key).upper()
            recording_countdown = 30  # Record for 30 frames
            print(f"\nRecording training data for '{training_letter}'")
            print(f"Please hold the '{training_letter}' hand sign steady...")
            
        # Handle training data recording
        if training_mode and training_letter and recording_countdown > 0:
            if results.multi_hand_landmarks:
                trainer.train_letter(training_letter, results.multi_hand_landmarks[0])
                # Visual feedback for recording
                cv2.putText(frame, f"Recording '{training_letter}'", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames remaining: {recording_countdown}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("\nNo hand detected! Please show your hand to the camera.")
            
            recording_countdown -= 1
            if recording_countdown == 0:
                print(f"\nFinished recording data for '{training_letter}'")
                print("You can now record another letter or exit training mode (press \\)")
                training_letter = None

        # Update display
        cv2.imshow('SASL Trainer', output_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()