import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp
from gesture_detector import GestureDetector, BettingGestures

class BettingMonitor:
    def __init__(self):
        self.gesture_detector = GestureDetector()
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Drawing settings
        self.betting_zones = {
            'main_bet': [(100, 300), (200, 400)],
            'side_bet': [(250, 300), (350, 400)],
            'split_zone': [(400, 300), (500, 400)],
            'insurance': [(550, 300), (650, 400)]
        }
        
        # Gesture history for smooth detection
        self.gesture_history = []
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.last_gesture_time = datetime.now()
        
        # Feedback settings
        self.feedback_messages = []
        self.feedback_duration = 2  # seconds

    def draw_betting_zones(self, frame):
        """Draw betting zones with improved visibility."""
        for zone_name, coords in self.betting_zones.items():
            # Draw zone rectangle
            cv2.rectangle(frame, coords[0], coords[1], (0, 255, 0), 2)
            
            # Add zone label with background
            label = zone_name.replace('_', ' ').title()
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(frame,
                        (coords[0][0], coords[0][1] - text_height - 10),
                        (coords[0][0] + text_width + 10, coords[0][1]),
                        (0, 255, 0),
                        -1)
            
            # Draw text
            cv2.putText(frame, label,
                       (coords[0][0] + 5, coords[0][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def draw_hand_landmarks(self, frame, hand_landmarks):
        """Draw enhanced hand landmarks with custom styling."""
        # Draw connections
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
        
        # Draw fingertips with different color
        fingertips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        for tip in fingertips:
            point = hand_landmarks.landmark[tip]
            x = int(point.x * frame.shape[1])
            y = int(point.y * frame.shape[0])
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

    def add_feedback(self, message, color=(255, 255, 255)):
        """Add feedback message to be displayed."""
        self.feedback_messages.append({
            'message': message,
            'color': color,
            'time': datetime.now()
        })

    def draw_feedback(self, frame):
        """Draw feedback messages with fade-out effect."""
        current_time = datetime.now()
        active_messages = []
        
        for msg in self.feedback_messages:
            time_diff = (current_time - msg['time']).total_seconds()
            if time_diff < self.feedback_duration:
                # Calculate fade-out alpha
                alpha = 1.0 - (time_diff / self.feedback_duration)
                color = tuple([int(c * alpha) for c in msg['color']])
                
                cv2.putText(frame, msg['message'],
                           (10, frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                active_messages.append(msg)
        
        self.feedback_messages = active_messages

    def process_frame(self, frame):
        """Process frame with enhanced visualization."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.gesture_detector.hands.process(rgb_frame)
        
        # Draw betting zones
        self.draw_betting_zones(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.draw_hand_landmarks(frame, hand_landmarks)
                
                # Detect gesture
                gesture, confidence = self.gesture_detector.detect_gesture(hand_landmarks)
                
                if gesture:
                    current_time = datetime.now()
                    time_since_last = (current_time - self.last_gesture_time).total_seconds()
                    
                    if time_since_last > 1.0:  # Prevent too frequent updates
                        self.current_gesture = gesture
                        self.gesture_confidence = confidence
                        self.last_gesture_time = current_time
                        
                        # Add feedback message
                        self.add_feedback(
                            f"Detected: {gesture.value} ({confidence:.1%})",
                            color=(0, 255, 0) if confidence > 0.8 else (255, 255, 0)
                        )
                    
                    # Draw gesture information
                    self.gesture_detector.draw_gesture_info(frame, gesture, confidence)
                    
                    # Draw gesture trail
                    if hasattr(hand_landmarks.landmark[0], 'x'):
                        pos = (
                            int(hand_landmarks.landmark[0].x * frame.shape[1]),
                            int(hand_landmarks.landmark[0].y * frame.shape[0])
                        )
                        self.gesture_history.append(pos)
                        if len(self.gesture_history) > 20:
                            self.gesture_history.pop(0)
                        
                        # Draw gesture trail
                        for i in range(1, len(self.gesture_history)):
                            cv2.line(frame,
                                   self.gesture_history[i-1],
                                   self.gesture_history[i],
                                   (0, 255, 255), 2)
        
        # Draw feedback messages
        self.draw_feedback(frame)
        
        # Draw instructions overlay
        self.draw_instructions(frame)
        
        return frame

    def draw_instructions(self, frame):
        """Draw instruction overlay."""
        instructions = [
            "Gesture Controls:",
            "- Closed Fist = Place Bet",
            "- Open Palm Up = Increase Bet",
            "- Two Fingers = Double Down",
            "- V Shape = Split",
            "- Open Palm Forward = Hold"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1]-300, 10),
                     (frame.shape[1]-10, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw instructions
        for i, text in enumerate(instructions):
            cv2.putText(frame, text,
                       (frame.shape[1]-290, 35 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Run the betting monitor."""
        print("Starting CatchMe.AI Betting Monitor...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Error: Could not read frame")
                    break
                
                # Process and display the frame
                processed_frame = self.process_frame(frame)
                
                # Show the frame
                cv2.imshow('CatchMe.AI - Betting Monitor', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error occurred: {e}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Application terminated")

if __name__ == "__main__":
    monitor = BettingMonitor()
    monitor.run()