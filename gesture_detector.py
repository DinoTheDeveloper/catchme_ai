import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
from collections import deque

class BettingGestures(Enum):
    PLACE_BET = "Place Bet"  # Closed fist
    INCREASE_BET = "Increase Bet"  # Open palm facing up
    DECREASE_BET = "Decrease Bet"  # Open palm facing down
    DOUBLE_DOWN = "Double Down"  # Two fingers
    SPLIT = "Split"  # V shape with fingers
    HOLD = "Hold"  # Palm forward
    CANCEL = "Cancel"  # X shape with arms
    CALL = "Call"  # Tap motion
    ALL_IN = "All In"  # Sweeping motion

class GestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.previous_positions = deque(maxlen=10)
        self.gesture_patterns = self._initialize_gesture_patterns()
        self.last_gesture = None
        self.gesture_confidence = 0
        self.gesture_stability_threshold = 3

    def _initialize_gesture_patterns(self):
        return {
            BettingGestures.PLACE_BET: {
                'finger_states': [1, 1, 1, 1, 1],  # Closed fist
                'palm_direction': 'down',
                'movement': 'static'
            },
            BettingGestures.INCREASE_BET: {
                'finger_states': [0, 0, 0, 0, 0],  # Open palm
                'palm_direction': 'up',
                'movement': 'up'
            },
            BettingGestures.DOUBLE_DOWN: {
                'finger_states': [1, 0, 0, 1, 1],  # Two fingers
                'palm_direction': 'forward',
                'movement': 'static'
            }
        }

    def _calculate_finger_states(self, hand_landmarks):
        """Calculate finger states using correct MediaPipe landmarks."""
        finger_states = []
        
        # Thumb (special case)
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
        finger_states.append(1 if thumb_tip.y > thumb_ip.y else 0)

        # Index finger
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        finger_states.append(1 if index_tip.y > index_pip.y else 0)

        # Middle finger
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        finger_states.append(1 if middle_tip.y > middle_pip.y else 0)

        # Ring finger
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        finger_states.append(1 if ring_tip.y > ring_pip.y else 0)

        # Pinky
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        finger_states.append(1 if pinky_tip.y > pinky_pip.y else 0)

        return finger_states

    def _detect_palm_direction(self, hand_landmarks):
        """Detect palm direction using wrist and middle finger MCP landmarks."""
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        palm_direction_z = middle_mcp.z - wrist.z
        
        if palm_direction_z > 0.1:
            return 'up'
        elif palm_direction_z < -0.1:
            return 'down'
        return 'forward'

    def _detect_movement_pattern(self, hand_landmarks):
        """Detect hand movement pattern."""
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        current_pos = np.array([wrist.x, wrist.y, wrist.z])
        
        self.previous_positions.append(current_pos)
        
        if len(self.previous_positions) < 3:
            return 'static'
            
        movement = np.array(self.previous_positions[-1]) - np.array(self.previous_positions[0])
        speed = np.linalg.norm(movement)
        
        if speed < 0.02:
            return 'static'
        
        if abs(movement[1]) > abs(movement[0]):
            return 'up' if movement[1] < 0 else 'down'
        else:
            return 'left' if movement[0] < 0 else 'right'

    def detect_gesture(self, hand_landmarks):
        """Detect gestures with confidence scoring."""
        finger_states = self._calculate_finger_states(hand_landmarks)
        palm_direction = self._detect_palm_direction(hand_landmarks)
        movement = self._detect_movement_pattern(hand_landmarks)
        
        # Calculate confidence scores for each gesture
        gesture_scores = {}
        
        for gesture, pattern in self.gesture_patterns.items():
            score = self._calculate_gesture_confidence(
                finger_states, palm_direction, movement, pattern
            )
            gesture_scores[gesture] = score
        
        # Get the gesture with highest confidence
        if gesture_scores:
            best_gesture = max(gesture_scores.items(), key=lambda x: x[1])
            if best_gesture[1] > 0.7:  # Confidence threshold
                if best_gesture[0] == self.last_gesture:
                    self.gesture_confidence += 1
                else:
                    self.gesture_confidence = 1
                
                self.last_gesture = best_gesture[0]
                
                if self.gesture_confidence >= self.gesture_stability_threshold:
                    return best_gesture[0], best_gesture[1]
        
        return None, 0.0

    def _calculate_gesture_confidence(self, finger_states, palm_direction, movement, pattern):
        """Calculate confidence score for a gesture match."""
        finger_conf = sum(1 for a, b in zip(finger_states, pattern['finger_states']) if a == b) / 5
        palm_conf = 1.0 if palm_direction == pattern['palm_direction'] else 0.0
        movement_conf = 1.0 if movement == pattern['movement'] else 0.0
        
        return (0.5 * finger_conf + 0.3 * palm_conf + 0.2 * movement_conf)

    def draw_gesture_info(self, frame, gesture, confidence):
        """Draw gesture information on frame."""
        if gesture:
            # Draw gesture name with confidence
            text = f"{gesture.value}: {confidence:.2%}"
            cv2.putText(frame, text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw hint
            hint_text = self._get_gesture_hint(gesture)
            cv2.putText(frame, hint_text,
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _get_gesture_hint(self, gesture):
        """Get hint text for each gesture."""
        hints = {
            BettingGestures.PLACE_BET: "Make a fist to place bet",
            BettingGestures.INCREASE_BET: "Open palm facing up to increase",
            BettingGestures.DOUBLE_DOWN: "Show two fingers to double down",
            BettingGestures.SPLIT: "Make V shape to split",
            BettingGestures.HOLD: "Open palm forward to hold"
        }
        return hints.get(gesture, "")