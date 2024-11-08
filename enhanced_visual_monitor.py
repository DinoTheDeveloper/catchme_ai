import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple, Dict
import logging

@dataclass
class TableZone:
    name: str
    corners: np.ndarray
    active: bool = False
    last_activity: datetime = None
    gesture_history: List[str] = None

    def __post_init__(self):
        self.gesture_history = []

class BettingMonitor:
    def __init__(self):
        # Initialize logging
        logging.basicConfig(
            filename='table_monitor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize table zones with default positions
        self.table_zones = self._initialize_table_zones()
        
        # Activity tracking
        self.suspicious_activity_threshold = 3
        self.last_alert_time = datetime.now()
        self.alert_cooldown = 30  # seconds
        
        # Visualization settings
        self.show_debug = False
        self.hand_trail = []
        self.max_trail_length = 20

    def _initialize_table_zones(self) -> Dict[str, TableZone]:
        """Initialize table zones with betting areas."""
        return {
            'main_bet': TableZone(
                name='Main Bet',
                corners=np.array([[200, 200], [400, 200], [400, 400], [200, 400]])
            ),
            'side_bet': TableZone(
                name='Side Bet',
                corners=np.array([[450, 200], [650, 200], [650, 400], [450, 400]])
            ),
            'insurance': TableZone(
                name='Insurance',
                corners=np.array([[700, 200], [900, 200], [900, 400], [700, 400]])
            )
        }

    def _point_in_zone(self, point: np.ndarray, zone_corners: np.ndarray) -> bool:
        """Check if a point is within a table zone."""
        return cv2.pointPolygonTest(zone_corners, tuple(point), False) >= 0

    def _log_suspicious_activity(self, message: str):
        """Log suspicious activity with timestamp."""
        current_time = datetime.now()
        if (current_time - self.last_alert_time).total_seconds() > self.alert_cooldown:
            logging.warning(f"SUSPICIOUS ACTIVITY: {message}")
            self.last_alert_time = current_time

    def _draw_table_zones(self, frame: np.ndarray):
        """Draw enhanced table zones with activity indicators."""
        overlay = frame.copy()
        
        for zone in self.table_zones.values():
            # Create semi-transparent fill
            alpha = 0.3
            fill_color = (0, 255, 0) if zone.active else (0, 0, 255)
            cv2.fillPoly(overlay, [zone.corners], fill_color)
            
            # Draw zone borders
            cv2.polylines(frame, [zone.corners], True, fill_color, 2)
            
            # Add zone labels with background
            center = np.mean(zone.corners, axis=0).astype(int)
            text_size = cv2.getTextSize(
                zone.name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (center[0] - text_size[0]//2 - 5, center[1] - text_size[1] - 5),
                (center[0] + text_size[0]//2 + 5, center[1] + 5),
                fill_color,
                -1
            )
            
            # Draw zone name
            cv2.putText(
                frame,
                zone.name,
                (center[0] - text_size[0]//2, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        
        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process frame and detect hand activities."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Draw table zones
        self._draw_table_zones(frame)
        
        suspicious_activity = False
        active_zones = set()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2),
                    self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1)
                )
                
                # Get hand position
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                hand_pos = np.array([
                    int(wrist.x * frame.shape[1]),
                    int(wrist.y * frame.shape[0])
                ])
                
                # Update hand trail
                self.hand_trail.append(tuple(hand_pos))
                if len(self.hand_trail) > self.max_trail_length:
                    self.hand_trail.pop(0)
                
                # Draw hand trail
                for i in range(1, len(self.hand_trail)):
                    thickness = int((i / len(self.hand_trail)) * 3) + 1
                    cv2.line(frame,
                            self.hand_trail[i-1],
                            self.hand_trail[i],
                            (255, 0, 255),
                            thickness)
                
                # Check zones for activity
                for zone_name, zone in self.table_zones.items():
                    if self._point_in_zone(hand_pos, zone.corners):
                        zone.active = True
                        active_zones.add(zone_name)
                        
                        # Check for suspicious patterns
                        if zone.last_activity:
                            time_diff = (datetime.now() - zone.last_activity).total_seconds()
                            if time_diff < 0.5:
                                suspicious_activity = True
                                self._log_suspicious_activity(
                                    f"Rapid movement detected in {zone.name}"
                                )
                        
                        zone.last_activity = datetime.now()
                    else:
                        zone.active = False
        
        # Draw debug information if enabled
        if self.show_debug:
            self._draw_debug_info(frame, active_zones)
        
        return frame, suspicious_activity

    def _draw_debug_info(self, frame: np.ndarray, active_zones: set):
        """Draw debug information on frame."""
        y_offset = 30
        cv2.putText(
            frame,
            f"Active Zones: {', '.join(active_zones) if active_zones else 'None'}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    def run(self):
        """Run the betting monitor."""
        print("Starting Table Betting Monitor...")
        print("Controls:")
        print("  - Press 'd' to toggle debug information")
        print("  - Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logging.error("Failed to open camera")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, suspicious = self.process_frame(frame)
                
                if suspicious:
                    cv2.putText(
                        processed_frame,
                        "SUSPICIOUS ACTIVITY DETECTED",
                        (10, 700),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow('Table Betting Monitor', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.show_debug = not self.show_debug
                    print(f"Debug mode: {'ON' if self.show_debug else 'OFF'}")

        except Exception as e:
            logging.error(f"Error during monitoring: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logging.info("Table monitor stopped")

if __name__ == "__main__":
    monitor = BettingMonitor()
    monitor.run()