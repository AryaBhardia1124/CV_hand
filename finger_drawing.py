import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FingerPainter:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        
        # MediaPipe hands with higher tracking confidence for smoother tracking
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Drawing canvas
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Position smoothing - using exponential moving average
        self.smoothed_x = None
        self.smoothed_y = None
        self.smoothing_factor = 0.7  # Higher = more smoothing (0-1)
        
        # Position history for interpolation
        self.position_history = deque(maxlen=5)
        
        # Drawing settings
        self.drawing = False
        self.color = (0, 255, 0)  # Green by default
        self.brush_size = 12
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (255, 255, 255) # White
        ]
        self.current_color_index = 0
        
        # Gesture state tracking with hysteresis to prevent flickering
        self.fingers_joined_state = False
        self.fingers_joined_counter = 0
        self.fingers_joined_threshold = 3  # Frames needed to confirm state change
        
    def find_hands(self, img):
        """Detect hands and return landmarks"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        return results
    
    def are_fingers_joined(self, hand_landmarks, img):
        """Check if index and middle fingers are joined together with smoothing"""
        h, w, c = img.shape
        
        # Get index finger tip (landmark 8)
        index_lm = hand_landmarks.landmark[8]
        index_x = int(index_lm.x * w)
        index_y = int(index_lm.y * h)
        
        # Get middle finger tip (landmark 12)
        middle_lm = hand_landmarks.landmark[12]
        middle_x = int(middle_lm.x * w)
        middle_y = int(middle_lm.y * h)
        
        # Calculate distance between fingertips
        distance = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
        
        # Adaptive threshold based on hand size
        wrist_to_index = np.sqrt(
            (index_x - int(hand_landmarks.landmark[0].x * w))**2 + 
            (index_y - int(hand_landmarks.landmark[0].y * h))**2
        )
        threshold = max(30, wrist_to_index * 0.15)  # Adaptive threshold
        
        joined = distance < threshold
        
        # Use hysteresis to prevent flickering
        if joined:
            self.fingers_joined_counter += 1
        else:
            self.fingers_joined_counter -= 1
        
        self.fingers_joined_counter = max(0, min(self.fingers_joined_threshold * 2, self.fingers_joined_counter))
        
        if self.fingers_joined_counter >= self.fingers_joined_threshold:
            self.fingers_joined_state = True
        elif self.fingers_joined_counter <= 0:
            self.fingers_joined_state = False
        
        # Calculate midpoint for drawing position
        mid_x = (index_x + middle_x) // 2
        mid_y = (index_y + middle_y) // 2
        
        return self.fingers_joined_state, (index_x, index_y), (middle_x, middle_y), (mid_x, mid_y), distance
    
    def smooth_position(self, x, y):
        """Apply exponential moving average smoothing to position"""
        if self.smoothed_x is None or self.smoothed_y is None:
            # Initialize smoothed position
            self.smoothed_x = float(x)
            self.smoothed_y = float(y)
        else:
            # Apply exponential moving average
            self.smoothed_x = self.smoothing_factor * self.smoothed_x + (1 - self.smoothing_factor) * x
            self.smoothed_y = self.smoothing_factor * self.smoothed_y + (1 - self.smoothing_factor) * y
        
        return int(self.smoothed_x), int(self.smoothed_y)
    
    def interpolate_points(self, prev_point, curr_point, num_points=3):
        """Generate interpolated points between two positions for smoother lines"""
        if prev_point is None or curr_point is None:
            return [curr_point]
        
        px, py = prev_point
        cx, cy = curr_point
        
        # Calculate distance
        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
        
        # Only interpolate if distance is significant
        if dist < 5:
            return [curr_point]
        
        points = []
        for i in range(num_points + 1):
            t = i / (num_points + 1)
            x = int(px + t * (cx - px))
            y = int(py + t * (cy - py))
            points.append((x, y))
        
        return points
    
    def fingers_up(self, hand_landmarks):
        """Check which fingers are up"""
        fingers = []
        
        # Thumb (check x-coordinate for right hand, y for left)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        
        # Determine hand orientation
        wrist = hand_landmarks.landmark[0]
        index_mcp = hand_landmarks.landmark[5]
        
        if index_mcp.x > wrist.x:  # Right hand
            if thumb_tip.x > thumb_ip.x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            if thumb_tip.x < thumb_ip.x:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Other fingers (check y-coordinate)
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]  # PIP joints
        
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def draw_ui(self, img):
        """Draw UI elements on the screen"""
        # Color palette at the top
        color_width = 1280 // len(self.colors)
        for i, color in enumerate(self.colors):
            x1 = i * color_width
            x2 = (i + 1) * color_width
            cv2.rectangle(img, (x1, 0), (x2, 50), color, -1)
            if i == self.current_color_index:
                cv2.rectangle(img, (x1, 0), (x2, 50), (255, 255, 255), 3)
        
        # Instructions
        cv2.putText(img, "Index + Middle fingers joined: Draw | Thumb up: Change color | All fingers up: Clear", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(img, "Press 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    
    def run(self):
        """Main loop"""
        prev_draw_point = None
        
        while True:
            success, img = self.cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1)
            img = cv2.resize(img, (1280, 720))
            
            # Find hands
            results = self.find_hands(img)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
                    
                    # Check if index and middle fingers are joined
                    fingers_joined, index_pos, middle_pos, draw_pos, distance = self.are_fingers_joined(hand_landmarks, img)
                    
                    # Smooth the drawing position
                    smooth_x, smooth_y = self.smooth_position(draw_pos[0], draw_pos[1])
                    smooth_pos = (smooth_x, smooth_y)
                    
                    # Check which fingers are up
                    fingers = self.fingers_up(hand_landmarks)
                    
                    # Drawing mode: Index and middle fingers joined together
                    if fingers_joined:
                        if not self.drawing:
                            self.drawing = True
                            prev_draw_point = smooth_pos
                            self.position_history.clear()
                        
                        # Add current position to history
                        self.position_history.append(smooth_pos)
                        
                        # Draw on canvas with interpolation for smoother lines
                        if prev_draw_point is not None:
                            # Generate interpolated points
                            interpolated = self.interpolate_points(prev_draw_point, smooth_pos, num_points=2)
                            
                            # Draw lines between interpolated points
                            for i in range(len(interpolated) - 1):
                                pt1 = interpolated[i]
                                pt2 = interpolated[i + 1]
                                cv2.line(self.canvas, pt1, pt2, self.color, self.brush_size, cv2.LINE_AA)
                        
                        prev_draw_point = smooth_pos
                        
                        # Visual indicator - draw circle at smoothed position
                        cv2.circle(img, smooth_pos, self.brush_size, self.color, -1, cv2.LINE_AA)
                        cv2.circle(img, smooth_pos, self.brush_size + 5, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        # Visual feedback: draw line connecting the two fingers
                        cv2.line(img, index_pos, middle_pos, (0, 255, 255), 2, cv2.LINE_AA)
                        
                        # Draw small circles on fingertips
                        cv2.circle(img, index_pos, 8, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(img, middle_pos, 8, (0, 255, 255), -1, cv2.LINE_AA)
                    else:
                        if self.drawing:
                            self.drawing = False
                            prev_draw_point = None
                            self.position_history.clear()
                            # Reset smoothed position when stopping
                            self.smoothed_x = None
                            self.smoothed_y = None
                    
                    # Change color: Thumb up (with debouncing)
                    if fingers[0] == 1 and sum(fingers[1:]) <= 1:  # Allow thumb + one other finger
                        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
                        self.color = self.colors[self.current_color_index]
                        cv2.waitKey(200)  # Debounce
                    
                    # Clear canvas: All fingers up
                    if sum(fingers) == 5:
                        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                        prev_draw_point = None
                        self.position_history.clear()
                        cv2.waitKey(300)  # Debounce
            
            else:
                # No hand detected - reset drawing state
                if self.drawing:
                    self.drawing = False
                    prev_draw_point = None
                    self.position_history.clear()
                    self.smoothed_x = None
                    self.smoothed_y = None
            
            # Blend canvas with camera feed
            img = cv2.addWeighted(img, 0.5, self.canvas, 0.5, 0)
            
            # Draw UI
            self.draw_ui(img)
            
            # Show image
            cv2.imshow("Finger Painter", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    painter = FingerPainter()
    painter.run()