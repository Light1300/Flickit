import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import math
from dataclasses import dataclass
from collections import deque
from ultralytics import YOLO

@dataclass
class BallDetection:
    center: Tuple[int, int]
    radius: float
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h

class AdvancedBallDetector:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.tracker = None
        self.tracker_initialized = False
        self.detection_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Kalman filter for smoother tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman_initialized = False
    
    def _default_config(self) -> Dict:
        return {
            'orange_ball': {
                'lower_hsv': [10, 50, 50],
                'upper_hsv': [25, 255, 255]
            },
            'white_ball': {
                'lower_hsv': [0, 0, 200],
                'upper_hsv': [180, 30, 255]
            },
            'min_area': 100,
            'max_area': 10000,
            'min_circularity': 0.6,
            'hough_circles': {
                'dp': 1,
                'min_dist': 50,
                'param1': 100,
                'param2': 30,
                'min_radius': 10,
                'max_radius': 100
            }
        }
    
    def detect_by_color(self, frame: np.ndarray) -> List[BallDetection]:
        """Detect ball using color-based segmentation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        
        # Try orange ball detection
        orange_lower = np.array(self.config['orange_ball']['lower_hsv'])
        orange_upper = np.array(self.config['orange_ball']['upper_hsv'])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
        
        # Try white ball detection
        white_lower = np.array(self.config['white_ball']['lower_hsv'])
        white_upper = np.array(self.config['white_ball']['upper_hsv'])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(orange_mask, white_mask)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.config['min_area'] or area > self.config['max_area']:
                continue
            
            # Check circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity >= self.config['min_circularity']:
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                
                # Calculate bounding box
                bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(contour)
                
                detection = BallDetection(
                    center=center,
                    radius=radius,
                    confidence=min(circularity * 1.5, 1.0),
                    bbox=(bbox_x, bbox_y, bbox_w, bbox_h)
                )
                detections.append(detection)
        
        return detections
    
    def detect_by_hough_circles(self, frame: np.ndarray) -> List[BallDetection]:
        """Detect ball using Hough Circle Transform"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.config['hough_circles']['dp'],
            minDist=self.config['hough_circles']['min_dist'],
            param1=self.config['hough_circles']['param1'],
            param2=self.config['hough_circles']['param2'],
            minRadius=self.config['hough_circles']['min_radius'],
            maxRadius=self.config['hough_circles']['max_radius']
        )
        
        detections = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:

                # Larger circles get higher confidence
                # But this is not working in a Stadium view, detecting random circled shaped white objects.
                # gives best results in closeUP view and fails with multiple Footballs are on screen.
                confidence = min(r / 50.0, 1.0) 
                
                bbox = (x - r, y - r, 2 * r, 2 * r)
                detection = BallDetection(
                    center=(x, y),
                    radius=float(r),
                    confidence=confidence,
                    bbox=bbox
                )
                detections.append(detection)
        
        return detections

    def detect_by_yolo(self, frame: np.ndarray) -> List[BallDetection]:
        """Detect ball using YOLOv8 model"""
        detections = []
        results = self.yolo_model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 32:  # 32 is 'sports ball'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    center = (x1 + w // 2, y1 + h // 2)
                    radius = (w + h) / 4  # Approximate radius
                    confidence = float(box.conf[0])

                    detection = BallDetection(
                        center=center,
                        radius=radius,
                        confidence=confidence,
                        bbox=(x1, y1, w, h)
                    )
                    detections.append(detection)
        return detections
    
    def fuse_detections(self, color_detections: List[BallDetection],
                       hough_detections: List[BallDetection],
                       yolo_detections: List[BallDetection]) -> Optional[BallDetection]:
        """Fuse multiple detection methods"""
        for det in yolo_detections:
            det.confidence = min(det.confidence * 1.2, 1.0)
        all_detections = color_detections + hough_detections + yolo_detections
        
        if not all_detections:
            return None
        
        
        if len(self.detection_history) > 0:
            last_pos = self.detection_history[-1].center
            
            # Calculate distances and weight by confidence
            weighted_detections = []
            for detection in all_detections:
                distance = math.sqrt(
                    (detection.center[0] - last_pos[0])**2 + 
                    (detection.center[1] - last_pos[1])**2
                )
                
                # Penalize distant detections
                distance_weight = max(0, 1 - distance / 200)  # 200 pixels max reasonable movement
                weighted_confidence = detection.confidence * distance_weight
                
                weighted_detections.append((detection, weighted_confidence))
            
            # Sort by weighted confidence
            weighted_detections.sort(key=lambda x: x[1], reverse=True)
            return weighted_detections[0][0] if weighted_detections else None
        
        # If no history, return highest confidence detection
        best_detection = max(all_detections, key=lambda x: x.confidence)
        return best_detection
    
    def track_with_kalman(self, detection: BallDetection) -> Tuple[int, int]:
        """Use Kalman filter for smoother tracking"""
        if not self.kalman_initialized:
            # Initialize Kalman filter
            self.kalman.statePre = np.array([detection.center[0], detection.center[1], 0, 0], 
                                          dtype=np.float32)
            self.kalman.statePost = np.array([detection.center[0], detection.center[1], 0, 0], 
                                           dtype=np.float32)
            self.kalman_initialized = True
            return detection.center
        
        # Predict
        prediction = self.kalman.predict()
        
        # Update with measurement
        measurement = np.array([[detection.center[0]], [detection.center[1]]], dtype=np.float32)
        self.kalman.correct(measurement)
        
        # Return smoothed position
        smoothed_x = int(self.kalman.statePost[0])
        smoothed_y = int(self.kalman.statePost[1])
        
        return (smoothed_x, smoothed_y)
    
    def detect_ball(self, frame: np.ndarray) -> Optional[BallDetection]:
        """Main ball detection method combining multiple approaches"""
        # Method 1: Color-based detection
        color_detections = self.detect_by_color(frame)
        
        # Method 2: Hough circles detection
        hough_detections = self.detect_by_hough_circles(frame)
        
        # Method 3: YOLO detection
        yolo_detections = self.detect_by_yolo(frame)
        
        # Fuse detections
        best_detection = self.fuse_detections(color_detections, hough_detections, yolo_detections)
        
        if best_detection and best_detection.confidence > 0.4:
            # Apply Kalman filtering for smoother tracking
            smoothed_center = self.track_with_kalman(best_detection)
            
            # Update detection with smoothed position
            best_detection.center = smoothed_center
            
            # Add to history
            self.detection_history.append(best_detection)
            
            return best_detection
        
        return None
    
    def calculate_ball_velocity(self) -> Tuple[float, float, float]:
        """Calculate ball velocity and speed from detection history"""
        if len(self.detection_history) < 2:
            return (0.0, 0.0, 0.0)
        
        # Get recent positions
        current_pos = self.detection_history[-1].center
        prev_pos = self.detection_history[-2].center
        
        # Calculate velocity components
        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        speed = math.sqrt(dx*dx + dy*dy)
        
        # Store in velocity history for smoothing
        self.velocity_history.append((dx, dy, speed))
        
        # Return smoothed velocity
        if len(self.velocity_history) >= 3:
            recent_velocities = list(self.velocity_history)[-3:]
            avg_dx = sum(v[0] for v in recent_velocities) / len(recent_velocities)
            avg_dy = sum(v[1] for v in recent_velocities) / len(recent_velocities)
            avg_speed = sum(v[2] for v in recent_velocities) / len(recent_velocities)
            return (avg_dx, avg_dy, avg_speed)
        
        return (dx, dy, speed)
    
    def estimate_rotation_direction(self) -> str:
        """Estimate ball rotation direction based on movement pattern"""
        if len(self.detection_history) < 3:
            return "unknown"
        
        dx, dy, speed = self.calculate_ball_velocity()
        
        if speed < 5:  # Ball is stationary or moving very slowly
            return "none"
        
        # Simple heuristic based on horizontal movement
        if dx > 2:
            return "forward"  # Moving right
        elif dx < -2:
            return "backward"  # Moving left
        else:
            return "none"
    
    def get_ball_trajectory_points(self, num_points: int = 5) -> List[Tuple[int, int]]:
        """Get recent ball trajectory points for visualization"""
        if len(self.detection_history) < num_points:
            return [det.center for det in self.detection_history]
        else:
            return [det.center for det in list(self.detection_history)[-num_points:]]
    
    def reset_tracking(self):
        """Reset all tracking state"""
        self.tracker = None
        self.tracker_initialized = False
        self.detection_history.clear()
        self.velocity_history.clear()
        self.kalman_initialized = False