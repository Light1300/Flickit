import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
from enum import Enum

class LegSide(Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"

@dataclass
class PoseKeypoint:
    x: float
    y: float
    visibility: float
    confidence: float

@dataclass
class PlayerState:
    center_of_mass: Tuple[float, float]
    velocity: float
    acceleration: float
    leg_positions: Dict[str, Tuple[float, float]]
    body_angle: float
    is_running: bool
    is_kicking: bool

class EnhancedPoseAnalyzer:
    def __init__(self):
       
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.pose_history = deque(maxlen=5)
        
        # Touch detection state 
        self.last_ball_contact = {"left": -1, "right": -1}
        self.touch_threshold = 60  # for pixels
        self.touch_cooldown = 10  # for frames
        
    def detect_pose(self, frame: np.ndarray) -> Optional[Any]:
        """Detect pose landmarks using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            self.pose_history.append(results.pose_landmarks)
        
        return results.pose_landmarks
    
    def get_keypoint(self, landmarks, landmark_id: int, frame_shape: Tuple[int, int]) -> Optional[PoseKeypoint]:
        """Extract keypoint with pixel coordinates"""
        if not landmarks or landmark_id >= len(landmarks.landmark):
            return None
        
        landmark = landmarks.landmark[landmark_id]
        h, w = frame_shape[:2]
        
        return PoseKeypoint(
            x=landmark.x * w,
            y=landmark.y * h,
            visibility=landmark.visibility,
            confidence=1.0 - abs(0.5 - landmark.visibility)  
        )
    
    def calculate_center_of_mass(self, landmarks, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        
        if not landmarks:
            return (0, 0)
        
        # body landmarks for center of mass calculation
        key_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]
        
        total_x = 0
        total_y = 0
        valid_points = 0
        
        for landmark_id in key_landmarks:
            keypoint = self.get_keypoint(landmarks, landmark_id, frame_shape)
            if keypoint and keypoint.visibility > 0.5:
                total_x += keypoint.x
                total_y += keypoint.y
                valid_points += 1
        
        if valid_points > 0:
            center_x = total_x / valid_points
            center_y = total_y / valid_points
            return (center_x, center_y)
        
        return (0, 0)
    
    def calculate_velocity_and_acceleration(self, current_position: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate player velocity and acceleration"""
        self.position_history.append(current_position)
        
        if len(self.position_history) < 2:
            return (0.0, 0.0)
        
        #  checking velocity of player
        prev_pos = self.position_history[-2]
        dx = current_position[0] - prev_pos[0]
        dy = current_position[1] - prev_pos[1]
        velocity = math.sqrt(dx*dx + dy*dy)
        
        self.velocity_history.append(velocity)
        
        # Calculate acceleration of player
        acceleration = 0.0
        if len(self.velocity_history) >= 2:
            acceleration = self.velocity_history[-1] - self.velocity_history[-2]
        
        return (velocity, acceleration)
    
    def get_leg_positions(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
        """Get foot and ankle positions for both legs"""
        positions = {}
        
        # Left leg landmarks
        left_ankle = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, frame_shape)
        left_foot = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX, frame_shape)
        left_heel = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.LEFT_HEEL, frame_shape)
        
        # Right leg landmarks
        right_ankle = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE, frame_shape)
        right_foot = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX, frame_shape)
        right_heel = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.RIGHT_HEEL, frame_shape)
        
        # Calculate average position for each leg
        if left_ankle and left_foot:
            left_x = (left_ankle.x + left_foot.x) / 2
            left_y = (left_ankle.y + left_foot.y) / 2
            if left_heel:
                left_x = (left_x + left_heel.x) / 2
                left_y = (left_y + left_heel.y) / 2
            positions['left'] = (left_x, left_y)
        
        if right_ankle and right_foot:
            right_x = (right_ankle.x + right_foot.x) / 2
            right_y = (right_ankle.y + right_foot.y) / 2
            if right_heel:
                right_x = (right_x + right_heel.x) / 2
                right_y = (right_y + right_heel.y) / 2
            positions['right'] = (right_x, right_y)
        
        return positions
    
    def calculate_body_angle(self, landmarks, frame_shape: Tuple[int, int]) -> float:
        """Calculate player's body orientation angle"""
        left_shoulder = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, frame_shape)
        right_shoulder = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, frame_shape)
        
        if not (left_shoulder and right_shoulder):
            return 0.0
        
        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        return angle
    
    def detect_running_state(self, velocity: float, leg_positions: Dict) -> bool:
      
        # Simple heuristic: for velocity is high and we have valid leg positions
        return velocity > 15 and len(leg_positions) == 2
    
    def detect_kicking_motion(self, landmarks, frame_shape: Tuple[int, int]) -> LegSide:
        """Detect kicking motion based on leg extension and velocity"""
        if not landmarks:
            return LegSide.NONE
        
        # check knee if he is kicking or not
        left_knee = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.LEFT_KNEE, frame_shape)
        left_ankle = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.LEFT_ANKLE, frame_shape)
        right_knee = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.RIGHT_KNEE, frame_shape)
        right_ankle = self.get_keypoint(landmarks, self.mp_pose.PoseLandmark.RIGHT_ANKLE, frame_shape)
        
        # Calculate leg extension for both legs
        left_extension = 0
        right_extension = 0
        
        if left_knee and left_ankle:
            left_extension = math.sqrt(
                (left_ankle.x - left_knee.x)**2 + (left_ankle.y - left_knee.y)**2
            )
        
        if right_knee and right_ankle:
            right_extension = math.sqrt(
                (right_ankle.x - right_knee.x)**2 + (right_ankle.y - right_knee.y)**2
            )
        
        # Simple heuristic for checking one leg is significantly more extended
        if left_extension > right_extension * 1.2 and left_extension > 100:
            return LegSide.LEFT
        elif right_extension > left_extension * 1.2 and right_extension > 100:
            return LegSide.RIGHT
        
        return LegSide.NONE
    
    def analyze_ball_contact(self, landmarks, ball_position: Tuple[int, int], 
                           frame_number: int, frame_shape: Tuple[int, int]) -> Optional[LegSide]:
        """Detect which leg is in contact with the ball"""
        if not landmarks or not ball_position:
            return None
        
        leg_positions = self.get_leg_positions(landmarks, frame_shape)
        
        if not leg_positions:
            return None
        
        ball_x, ball_y = ball_position
        
        # Check contact for each leg
        for leg_side in ['left', 'right']:
            if leg_side not in leg_positions:
                continue
            
            leg_x, leg_y = leg_positions[leg_side]
            
            # Calculate distance between leg and ball
            distance = math.sqrt((leg_x - ball_x)**2 + (leg_y - ball_y)**2)
            
            # To Check if within touch threshold and cooldown period has passed
            if (distance < self.touch_threshold and 
                frame_number - self.last_ball_contact[leg_side] > self.touch_cooldown):
                
                self.last_ball_contact[leg_side] = frame_number
                return LegSide.LEFT if leg_side == 'left' else LegSide.RIGHT
        
        return None
    
    def get_player_state(self, landmarks, frame_shape: Tuple[int, int]) -> PlayerState:
        
        if not landmarks:
            return PlayerState(
                center_of_mass=(0, 0),
                velocity=0.0,
                acceleration=0.0,
                leg_positions={},
                body_angle=0.0,
                is_running=False,
                is_kicking=False
            )
        
        
        center_of_mass = self.calculate_center_of_mass(landmarks, frame_shape)
        
      
        velocity, acceleration = self.calculate_velocity_and_acceleration(center_of_mass)
        
     
        leg_positions = self.get_leg_positions(landmarks, frame_shape)
        
        body_angle = self.calculate_body_angle(landmarks, frame_shape)
        
        # Detect states
        is_running = self.detect_running_state(velocity, leg_positions)
        kicking_leg = self.detect_kicking_motion(landmarks, frame_shape)
        is_kicking = kicking_leg != LegSide.NONE
        
        return PlayerState(
            center_of_mass=center_of_mass,
            velocity=velocity,
            acceleration=acceleration,
            leg_positions=leg_positions,
            body_angle=body_angle,
            is_running=is_running,
            is_kicking=is_kicking
        )
    
    def draw_enhanced_pose(self, frame: np.ndarray, landmarks, player_state: PlayerState):

        if not landmarks:
            return frame
        
        # Draw standard pose landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
        
        # Draw center of mass
        com_x, com_y = player_state.center_of_mass
        cv2.circle(frame, (int(com_x), int(com_y)), 8, (255, 0, 255), -1)
        
        # velocity vector
        if player_state.velocity > 5:
            vector_length = min(int(player_state.velocity * 2), 100)
            end_x = int(com_x + vector_length * math.cos(math.radians(player_state.body_angle)))
            end_y = int(com_y + vector_length * math.sin(math.radians(player_state.body_angle)))
            cv2.arrowedLine(frame, (int(com_x), int(com_y)), (end_x, end_y), (255, 255, 0), 3)
        
        # leg positions
        for leg_side, position in player_state.leg_positions.items():
            color = (255, 0, 0) if leg_side == 'left' else (0, 0, 255)
            cv2.circle(frame, (int(position[0]), int(position[1])), 12, color, 3)
        
        return frame