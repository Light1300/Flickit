"""
Requirements:
pip install opencv-python numpy mediapipe scipy matplotlib ultralytics

Usage:
python main.py --video path/to/video.mp4
"""
import cv2
import numpy as np
import argparse
import json
from typing import Dict, List, Tuple, Optional, Any
import math
from collections import deque
from dataclasses import dataclass
import time

from pose_analyzer import EnhancedPoseAnalyzer, LegSide, PlayerState
from ball_detector import AdvancedBallDetector, BallDetection

@dataclass
class TouchEvent:
    frame_number: int
    timestamp: float
    leg: str  # 'left' or 'right'
    ball_position: Tuple[int, int]
    player_velocity: float
    confidence: float

@dataclass
class BallState:
    position: Tuple[int, int]
    velocity: Tuple[float, float]
    rotation_direction: str  
    speed: float

class FootballAnalyzer:
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold

       
        self.pose_analyzer = EnhancedPoseAnalyzer()
        self.ball_detector = AdvancedBallDetector()

        
        self.left_touches = 0
        self.right_touches = 0
        self.touch_events: List[TouchEvent] = []

   
        self.overlay_alpha = 0.8

    def draw_overlay(self, frame: np.ndarray, landmarks: Optional[Any], ball_state: Optional[BallState], player_state: PlayerState):
      
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw pose landmarks
        if landmarks:
            self.pose_analyzer.draw_enhanced_pose(overlay, landmarks, player_state)
        
        # Draw ball tracking
        if ball_state:
            x, y = ball_state.position
            cv2.circle(overlay, (x, y), int(self.ball_detector.detection_history[-1].radius), (0, 255, 255), 3)
            cv2.putText(overlay, f"Ball Speed: {ball_state.speed:.1f}", 
                       (x-50, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, f"Rotation: {ball_state.rotation_direction}", 
                       (x-50, y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw statistics panel
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        stats_text = [
            f"Left Leg Touches: {self.left_touches}",
            f"Right Leg Touches: {self.right_touches}",
            f"Total Touches: {self.left_touches + self.right_touches}",
            f"Player Velocity: {player_state.velocity:.2f}",
        ]
        
        if ball_state:
            stats_text.extend([
                f"Ball Speed: {ball_state.speed:.1f}",
                f"Ball Rotation: {ball_state.rotation_direction}"
            ])
        
        for i, text in enumerate(stats_text):
            cv2.putText(panel, text, (20, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
   
        result = cv2.addWeighted(frame, 1 - self.overlay_alpha, overlay, self.overlay_alpha, 0)
        result = np.vstack([result, panel])
        
        return result
    
    def process_video(self, video_path: str, output_path: str = "output.mp4"):
       
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
    
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 200))
        
        frame_count = 0
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
    
            landmarks = self.pose_analyzer.detect_pose(frame)
            player_state = self.pose_analyzer.get_player_state(landmarks, frame.shape)
            
        
            ball_detection = self.ball_detector.detect_ball(frame)
            ball_state = None
            if ball_detection:
                ball_dx, ball_dy, ball_speed = self.ball_detector.calculate_ball_velocity()
                rotation = self.ball_detector.estimate_rotation_direction()
                ball_state = BallState(
                    position=ball_detection.center,
                    velocity=(ball_dx, ball_dy),
                    rotation_direction=rotation,
                    speed=ball_speed
                )
            
            # Detect touches
            if landmarks and ball_state:
                touch_leg_side = self.pose_analyzer.analyze_ball_contact(
                    landmarks, ball_state.position, frame_count, frame.shape
                )
                
                if touch_leg_side and touch_leg_side != LegSide.NONE:
                    if touch_leg_side == LegSide.LEFT:
                        self.left_touches += 1
                    else:
                        self.right_touches += 1

                    touch_event = TouchEvent(
                        frame_number=frame_count,
                        timestamp=frame_count / fps,
                        leg=touch_leg_side.value,
                        ball_position=ball_state.position,
                        player_velocity=player_state.velocity,
                        confidence=0.9  
                    )
                    self.touch_events.append(touch_event)
                    print(f"Touch detected! {touch_leg_side.value.capitalize()} leg at frame {frame_count}")
            
            # Draw overlay
            annotated_frame = self.draw_overlay(frame, landmarks, ball_state, player_state)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        print(f"\nProcessing complete!")
        print(f"Left leg touches: {self.left_touches}")
        print(f"Right leg touches: {self.right_touches}")
        print(f"Total touches: {self.left_touches + self.right_touches}")
        print(f"Output saved to: {output_path}")
        
   
        self.save_analysis_data(output_path.replace('.mp4', '_analysis.json'))
    
    def save_analysis_data(self, filepath: str):
   
        data = {
            "summary": {
                "left_touches": self.left_touches,
                "right_touches": self.right_touches,
                "total_touches": self.left_touches + self.right_touches
            },
            "touch_events": [
                {
                    "frame": event.frame_number,
                    "timestamp": event.timestamp,
                    "leg": event.leg,
                    "ball_position": event.ball_position,
                    "player_velocity": event.player_velocity,
                    "confidence": event.confidence
                }
                for event in self.touch_events
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Analysis data saved to: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Football Touch Detection System')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', default='output.mp4', help='Path to output video file')
    parser.add_argument('--confidence', type=float, default=0.7, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    try:
        analyzer = FootballAnalyzer(confidence_threshold=args.confidence)
        analyzer.process_video(args.video, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())