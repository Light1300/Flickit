#Advanced Visualization Module for dynamic overlays and statistical displays for football analysis


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass
from collections import deque

@dataclass
class TouchStats:
    left_touches: int
    right_touches: int
    touch_events: List[Dict]
    avg_touch_velocity: float
    max_velocity: float

@dataclass
class BallStats:
    current_speed: float
    max_speed: float
    avg_speed: float
    rotation_direction: str
    trajectory_points: List[Tuple[int, int]]

class AdvancedVisualizer:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Colors (BGR format)
        self.colors = {
            'left_leg': (255, 0, 0),      
            'right_leg': (0, 0, 255),     
            'ball': (0, 255, 255),        
            'trajectory': (255, 0, 255),   
            'text': (255, 255, 255),       # White
            'background': (0, 0, 0),       
            'success': (0, 255, 0),        # Green
            'warning': (0, 165, 255),      # Orange
        }
        
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        
        self.panel_height = 200
        self.panel_width = frame_width
        
        
        self.velocity_history = deque(maxlen=100)
        self.touch_velocity_history = deque(maxlen=50)
    
    def create_stats_panel(self, touch_stats: TouchStats, ball_stats: BallStats, 
                          player_velocity: float) -> np.ndarray:
        """Create a comprehensive statistics panel"""
        panel = np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
        
        # Background 
        for i in range(self.panel_height):
            alpha = i / self.panel_height
            color_val = int(30 * (1 - alpha))
            panel[i, :] = (color_val, color_val, color_val)
        
        # Left column - Touch Statistics
        left_col_x = 20
        y_offset = 30
        line_spacing = 25
        
        stats_left = [
            f"TOUCH STATISTICS",
            f"Left Leg: {touch_stats.left_touches:02d}",
            f"Right Leg: {touch_stats.right_touches:02d}",
            f"Total: {touch_stats.left_touches + touch_stats.right_touches:02d}",
            f"Avg Touch Speed: {touch_stats.avg_touch_velocity:.1f}",
            f"Max Touch Speed: {touch_stats.max_velocity:.1f}"
        ]
        
        for i, text in enumerate(stats_left):
            y_pos = y_offset + i * line_spacing
            color = self.colors['warning'] if i == 0 else self.colors['text']
            weight = 2 if i == 0 else 1
            
            cv2.putText(panel, text, (left_col_x, y_pos), self.font, 
                       self.font_scale, color, weight)
        
        # Middle column - Ball Statistics
        mid_col_x = 280
        
        stats_mid = [
            f"BALL ANALYSIS",
            f"Current Speed: {ball_stats.current_speed:.1f}",
            f"Max Speed: {ball_stats.max_speed:.1f}",
            f"Avg Speed: {ball_stats.avg_speed:.1f}",
            f"Rotation: {ball_stats.rotation_direction.title()}",
            f"Trajectory Points: {len(ball_stats.trajectory_points)}"
        ]
        
        for i, text in enumerate(stats_mid):
            y_pos = y_offset + i * line_spacing
            color = self.colors['warning'] if i == 0 else self.colors['text']
            weight = 2 if i == 0 else 1
            
            cv2.putText(panel, text, (mid_col_x, y_pos), self.font, 
                       self.font_scale, color, weight)
        
        # Right column - Player Statistics
        right_col_x = 540
        
        # Update velocity
        self.velocity_history.append(player_velocity)
        avg_velocity = sum(self.velocity_history) / len(self.velocity_history)
        max_velocity = max(self.velocity_history) if self.velocity_history else 0
        
        stats_right = [
            f"PLAYER MOTION",
            f"Current Speed: {player_velocity:.1f}",
            f"Avg Speed: {avg_velocity:.1f}",
            f"Max Speed: {max_velocity:.1f}",
            f"Status: {'RUNNING' if player_velocity > 15 else 'WALKING' if player_velocity > 5 else 'STATIC'}",
            f"Motion Samples: {len(self.velocity_history)}"
        ]
        
        for i, text in enumerate(stats_right):
            y_pos = y_offset + i * line_spacing
            color = self.colors['warning'] if i == 0 else self.colors['text']
            weight = 2 if i == 0 else 1
            
            cv2.putText(panel, text, (right_col_x, y_pos), self.font, 
                       self.font_scale, color, weight)
        
        # performance indicators
        self._draw_performance_bars(panel)
        
        return panel
    
    def _draw_performance_bars(self, panel: np.ndarray):
        """Draw performance indicator bars"""
        bar_width = 150
        bar_height = 10
        bar_x = self.panel_width - bar_width - 20
        bar_y_start = 30
        
        # Touch accuracy bar 
        touch_accuracy = min(1.0, len(self.touch_velocity_history) / 50.0)
        self._draw_progress_bar(panel, (bar_x, bar_y_start), bar_width, bar_height, 
                               touch_accuracy, "Touch Accuracy", self.colors['success'])
        
        # Ball tracking confidence 
        tracking_confidence = 0.85  # This would come from actual tracking confidence
        self._draw_progress_bar(panel, (bar_x, bar_y_start + 40), bar_width, bar_height, 
                               tracking_confidence, "Tracking Quality", self.colors['ball'])
        
        # Activity level
        activity_level = min(1.0, (sum(self.velocity_history) / len(self.velocity_history) / 30.0) 
                           if self.velocity_history else 0)
        self._draw_progress_bar(panel, (bar_x, bar_y_start + 80), bar_width, bar_height, 
                               activity_level, "Activity Level", self.colors['warning'])
    
    def _draw_progress_bar(self, panel: np.ndarray, position: Tuple[int, int], 
                          width: int, height: int, progress: float, 
                          label: str, color: Tuple[int, int, int]):
        """Draw a progress bar with label"""
        x, y = position
        
        # Background
        cv2.rectangle(panel, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        # Progress
        fill_width = int(width * progress)
        cv2.rectangle(panel, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(panel, (x, y), (x + width, y + height), self.colors['text'], 1)
        
        # Label
        cv2.putText(panel, f"{label}: {progress*100:.0f}%", 
                   (x, y - 5), self.font, 0.4, self.colors['text'], 1)
    
    def draw_ball_trajectory(self, frame: np.ndarray, trajectory_points: List[Tuple[int, int]], 
                           current_position: Optional[Tuple[int, int]] = None):
        """Draw ball trajectory with fade effect"""
        if len(trajectory_points) < 2:
            return frame
        
        # Draw trajectory line with fading effect
        for i in range(1, len(trajectory_points)):
            alpha = i / len(trajectory_points)  # Fade from old to new
            color_intensity = int(255 * alpha)
            
            pt1 = trajectory_points[i-1]
            pt2 = trajectory_points[i]
            
            # Create faded color
            faded_color = (color_intensity // 2, color_intensity, color_intensity)
            
            cv2.line(frame, pt1, pt2, faded_color, 2)
        
        # Draw trajectory points
        for i, point in enumerate(trajectory_points):
            alpha = i / len(trajectory_points)
            radius = int(3 + 2 * alpha)
            color_intensity = int(255 * alpha)
            point_color = (color_intensity // 3, color_intensity // 2, color_intensity)
            
            cv2.circle(frame, point, radius, point_color, -1)
        
        # Highlight current position
        if current_position:
            cv2.circle(frame, current_position, 15, self.colors['ball'], 3)
            cv2.circle(frame, current_position, 8, self.colors['ball'], -1)
        
        return frame
    
    def draw_touch_indicators(self, frame: np.ndarray, leg_positions: Dict[str, Tuple[float, float]], 
                            recent_touches: Dict[str, int]):
        """Draw touch indicators around legs"""
        for leg_side, position in leg_positions.items():
            x, y = int(position[0]), int(position[1])
            color = self.colors['left_leg'] if leg_side == 'left' else self.colors['right_leg']
            
            # Base circle
            cv2.circle(frame, (x, y), 15, color, 2)
            
            # Highlight recent touches
            if leg_side in recent_touches and recent_touches[leg_side] > 0:
                # Pulsing effect for recent touches
                pulse_radius = 20 + int(5 * math.sin(recent_touches[leg_side] * 0.5))
                cv2.circle(frame, (x, y), pulse_radius, color, 1)
                
                # Touch counter
                cv2.putText(frame, f"{recent_touches[leg_side]}", 
                           (x - 10, y - 25), self.font, 0.8, color, 2)
        
        return frame
    
    def draw_velocity_vector(self, frame: np.ndarray, position: Tuple[int, int], 
                           velocity: Tuple[float, float], scale: float = 2.0):
        """Draw velocity """
        if not velocity or (velocity[0] == 0 and velocity[1] == 0):
            return frame
        
        start_point = position
        dx, dy = velocity
        
        # Scale and limit vector length
        magnitude = math.sqrt(dx*dx + dy*dy)
        if magnitude > 0:
            max_length = 100
            actual_length = min(magnitude * scale, max_length)
            
            # Calculate end point
            unit_x = dx / magnitude
            unit_y = dy / magnitude
            end_point = (
                int(start_point[0] + unit_x * actual_length),
                int(start_point[1] + unit_y * actual_length)
            )
            
            # Draw arrow
            cv2.arrowedLine(frame, start_point, end_point, self.colors['success'], 3, tipLength=0.3)
            
            # Draw magnitude text
            mid_point = (
                int((start_point[0] + end_point[0]) / 2),
                int((start_point[1] + end_point[1]) / 2) - 10
            )
            cv2.putText(frame, f"{magnitude:.1f}", mid_point, self.font, 0.5, self.colors['text'], 1)
        
        return frame
    
    def create_heatmap_overlay(self, frame: np.ndarray, touch_positions: List[Tuple[int, int]], 
                              alpha: float = 0.6) -> np.ndarray:
        """Create a heatmap overlay showing touch density"""
        if not touch_positions:
            return frame
        
        # Create heatmap
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        # Add Gaussian blobs for each touch position
        for pos in touch_positions:
            x, y = pos
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                # Create Gaussian kernel
                kernel_size = 50
                sigma = kernel_size / 3
                
                # Calculate bounds
                x_min = max(0, x - kernel_size)
                x_max = min(frame.shape[1], x + kernel_size)
                y_min = max(0, y - kernel_size)
                y_max = min(frame.shape[0], y + kernel_size)
                
                # Create meshgrid for the region
                xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
                
                # Calculate Gaussian values
                gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
                
                # Add to heatmap
                heatmap[y_min:y_max, x_min:x_max] += gaussian
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert to color
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original frame
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def draw_performance_metrics(self, frame: np.ndarray, metrics: Dict):
        """Draw real-time performance metrics"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Metrics box
        box_height = 120
        box_width = 300
        box_x = frame.shape[1] - box_width - 20
        box_y = 20
        
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 0, 0), -1)
        
        # Blend overlay
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw metrics text
        y_offset = box_y + 25
        line_height = 20
        
        metric_texts = [
            f"FPS: {metrics.get('fps', 0):.1f}",
            f"Processing Time: {metrics.get('process_time', 0):.1f}ms",
            f"Detection Conf: {metrics.get('detection_conf', 0):.2f}",
            f"Tracking Quality: {metrics.get('tracking_quality', 0):.2f}",
            f"Frame: {metrics.get('frame_number', 0)}"
        ]
        
        for i, text in enumerate(metric_texts):
            cv2.putText(frame, text, (box_x + 10, y_offset + i * line_height), 
                       self.font, 0.5, self.colors['text'], 1)
        
        return frame
    
    def create_analysis_dashboard(self, touch_stats: TouchStats, ball_stats: BallStats, 
                                player_velocity: float) -> np.ndarray:
        """Create a comprehensive analysis dashboard using matplotlib"""
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.patch.set_facecolor('black')
        
        # Plot 1: Touch Distribution
        legs = ['Left Leg', 'Right Leg']
        touches = [touch_stats.left_touches, touch_stats.right_touches]
        colors_plot = ['#FF4444', '#4444FF']
        
        ax1.pie(touches, labels=legs, colors=colors_plot, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Touch Distribution', color='white', fontsize=14)
        ax1.set_facecolor('black')
        
        # Plot 2: Velocity Timeline
        if len(self.velocity_history) > 1:
            ax2.plot(list(self.velocity_history), color='#44FF44', linewidth=2)
            ax2.fill_between(range(len(self.velocity_history)), list(self.velocity_history), 
                            alpha=0.3, color='#44FF44')
            ax2.set_title('Player Velocity Over Time', color='white', fontsize=14)
            ax2.set_xlabel('Time (frames)', color='white')
            ax2.set_ylabel('Velocity', color='white')
            ax2.set_facecolor('black')
            ax2.tick_params(colors='white')
        
        # Plot 3: Ball Speed Analysis
        if ball_stats.trajectory_points:
            speeds = [ball_stats.current_speed, ball_stats.max_speed, ball_stats.avg_speed]
            labels = ['Current', 'Maximum', 'Average']
            bars = ax3.bar(labels, speeds, color=['#FFFF44', '#FF44FF', '#44FFFF'])
            ax3.set_title('Ball Speed Analysis', color='white', fontsize=14)
            ax3.set_ylabel('Speed (pixels/frame)', color='white')
            ax3.set_facecolor('black')
            ax3.tick_params(colors='white')
            
            # Add value labels on bars
            for bar, speed in zip(bars, speeds):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{speed:.1f}', ha='center', va='bottom', color='white')
        
        # Plot 4: Performance Summary
        categories = ['Touches', 'Ball Speed', 'Player Speed', 'Activity']
        values = [
            min(100, (touch_stats.left_touches + touch_stats.right_touches) * 10),
            min(100, ball_stats.avg_speed * 2),
            min(100, player_velocity * 3),
            min(100, len(self.velocity_history) * 2)
        ]
        
        ax4.barh(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax4.set_title('Performance Overview', color='white', fontsize=14)
        ax4.set_xlabel('Score', color='white')
        ax4.set_xlim(0, 100)
        ax4.set_facecolor('black')
        ax4.tick_params(colors='white')
        
        # Convert plot to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        
        # Convert to OpenCV format
        dashboard = np.frombuffer(raw_data, dtype=np.uint8)
        dashboard = dashboard.reshape((int(size[1]), int(size[0]), 3))
        dashboard = cv2.cvtColor(dashboard, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        return dashboard
    
    def create_final_overlay(self, frame: np.ndarray, touch_stats: TouchStats, 
                           ball_stats: BallStats, player_velocity: float,
                           leg_positions: Dict, recent_touches: Dict,
                           current_ball_pos: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Create the complete overlay combining all visualization elements"""
        
        # Start with the original frame
        result = frame.copy()
        
        # Draw ball trajectory
        if ball_stats.trajectory_points:
            result = self.draw_ball_trajectory(result, ball_stats.trajectory_points, current_ball_pos)
        
        # Draw touch indicators
        if leg_positions:
            result = self.draw_touch_indicators(result, leg_positions, recent_touches)
        
        # Draw velocity vectors
        if current_ball_pos and ball_stats.current_speed > 0:
            # Estimate ball velocity direction from trajectory
            if len(ball_stats.trajectory_points) >= 2:
                last_pos = ball_stats.trajectory_points[-1]
                prev_pos = ball_stats.trajectory_points[-2]
                ball_velocity = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])
                result = self.draw_velocity_vector(result, current_ball_pos, ball_velocity, 3.0)
        
        # Create and attach stats panel
        stats_panel = self.create_stats_panel(touch_stats, ball_stats, player_velocity)
        result = np.vstack([result, stats_panel])
        
        # Add performance metrics
        metrics = {
            'fps': 30.0,  # Would be calculated from actual processing
            'process_time': 33.3,  # Would be measured
            'detection_conf': 0.85,
            'tracking_quality': 0.92,
            'frame_number': len(self.velocity_history)
        }
        result = self.draw_performance_metrics(result, metrics)
        
        return result
    
    def save_analysis_report(self, filepath: str, touch_stats: TouchStats, 
                           ball_stats: BallStats):
        """Save a comprehensive analysis report as image"""
        dashboard = self.create_analysis_dashboard(touch_stats, ball_stats, 
                                                 self.velocity_history[-1] if self.velocity_history else 0)
        #finally and donee yayy
        cv2.imwrite(filepath, dashboard)
        print(f"Analysis report saved to: {filepath}")