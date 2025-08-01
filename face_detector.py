"""
Face Detection and Comparison Tool

This module implements a real-time face detection system that captures video from Mac camera,
detects faces in the video stream, and compares them with reference images stored locally.
The system uses OpenCV for camera handling and face_recognition library for face detection and comparison.
"""

import cv2
import face_recognition
import numpy as np
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def log_with_timestamp(message: str, prefix: str = "") -> None:
    """
    Print log message with timestamp
    
    Args:
        message: The log message to print
        prefix: Optional prefix for the log message (e.g., "[INFO]", "[ERROR]")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {prefix}{message}" if prefix else f"[{timestamp}] {message}"
    print(full_message)


class FaceDetector:
    """
    Face detection and comparison system
    
    This class handles camera operations, face detection, and face comparison
    with reference images stored in the face directory.
    """
    
    def __init__(self, face_dir: str = "face", tolerance: float = 0.6, 
                 trigger_script: Optional[str] = None, cooldown_seconds: int = 10):
        """
        Initialize the face detector
        
        Args:
            face_dir: Directory containing reference face images
            tolerance: Face comparison tolerance (lower = more strict)
            trigger_script: Path to script to execute when face is matched
            cooldown_seconds: Minimum seconds between script executions
        """
        self.face_dir = Path(face_dir)
        self.tolerance = tolerance
        self.trigger_script = trigger_script
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0
        self.known_faces = []
        self.known_names = []
        self.load_reference_faces()
    
    def load_reference_faces(self) -> None:
        """
        Load and encode reference face images from the face directory
        
        Supports common image formats: jpg, jpeg, png, bmp
        """
        if not self.face_dir.exists():
            log_with_timestamp(f"Error: Face directory '{self.face_dir}' does not exist!", "[ERROR] ")
            sys.exit(1)
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in self.face_dir.iterdir() 
            if f.suffix.lower() in supported_formats
        ]
        
        if not image_files:
            log_with_timestamp(f"Error: No reference images found in '{self.face_dir}' directory!", "[ERROR] ")
            log_with_timestamp("Please add reference images (jpg, jpeg, png, bmp formats)", "[ERROR] ")
            sys.exit(1)
        
        log_with_timestamp(f"Loading {len(image_files)} reference images...", "[INFO] ")
        
        for image_file in image_files:
            try:
                # Load image and encode face
                image = face_recognition.load_image_file(str(image_file))
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Use the first face found in the image
                    self.known_faces.append(face_encodings[0])
                    # Use filename (without extension) as the name
                    name = image_file.stem
                    self.known_names.append(name)
                    log_with_timestamp(f"✓ Loaded: {name}", "[INFO] ")
                else:
                    log_with_timestamp(f"✗ No face found in: {image_file.name}", "[WARN] ")
                    
            except Exception as e:
                log_with_timestamp(f"✗ Error loading {image_file.name}: {e}", "[ERROR] ")
        
        if not self.known_faces:
            log_with_timestamp("Error: No valid faces found in reference images!", "[ERROR] ")
            sys.exit(1)
        
        log_with_timestamp(f"Successfully loaded {len(self.known_faces)} reference faces", "[INFO] ")
        print()  # Empty line for formatting
    
    def find_best_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Find the best available camera, preferring built-in cameras over external ones
        
        Returns:
            VideoCapture object for the best camera, or None if no camera found
        """
        log_with_timestamp("Searching for available cameras...", "[INFO] ")
        
        # Test different camera indices
        for camera_index in range(5):
            try:
                temp_capture = cv2.VideoCapture(camera_index)
                
                if temp_capture.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = temp_capture.read()
                    
                    if ret and test_frame is not None:
                        # Get camera info if available
                        width = temp_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                        height = temp_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        fps = temp_capture.get(cv2.CAP_PROP_FPS)
                        
                        log_with_timestamp(f"Camera {camera_index}: {int(width)}x{int(height)} @ {fps:.1f}fps", "[INFO] ")
                        
                        # Prefer cameras with reasonable resolution (not too high, not too low)
                        if 480 <= width <= 1920 and 360 <= height <= 1080:
                            log_with_timestamp(f"✓ Selected camera {camera_index}", "[INFO] ")
                            return temp_capture
                        else:
                            log_with_timestamp(f"  Camera {camera_index} resolution not ideal, continuing search...", "[WARN] ")
                            temp_capture.release()
                    else:
                        temp_capture.release()
                else:
                    if temp_capture is not None:
                        temp_capture.release()
                        
            except Exception as e:
                log_with_timestamp(f"Error testing camera {camera_index}: {e}", "[ERROR] ")
                continue
        
        log_with_timestamp("No suitable camera found!", "[ERROR] ")
        return None
    
    def execute_trigger_script(self, matched_name: str) -> None:
        """
        Execute the trigger script when a face is matched
        
        Args:
            matched_name: Name of the matched person
        """
        if not self.trigger_script:
            return
        
        current_time = time.time()
        # Check cooldown period to avoid spamming script execution
        if current_time - self.last_trigger_time < self.cooldown_seconds:
            return
        
        script_path = Path(self.trigger_script).expanduser()
        
        if not script_path.exists():
            log_with_timestamp(f"⚠️  Warning: Trigger script not found: {script_path}", "[WARN] ")
            return
        
        try:
            log_with_timestamp(f"🚀 Executing trigger script for: {matched_name}", "[INFO] ")
            log_with_timestamp(f"📁 Script path: {script_path}", "[INFO] ")
            
            # Execute the script with the matched name as argument
            result = subprocess.run(
                [str(script_path), matched_name],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                log_with_timestamp(f"✅ Script executed successfully", "[INFO] ")
                if result.stdout:
                    log_with_timestamp(f"📤 Output: {result.stdout.strip()}", "[INFO] ")
            else:
                log_with_timestamp(f"❌ Script execution failed (exit code: {result.returncode})", "[ERROR] ")
                if result.stderr:
                    log_with_timestamp(f"📥 Error: {result.stderr.strip()}", "[ERROR] ")
            
            self.last_trigger_time = current_time
            
        except subprocess.TimeoutExpired:
            log_with_timestamp(f"⏱️  Script execution timed out after 30 seconds", "[ERROR] ")
        except Exception as e:
            log_with_timestamp(f"❌ Error executing script: {e}", "[ERROR] ")
    
    def detect_and_compare_faces(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str], List[float]]:
        """
        Detect faces in the frame and compare with known faces
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple containing:
            - face_locations: List of face bounding boxes (top, right, bottom, left)
            - face_names: List of matched names for each face
            - face_distances: List of distances for each match
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB
        
        # Ensure the frame is in the correct data type (uint8)
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)
        
        # Find faces and encode them
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_distances = []
        
        for face_encoding in face_encodings:
            # Compare with known faces
            distances = face_recognition.face_distance(self.known_faces, face_encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            if best_distance < self.tolerance:
                name = self.known_names[best_match_index]
                face_names.append(name)
                face_distances.append(best_distance)
            else:
                face_names.append("Unknown")
                face_distances.append(best_distance)
        
        # Scale back up face locations
        face_locations = [(top*4, right*4, bottom*4, left*4) for (top, right, bottom, left) in face_locations]
        
        return face_locations, face_names, face_distances
    
    def draw_results(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]], 
                    face_names: List[str], face_distances: List[float]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame: Input frame
            face_locations: Face bounding boxes
            face_names: Face names
            face_distances: Face distances
            
        Returns:
            Frame with drawn results
        """
        for (top, right, bottom, left), name, distance in zip(face_locations, face_names, face_distances):
            # Choose color based on match result
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({distance:.2f})"
            else:
                color = (0, 255, 0)  # Green for known
                label = f"{name} ({distance:.2f})"
            
            # Draw rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            # Draw label text
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run_detection(self) -> None:
        """
        Main detection loop - captures video and performs real-time face comparison
        """
        log_with_timestamp("Starting face detection...", "[INFO] ")
        print("Controls:")
        print("  'q' - quit")
        print("  'r' - reload reference images")
        print("  '+' - increase tolerance (less strict)")
        print("  '-' - decrease tolerance (more strict)")
        print("=" * 50)
        
        # Find and initialize the best available camera
        video_capture = self.find_best_camera()
        
        if video_capture is None:
            log_with_timestamp("Error: Could not find a suitable camera!", "[ERROR] ")
            print("Troubleshooting steps:")
            print("1. Check camera permissions: System Preferences > Security & Privacy > Camera")
            print("2. Close other applications that might be using the camera")
            print("3. If using Continuity Camera, try disconnecting iPhone temporarily")
            print("4. Restart the application")
            return
        
        # Set camera properties for better compatibility
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        try:
            frame_count = 0
            while True:
                # Capture frame
                ret, frame = video_capture.read()
                if not ret or frame is None:
                    log_with_timestamp("Error: Could not read frame from camera!", "[ERROR] ")
                    break
                
                frame_count += 1
                # Process every 3rd frame for better performance
                if frame_count % 3 != 0:
                    # Still show the frame, but don't process it
                    cv2.imshow('Face Detection and Comparison', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        log_with_timestamp("Reloading reference images...", "[INFO] ")
                        self.known_faces.clear()
                        self.known_names.clear()
                        self.load_reference_faces()
                    continue
                
                try:
                    # Detect and compare faces
                    face_locations, face_names, face_distances = self.detect_and_compare_faces(frame)
                except Exception as e:
                    log_with_timestamp(f"Error in face detection: {e}", "[ERROR] ")
                    # Continue with empty results to keep the program running
                    face_locations, face_names, face_distances = [], [], []
                
                # Draw results
                result_frame = self.draw_results(frame, face_locations, face_names, face_distances)
                
                # Display results
                cv2.imshow('Face Detection and Comparison', result_frame)
                
                # Print match results to console and execute trigger script
                if face_names:
                    for name, distance in zip(face_names, face_distances):
                        match_status = "MATCH" if name != "Unknown" else "NO MATCH"
                        
                        # Add detailed analysis for debugging
                        confidence_level = "UNKNOWN"
                        if distance < 0.3:
                            confidence_level = "VERY HIGH"
                        elif distance < 0.4:
                            confidence_level = "HIGH"
                        elif distance < 0.5:
                            confidence_level = "MEDIUM"
                        elif distance < 0.6:
                            confidence_level = "LOW"
                        else:
                            confidence_level = "VERY LOW"
                        
                        log_with_timestamp(f"Face detected: {name} (distance: {distance:.3f}) - {match_status} - Confidence: {confidence_level}", "[DETECT] ")
                        
                        # Only execute trigger script for high confidence matches
                        if name != "Unknown" and distance < self.tolerance:
                            self.execute_trigger_script(name)
                        elif name != "Unknown" and distance >= self.tolerance:
                            log_with_timestamp(f"⚠️  Note: Match found but confidence too low (distance: {distance:.3f} >= threshold: {self.tolerance})", "[WARN] ")
                            log_with_timestamp(f"💡 Tip: If this should be a match, consider increasing tolerance to {distance + 0.05:.2f}", "[INFO] ")
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    log_with_timestamp("Reloading reference images...", "[INFO] ")
                    self.known_faces.clear()
                    self.known_names.clear()
                    self.load_reference_faces()
                elif key == ord('+') or key == ord('='):
                    # Increase tolerance (less strict)
                    old_tolerance = self.tolerance
                    self.tolerance = min(1.0, self.tolerance + 0.05)
                    log_with_timestamp(f"🔧 Tolerance adjusted: {old_tolerance:.2f} → {self.tolerance:.2f} (less strict)", "[CONFIG] ")
                elif key == ord('-') or key == ord('_'):
                    # Decrease tolerance (more strict)
                    old_tolerance = self.tolerance
                    self.tolerance = max(0.1, self.tolerance - 0.05)
                    log_with_timestamp(f"🔧 Tolerance adjusted: {old_tolerance:.2f} → {self.tolerance:.2f} (more strict)", "[CONFIG] ")
                    
        except KeyboardInterrupt:
            log_with_timestamp("Stopping face detection...", "[INFO] ")
        
        finally:
            # Clean up
            video_capture.release()
            cv2.destroyAllWindows()
            log_with_timestamp("Camera released and windows closed.", "[INFO] ")


def main():
    """
    Main function to run the face detection system
    """
    print("=" * 50)
    print("Face Detection and Comparison System")
    print("=" * 50)
    
    # Default trigger script path
    default_script = "~/scripts/raycast/hui.sh"
    
    # Create face detector instance with trigger script
    # Use more strict tolerance for better accuracy
    detector = FaceDetector(
        face_dir="face", 
        tolerance=0.44,  # More strict threshold to reduce false positives
        trigger_script=default_script,
        cooldown_seconds=5  # 5 seconds cooldown between executions
    )
    
    # Show configuration info
    print(f"🎯 Recognition tolerance: {detector.tolerance} (lower = more strict)")
    print(f"📊 Confidence levels: <0.3=VERY HIGH, <0.4=HIGH, <0.5=MEDIUM, <0.6=LOW")
    
    if detector.trigger_script:
        script_path = Path(detector.trigger_script).expanduser()
        if script_path.exists():
            print(f"🚀 Trigger script enabled: {script_path}")
            print(f"⏱️  Cooldown: {detector.cooldown_seconds} seconds")
        else:
            print(f"⚠️  Trigger script not found: {script_path}")
            print("🔧 Program will run without trigger functionality")
    
    print("\n💡 Tip: Watch the distance values to fine-tune recognition accuracy")
    print("   - If legitimate faces are rejected, increase tolerance")
    print("   - If wrong faces are accepted, decrease tolerance")
    print()
    
    # Run detection
    detector.run_detection()


if __name__ == "__main__":
    main()