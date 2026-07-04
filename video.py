# Enhanced Interview Evaluator for Windows VS Code
# - Improved eye detection with iris tracking
# - Better emotion consistency with confidence scoring
# - Dynamic thresholds based on video characteristics
# - Fixed Gemini integration with proper error handling
# - Adaptive sampling rate based on video length

import os
import cv2
import math
import time
import numpy as np
from collections import deque, Counter
from fer import FER
import mediapipe as mp
import google.generativeai as genai
import re
import json


def strip_emojis(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)  # keep only ASCII


def get_gemini_api_key():
    """Return your Gemini API key (simple hardcoded method)"""
    return "AIzaSyCddNdCN__Bd-8SYd7H6vbRYWe9RqCkfNk"

class InterviewAnalyzer:
    def __init__(self):
        # Initialize detectors with better configurations
        self.emotion_detector = FER(mtcnn=True)  # Use MTCNN for better face detection

        self.mp_holistic = mp.solutions.holistic
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        # Enhanced holistic model
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,  # Higher complexity for better accuracy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.7,  # Higher confidence threshold
            min_tracking_confidence=0.7
        )

        # Face mesh for detailed eye tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Dynamic thresholds (will be adjusted based on video)
        self.fidget_threshold = 0.06
        self.high_fidget_threshold = 0.15
        self.eye_contact_tolerance = 0.25

        # Smoothing parameters
        self.smooth_window = 7  # Larger window for better consistency
        self.emotion_confidence_threshold = 0.3

        # Buffers for smoothing
        self.emotion_hist = deque(maxlen=self.smooth_window)
        self.emotion_conf_hist = deque(maxlen=self.smooth_window)
        self.eye_hist = deque(maxlen=self.smooth_window)
        self.posture_hist = deque(maxlen=self.smooth_window)
        self.handvis_hist = deque(maxlen=self.smooth_window)
        self.handmov_hist = deque(maxlen=self.smooth_window)

        # Stats
        self.timeline = []
        self.stats = {
            'emotion_counter': Counter(),
            'eye_good': 0, 'eye_total': 0,
            'posture_upright': 0, 'posture_total': 0,
            'hands_visible_count': 0, 'hands_total': 0,
            'fidget_events': 0, 'nervous_ticks': 0
        }

        # Hand tracking
        self.last_left_wrist = None
        self.last_right_wrist = None
        self.hand_movement_history = deque(maxlen=10)

        # Video properties (will be set dynamically)
        self.fps = 25
        self.width = 640
        self.height = 480
        self.frame_interval = 1

    def analyze_video_properties(self, cap):
        """Analyze video to set dynamic parameters"""
        # Get video properties
        fps_read = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fps = int(round(fps_read)) if fps_read and fps_read > 0 else 25
        video_duration = total_frames / self.fps if self.fps > 0 else 0

        # Dynamic sampling rate based on video length
        if video_duration > 300:  # 5+ minutes
            self.frame_interval = int(self.fps * 2)  # Sample every 2 seconds
        elif video_duration > 120:  # 2+ minutes
            self.frame_interval = int(self.fps * 1.5)  # Sample every 1.5 seconds
        else:
            self.frame_interval = max(1, int(self.fps))  # Sample every second

        # Adjust thresholds based on video resolution
        resolution_factor = min(self.width, self.height) / 480.0
        self.fidget_threshold = 0.06 * resolution_factor
        self.high_fidget_threshold = 0.15 * resolution_factor

        print(f"Video: {self.width}x{self.height}, {self.fps}fps, {video_duration:.1f}s duration")
        print(f"Sampling every {self.frame_interval} frames ({self.frame_interval/self.fps:.1f}s)")

        return video_duration, total_frames

    def enhanced_eye_contact_detection(self, face_landmarks):
        """Improved eye contact detection using iris landmarks"""
        try:
            landmarks = face_landmarks.landmark
            # Eye corner landmarks (more reliable)
            left_eye_left = landmarks[33]   # Left eye left corner
            left_eye_right = landmarks[133]  # Left eye right corner
            right_eye_left = landmarks[362]  # Right eye left corner
            right_eye_right = landmarks[263] # Right eye right corner
            # Eye center points
            left_eye_center = landmarks[468] if len(landmarks) > 468 else None
            right_eye_center = landmarks[473] if len(landmarks) > 473 else None
            # Fallback to pupil estimates if iris landmarks not available
            if not left_eye_center:
                left_eye_center_x = (left_eye_left.x + left_eye_right.x) / 2
                left_eye_center_y = (left_eye_left.y + left_eye_right.y) / 2
                left_eye_center = type('obj', (object,), {'x': left_eye_center_x, 'y': left_eye_center_y})
            if not right_eye_center:
                right_eye_center_x = (right_eye_left.x + right_eye_right.x) / 2
                right_eye_center_y = (right_eye_left.y + right_eye_right.y) / 2
                right_eye_center = type('obj', (object,), {'x': right_eye_center_x, 'y': right_eye_center_y})
            # Calculate gaze ratios
            left_width = abs(left_eye_right.x - left_eye_left.x)
            right_width = abs(right_eye_right.x - right_eye_left.x)
            left_ratio = (left_eye_center.x - left_eye_left.x) / (left_width + 1e-6)
            right_ratio = (right_eye_center.x - right_eye_left.x) / (right_width + 1e-6)
            avg_ratio = (left_ratio + right_ratio) / 2.0
            # More nuanced eye contact assessment
            if 0.4 <= avg_ratio <= 0.6:
                return "Excellent", avg_ratio, 1.0
            elif 0.3 <= avg_ratio <= 0.7:
                return "Good", avg_ratio, 0.8
            elif 0.25 <= avg_ratio <= 0.75:
                return "Fair", avg_ratio, 0.6
            else:
                return "Looking Away", avg_ratio, 0.2
        except Exception as e:
            print(f"Eye detection error: {e}")
            return "Not Detected", None, 0.0

    def enhanced_emotion_detection(self, frame_rgb, face_landmarks=None):
        """Improved emotion detection with confidence scoring and face cropping"""
        try:
            # Use face landmarks to create better crop if available
            if face_landmarks:
                landmarks = face_landmarks.landmark
                xs = [p.x for p in landmarks]
                ys = [p.y for p in landmarks]

                # Add padding around face
                padding_x = 0.1 * self.width
                padding_y = 0.1 * self.height

                x_min = int(max(0, min(xs) * self.width - padding_x))
                x_max = int(min(self.width, max(xs) * self.width + padding_x))
                y_min = int(max(0, min(ys) * self.height - padding_y))
                y_max = int(min(self.height, max(ys) * self.height + padding_y))

                face_crop = frame_rgb[y_min:y_max, x_min:x_max]

                if face_crop.size > 0:
                    # Convert to BGR for FER
                    face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                    fer_results = self.emotion_detector.detect_emotions(face_bgr)
                else:
                    fer_results = None
            else:
                # Fallback to full frame
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                fer_results = self.emotion_detector.detect_emotions(frame_bgr)

            if fer_results and len(fer_results) > 0:
                emotions = fer_results[0].get("emotions", {})
                if emotions:
                    # Get emotion with highest confidence
                    best_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[best_emotion]

                    # Only accept emotions with sufficient confidence
                    if confidence >= self.emotion_confidence_threshold:
                        return best_emotion, confidence
                    else:
                        return "neutral", confidence  # Default to neutral for low confidence

            return "unknown", 0.0

        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "unknown", 0.0

    def improved_posture_analysis(self, pose_landmarks):
        """Enhanced posture analysis with shoulder alignment"""
        try:
            landmarks = pose_landmarks.landmark

            # Get key points
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
            left_hip = np.array([landmarks[23].x, landmarks[23].y])
            right_hip = np.array([landmarks[24].x, landmarks[24].y])
            nose = np.array([landmarks[0].x, landmarks[0].y])

            # Calculate midpoints
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0
            hip_mid = (left_hip + right_hip) / 2.0

            # Torso vector (from hips to shoulders)
            torso_vector = shoulder_mid - hip_mid
            vertical = np.array([0.0, -1.0])  # Upward vertical

            # Calculate angle from vertical
            dot_product = np.dot(torso_vector, vertical)
            norms = np.linalg.norm(torso_vector) * np.linalg.norm(vertical)
            cos_angle = np.clip(dot_product / (norms + 1e-6), -1.0, 1.0)
            angle = math.acos(cos_angle)

            # Shoulder alignment check
            shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
            shoulder_tilt = shoulder_diff > 0.05  # 5% of frame height

            # Head position relative to torso
            head_offset = abs(nose[0] - shoulder_mid[0])
            head_forward = head_offset > 0.08  # Head too far forward

            # Classify posture
            angle_deg = math.degrees(angle)
            if angle_deg < 8 and not shoulder_tilt and not head_forward:
                return "Excellent", angle, 1.0
            elif angle_deg < 15 and not (shoulder_tilt and head_forward):
                return "Good", angle, 0.8
            elif angle_deg < 25:
                return "Fair", angle, 0.6
            else:
                return "Poor", angle, 0.3

        except Exception as e:
            print(f"Posture analysis error: {e}")
            return "Unknown", None, 0.0

    def analyze_hand_behavior(self, results, norm_factor):
        """Enhanced hand analysis with gesture vs fidget classification"""
        left_vis = bool(results and results.left_hand_landmarks)
        right_vis = bool(results and results.right_hand_landmarks)
        hands_vis = left_vis or right_vis

        # Get wrist positions
        left_wrist = self.get_hand_wrist_coords(results.left_hand_landmarks) if left_vis else None
        right_wrist = self.get_hand_wrist_coords(results.right_hand_landmarks) if right_vis else None

        # Calculate movement
        left_move = right_move = 0.0
        if self.last_left_wrist and left_wrist:
            left_move = self.euclidean_distance(left_wrist, self.last_left_wrist) / norm_factor
        if self.last_right_wrist and right_wrist:
            right_move = self.euclidean_distance(right_wrist, self.last_right_wrist) / norm_factor

        # Update last positions (maintain tracking even if hand briefly disappears)
        if left_wrist:
            self.last_left_wrist = left_wrist
        if right_wrist:
            self.last_right_wrist = right_wrist

        # Track movement history for pattern analysis
        current_movement = max(left_move, right_move)
        self.hand_movement_history.append(current_movement)

        # Classify hand activity with more nuance
        if not hands_vis:
            return "Not Visible", current_movement, 0.0

        # Analyze movement patterns
        recent_movements = list(self.hand_movement_history)[-5:]  # Last 5 samples
        avg_recent_movement = np.mean(recent_movements) if recent_movements else 0
        movement_variance = np.var(recent_movements) if len(recent_movements) > 1 else 0

        # Dynamic thresholds based on overall movement patterns
        if avg_recent_movement > self.high_fidget_threshold or movement_variance > 0.01:
            activity = "Excessive Fidgeting"
            confidence = 0.9
            self.stats['fidget_events'] += 1
        elif avg_recent_movement > self.fidget_threshold:
            if movement_variance < 0.005:  # Consistent movement = likely gesture
                activity = "Purposeful Gesture"
                confidence = 0.7
            else:
                activity = "Mild Fidgeting"
                confidence = 0.6
        else:
            activity = "Calm"
            confidence = 0.8

        return activity, current_movement, confidence

    def get_hand_wrist_coords(self, hand_landmarks):
        """Get normalized wrist coordinates"""
        try:
            if hand_landmarks:
                wrist = hand_landmarks.landmark[0]  # Wrist is landmark 0
                return (wrist.x, wrist.y)
        except Exception:
            pass
        return None

    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def smooth_emotion_with_confidence(self):
        """Apply confidence-weighted smoothing to emotions"""
        if not self.emotion_hist or not self.emotion_conf_hist:
            return "unknown"

        # Weight recent emotions by confidence
        weighted_emotions = {}
        for emotion, confidence in zip(self.emotion_hist, self.emotion_conf_hist):
            if emotion != "unknown":
                weighted_emotions[emotion] = weighted_emotions.get(emotion, 0) + confidence

        if weighted_emotions:
            return max(weighted_emotions, key=weighted_emotions.get)
        return "unknown"

    def compute_normalization_factor(self, pose_landmarks):
        """Compute shoulder width for movement normalization"""
        try:
            landmarks = pose_landmarks.landmark
            left_shoulder = (landmarks[11].x, landmarks[11].y)
            right_shoulder = (landmarks[12].x, landmarks[12].y)
            shoulder_width = self.euclidean_distance(left_shoulder, right_shoulder)
            return max(shoulder_width, 0.1)  # Minimum threshold
        except Exception:
            return 1.0

    def timestamp_from_frame(self, frame_idx):
        """Convert frame index to MM:SS timestamp"""
        seconds = int(frame_idx / max(1, self.fps))
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def detect_nervous_behaviors(self, entry):
        """Enhanced nervous behavior detection"""
        tick_score = 0
        reasons = []

        # Eye contact scoring
        if entry["eye_contact"] in ["Looking Away"]:
            tick_score += 2
            reasons.append("poor_eye_contact")
        elif entry["eye_contact"] == "Fair":
            tick_score += 1
            reasons.append("inconsistent_eye_contact")

        # Posture scoring
        if entry["posture"] == "Poor":
            tick_score += 2
            reasons.append("poor_posture")
        elif entry["posture"] == "Fair":
            tick_score += 1
            reasons.append("slouching")

        # Hand behavior scoring
        if entry["hand_activity"] == "Excessive Fidgeting":
            tick_score += 3
            reasons.append("excessive_fidgeting")
        elif entry["hand_activity"] == "Mild Fidgeting":
            tick_score += 1
            reasons.append("fidgeting")

        # Emotion scoring
        if entry["emotion"] in ["angry", "fear", "sad"]:
            tick_score += 2
            reasons.append("negative_emotion")
        elif entry["emotion"] == "disgust":
            tick_score += 1
            reasons.append("discomfort")

        return tick_score, reasons

    def process_video(self, video_path):
        """Main video processing function"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Analyze video properties
        duration, total_frames = self.analyze_video_properties(cap)

        # Setup output video in same directory as input
        output_path = os.path.join(os.path.dirname(video_path), "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_idx = 0
        processed_samples = 0
        start_time = time.time()

        print("Processing video... This may take a few minutes.")

        # For overlay (shows last known values)
        last_emotion = "..."
        last_eye = "..."
        last_posture = "..."
        last_hands = "..."

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process every Nth frame for analysis
            if frame_idx % self.frame_interval == 0:
                processed_samples += 1

                # Run holistic analysis
                holistic_results = self.holistic.process(frame_rgb)
                face_mesh_results = self.face_mesh.process(frame_rgb)

                # Emotion analysis
                emotion, emotion_conf = self.enhanced_emotion_detection(
                    frame_rgb,
                    holistic_results.face_landmarks if holistic_results else None
                )

                self.emotion_hist.append(emotion)
                self.emotion_conf_hist.append(emotion_conf)
                smoothed_emotion = self.smooth_emotion_with_confidence()

                # Eye contact analysis (use face mesh for better accuracy)
                eye_contact, gaze_ratio, eye_conf = self.enhanced_eye_contact_detection(
                    face_mesh_results.multi_face_landmarks[0] if (face_mesh_results and face_mesh_results.multi_face_landmarks) else None
                )

                if eye_contact != "Not Detected":
                    self.stats['eye_total'] += 1
                    if eye_contact in ["Excellent", "Good"]:
                        self.stats['eye_good'] += 1

                # Posture analysis
                posture, posture_angle, posture_conf = self.improved_posture_analysis(
                    holistic_results.pose_landmarks if holistic_results else None
                )

                if posture != "Unknown":
                    self.stats['posture_total'] += 1
                    if posture in ["Excellent", "Good"]:
                        self.stats['posture_upright'] += 1

                # Hand analysis
                norm_factor = self.compute_normalization_factor(
                    holistic_results.pose_landmarks if holistic_results else None
                )
                hand_activity, hand_movement, hand_conf = self.analyze_hand_behavior(
                    holistic_results, norm_factor
                )

                hands_visible = hand_activity != "Not Visible"
                self.stats['hands_total'] += 1
                if hands_visible:
                    self.stats['hands_visible_count'] += 1

                # Update smoothing buffers
                self.eye_hist.append(eye_contact)
                self.posture_hist.append(posture)
                self.handvis_hist.append("Visible" if hands_visible else "Hidden")
                self.handmov_hist.append(hand_activity)

                # Get smoothed values
                smoothed_eye = Counter(self.eye_hist).most_common(1)[0][0] if self.eye_hist else "Not Detected"
                smoothed_posture = Counter(self.posture_hist).most_common(1)[0][0] if self.posture_hist else "Unknown"
                smoothed_handvis = Counter(self.handvis_hist).most_common(1)[0][0] if self.handvis_hist else "Hidden"
                smoothed_handmov = Counter(self.handmov_hist).most_common(1)[0][0] if self.handmov_hist else "Not Visible"

                # Create timeline entry
                timestamp = self.timestamp_from_frame(frame_idx)
                entry = {
                    "time": timestamp,
                    "frame": frame_idx,
                    "emotion": smoothed_emotion,
                    "emotion_confidence": emotion_conf,
                    "eye_contact": smoothed_eye,
                    "eye_confidence": eye_conf,
                    "posture": smoothed_posture,
                    "posture_confidence": posture_conf,
                    "hands_visible": (smoothed_handvis == "Visible"),
                    "hand_activity": smoothed_handmov,
                    "hand_confidence": hand_conf,
                    "gaze_ratio": gaze_ratio,
                    "posture_angle": posture_angle
                }

                # Detect nervous behaviors
                tick_score, tick_reasons = self.detect_nervous_behaviors(entry)
                entry["nervous_score"] = tick_score
                entry["tick_reasons"] = tick_reasons

                self.stats['nervous_ticks'] += tick_score
                self.timeline.append(entry)

                # Update overlay values
                last_emotion = smoothed_emotion
                last_eye = smoothed_eye
                last_posture = smoothed_posture
                last_hands = smoothed_handmov

                # Progress indicator
                if processed_samples % 10 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({processed_samples} samples)")

            # Add annotations to every frame
            self.annotate_frame(frame, last_emotion, last_eye, last_posture, last_hands)
            out.write(frame)
            frame_idx += 1

        # Cleanup
        cap.release()
        out.release()
        self.holistic.close()
        self.face_mesh.close()

        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.1f}s. Analyzed {len(self.timeline)} samples.")
        print(f"Annotated video saved to: {output_path}")

        return duration

    def annotate_frame(self, frame, emotion, eye_contact, posture, hand_activity):
        """Add informative annotations to frame"""
        # Define colors based on quality
        color_map = {
            "Excellent": (0, 255, 0),    # Green
            "Good": (0, 200, 100),       # Light green
            "Fair": (0, 165, 255),       # Orange
            "Poor": (0, 0, 255),         # Red
            "Looking Away": (0, 0, 255),
            "Slouched": (0, 0, 255),
            "Excessive Fidgeting": (0, 0, 255)
        }

        # Background for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Text annotations with color coding
        emotion_color = color_map.get(emotion, (255, 255, 255))
        eye_color = color_map.get(eye_contact, (255, 255, 255))
        posture_color = color_map.get(posture, (255, 255, 255))
        hand_color = color_map.get(hand_activity, (255, 255, 255))

        cv2.putText(frame, f"Emotion: {emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        cv2.putText(frame, f"Eye Contact: {eye_contact}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"Posture: {posture}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, posture_color, 2)
        cv2.putText(frame, f"Hands: {hand_activity}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

    def generate_highlights(self):
        """Generate highlights of concerning periods"""
        highlights = []
        current_highlight = None
        consecutive_threshold = 3  # Minimum consecutive samples for highlight

        for i, entry in enumerate(self.timeline):
            nervous_score = entry.get("nervous_score", 0)

            if nervous_score >= 2:  # Significant concern threshold
                if current_highlight is None:
                    current_highlight = {
                        "start_time": entry["time"],
                        "start_frame": entry["frame"],
                        "end_time": entry["time"],
                        "end_frame": entry["frame"],
                        "reasons": set(entry["tick_reasons"]),
                        "duration": 1,
                        "avg_score": nervous_score,
                        "max_score": nervous_score
                    }
                else:
                    # Extend current highlight
                    current_highlight["end_time"] = entry["time"]
                    current_highlight["end_frame"] = entry["frame"]
                    current_highlight["reasons"].update(entry["tick_reasons"])
                    current_highlight["duration"] += 1
                    current_highlight["avg_score"] = (current_highlight["avg_score"] + nervous_score) / 2
                    current_highlight["max_score"] = max(current_highlight["max_score"], nervous_score)
            else:
                # End current highlight if it meets threshold
                if current_highlight and current_highlight["duration"] >= consecutive_threshold:
                    current_highlight["reasons"] = list(current_highlight["reasons"])
                    highlights.append(current_highlight)
                current_highlight = None

        # Don't forget the last highlight
        if current_highlight and current_highlight["duration"] >= consecutive_threshold:
            current_highlight["reasons"] = list(current_highlight["reasons"])
            highlights.append(current_highlight)

        return highlights

    def generate_summary(self, video_duration):
        """Generate comprehensive summary statistics"""
        total_samples = len(self.timeline)

        # Calculate percentages
        eye_contact_pct = round(100.0 * self.stats['eye_good'] / max(1, self.stats['eye_total']), 1)
        posture_pct = round(100.0 * self.stats['posture_upright'] / max(1, self.stats['posture_total']), 1)
        hands_visible_pct = round(100.0 * self.stats['hands_visible_count'] / max(1, self.stats['hands_total']), 1)

        # Emotion analysis
        emotion_counts = Counter([e["emotion"] for e in self.timeline if e["emotion"] != "unknown"])
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "unknown"

        # Confidence scores
        avg_emotion_conf = np.mean([e.get("emotion_confidence", 0) for e in self.timeline])
        avg_eye_conf = np.mean([e.get("eye_confidence", 0) for e in self.timeline if e.get("eye_confidence")])
        avg_posture_conf = np.mean([e.get("posture_confidence", 0) for e in self.timeline if e.get("posture_confidence")])

        # Overall performance score (0-100)
        performance_score = (
            eye_contact_pct * 0.3 +
            posture_pct * 0.25 +
            hands_visible_pct * 0.15 +
            max(0, 100 - self.stats['nervous_ticks'] * 2) * 0.3
        )

        return {
            "video_duration_seconds": video_duration,
            "total_samples_analyzed": total_samples,
            "sampling_rate_seconds": self.frame_interval / self.fps,
            "eye_contact_good_percent": eye_contact_pct,
            "posture_upright_percent": posture_pct,
            "hands_visible_percent": hands_visible_pct,
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": dict(emotion_counts),
            "fidget_events": self.stats['fidget_events'],
            "nervous_ticks_total": self.stats['nervous_ticks'],
            "overall_performance_score": round(performance_score, 1),
            "confidence_scores": {
                "emotion_avg": round(avg_emotion_conf, 3),
                "eye_contact_avg": round(avg_eye_conf, 3),
                "posture_avg": round(avg_posture_conf, 3)
            }
        }

    def generate_report(self, summary, highlights, video_duration):
        """Generate human-readable report"""
        report_lines = [
            "Enhanced Interview Analysis Report",
            "==================================",
            f"Video Duration: {video_duration:.1f} seconds",
            f"Samples Analyzed: {summary['total_samples_analyzed']} (every {summary['sampling_rate_seconds']:.1f}s)",
            f"Overall Performance Score: {summary['overall_performance_score']}/100",
            "",
            "DETAILED METRICS:",
            "----------------",
            f"Eye Contact (Good/Excellent): {summary['eye_contact_good_percent']}%",
            f"Posture (Good/Excellent): {summary['posture_upright_percent']}%",
            f"Hand Visibility: {summary['hands_visible_percent']}%",
            f"Dominant Emotion: {summary['dominant_emotion']}",
            "",
            "BEHAVIORAL ANALYSIS:",
            "-------------------",
            f"Fidgeting Events: {summary['fidget_events']}",
            f"Nervous Behavior Score: {summary['nervous_ticks_total']}",
            "",
            "EMOTION BREAKDOWN:",
            "-----------------"
        ]
        
        for emotion, count in summary['emotion_distribution'].items():
            percentage = round(100 * count / summary['total_samples_analyzed'], 1)
            report_lines.append(f"- {emotion.title()}: {count} samples ({percentage}%)")
        
        report_lines.extend([
            "",
            "CONFIDENCE SCORES:",
            "-----------------",
            f"Emotion Detection: {summary['confidence_scores']['emotion_avg']:.1%}",
            f"Eye Contact Detection: {summary['confidence_scores']['eye_contact_avg']:.1%}",
            f"Posture Detection: {summary['confidence_scores']['posture_avg']:.1%}",
            "",
            "AREAS OF CONCERN (Highlights):",
            "-----------------------------"
        ])
        
        if highlights:
            for i, h in enumerate(highlights, 1):
                duration = h['duration'] * summary['sampling_rate_seconds']
                report_lines.append(
                    f"{i}. {h['start_time']} - {h['end_time']} ({duration:.1f}s): "
                    f"{', '.join(h['reasons'])} (severity: {h['max_score']}/10)"
                )
        else:
            report_lines.append("✅ No significant periods of concern detected!")
        
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "---------------"
        ])
        
        # Dynamic recommendations based on performance
        if summary['eye_contact_good_percent'] < 60:
            report_lines.append("• Practice maintaining eye contact with the camera/interviewer")
        if summary['posture_upright_percent'] < 70:
            report_lines.append("• Work on maintaining upright, confident posture")
        if summary['fidget_events'] > 5:
            report_lines.append("• Practice keeping hands calm and purposeful")
        if summary['nervous_ticks_total'] > 20:
            report_lines.append("• Consider relaxation techniques before interviews")
        
        # Positive reinforcement
        if summary['overall_performance_score'] > 80:
            report_lines.append("✅ Strong overall performance - keep up the good work!")
        elif summary['overall_performance_score'] > 60:
            report_lines.append("• Good foundation - focus on consistency improvements")
        else:
            report_lines.append("• Consider additional practice sessions and mock interviews")
        
        return "\n".join(report_lines)

    def setup_gemini(self, api_key=None):
     """Setup Gemini API with proper error handling"""
     if not api_key:
        api_key = get_gemini_api_key()  # 👈 always fetch from function

    

     try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        test_response = model.generate_content("Test connection")
        return True, model
     except Exception as e:
        return False, f"Gemini setup failed: {str(e)}"


    def generate_gemini_feedback(self, summary, highlights, timeline_sample):
        """Generate AI-powered coaching feedback"""
        # Get Gemini API key from environment or user input
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("\n🔑 Gemini API Key Required for AI Feedback")
            print("Please enter your Gemini API key (or press Enter to skip):")
            try:
                from getpass import getpass
                api_key = getpass("API Key: ").strip()
            except:
                api_key = input("API Key: ").strip()
        
        if not api_key:
            return self.generate_fallback_feedback(summary, highlights)
        
        success, model_or_error = self.setup_gemini(api_key)
        if not success:
            print(f"❌ {model_or_error}")
            return self.generate_fallback_feedback(summary, highlights)
        
        model = model_or_error
        
        # Create comprehensive prompt for Gemini
        prompt = f"""You are an expert interview coach analyzing a candidate's performance. Based on the following data, provide personalized, actionable feedback.

PERFORMANCE SUMMARY:
- Overall Score: {summary['overall_performance_score']}/100
- Eye Contact (Good): {summary['eye_contact_good_percent']}%
- Posture (Upright): {summary['posture_upright_percent']}%
- Hand Visibility: {summary['hands_visible_percent']}%
- Dominant Emotion: {summary['dominant_emotion']}
- Fidgeting Events: {summary['fidget_events']}
- Nervous Behavior Score: {summary['nervous_ticks_total']}
- Video Duration: {summary['video_duration_seconds']:.1f} seconds

EMOTION BREAKDOWN:
{json.dumps(summary['emotion_distribution'], indent=2)}

PROBLEMATIC MOMENTS:
{json.dumps(highlights, indent=2)}

SAMPLE TIMELINE DATA (first 20 entries):
{json.dumps(timeline_sample[:20], indent=2)}

Please provide:
1. An overall assessment (2-3 sentences)
2. Top 3 strengths identified
3. Top 3 areas for improvement with specific examples
4. 5 actionable practice recommendations
5. Encouragement and next steps

Keep the tone supportive but honest. Use specific timestamps when referencing concerning moments.
"""

        try:
            print("\n🤖 Generating AI-powered coaching feedback...")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            return self.generate_fallback_feedback(summary, highlights)

    def generate_fallback_feedback(self, summary, highlights):
        """Generate local feedback when Gemini is unavailable"""
        score = summary['overall_performance_score']
        
        feedback_lines = [
            "Interview Performance Feedback",
            "==============================",
            "",
            f"Overall Performance: {score}/100",
        ]
        
        # Assessment based on score
        if score >= 85:
            feedback_lines.append("🌟 Excellent interview presence! You demonstrate strong professional communication skills.")
        elif score >= 70:
            feedback_lines.append("✅ Good interview performance with room for targeted improvements.")
        elif score >= 55:
            feedback_lines.append("📈 Solid foundation with several areas that would benefit from focused practice.")
        else:
            feedback_lines.append("🎯 Significant opportunity for improvement - consider intensive practice sessions.")
        
        feedback_lines.extend([
            "",
            "STRENGTHS IDENTIFIED:",
            "--------------------"
        ])
        
        # Dynamic strengths based on performance
        strengths = []
        if summary['eye_contact_good_percent'] >= 70:
            strengths.append(f"• Strong eye contact ({summary['eye_contact_good_percent']}% good/excellent)")
        if summary['posture_upright_percent'] >= 75:
            strengths.append(f"• Excellent posture maintenance ({summary['posture_upright_percent']}%)")
        if summary['hands_visible_percent'] >= 80:
            strengths.append("• Good hand visibility and positioning")
        if summary['fidget_events'] <= 3:
            strengths.append("• Calm, controlled hand movements")
        if summary['dominant_emotion'] in ['happy', 'neutral']:
            strengths.append(f"• Positive emotional presentation ({summary['dominant_emotion']})")
        
        if not strengths:
            strengths.append("• Face and body clearly visible throughout the interview")
            strengths.append("• Willingness to practice and improve")
        
        feedback_lines.extend(strengths)
        feedback_lines.extend([
            "",
            "IMPROVEMENT OPPORTUNITIES:",
            "-------------------------"
        ])
        
        # Dynamic improvement areas
        improvements = []
        if summary['eye_contact_good_percent'] < 60:
            improvements.append(f"• Eye contact needs work ({summary['eye_contact_good_percent']}% good) - practice looking directly at camera")
        if summary['posture_upright_percent'] < 70:
            improvements.append(f"• Posture could be more consistent ({summary['posture_upright_percent']}% upright)")
        if summary['fidget_events'] > 5:
            improvements.append(f"• Reduce fidgeting ({summary['fidget_events']} events detected)")
        if summary['nervous_ticks_total'] > 15:
            improvements.append("• Work on reducing nervous behaviors and building confidence")
        
        if not improvements:
            improvements.append("• Continue practicing to maintain excellent performance")
        
        feedback_lines.extend(improvements)
        
        # Highlight specific moments
        if highlights:
            feedback_lines.extend([
                "",
                "SPECIFIC MOMENTS TO REVIEW:",
                "---------------------------"
            ])
            for h in highlights[:3]:  # Top 3 highlights
                duration = h['duration'] * (self.frame_interval / self.fps)
                feedback_lines.append(f"• {h['start_time']}-{h['end_time']} ({duration:.1f}s): {', '.join(h['reasons'])}")
        
        feedback_lines.extend([
            "",
            "PRACTICE RECOMMENDATIONS:",
            "------------------------",
            "• Record 5-minute mock interviews daily focusing on your weak areas",
            "• Practice the STAR method (Situation, Task, Action, Result) for responses",
            "• Use a mirror or camera to practice maintaining eye contact",
            "• Work on breathing exercises to reduce nervous energy",
            "• Practice common interview questions while monitoring your body language",
            "",
            "Keep practicing - improvement comes with consistent effort! 💪"
        ])
        
        return "\n".join(feedback_lines)

    def save_results(self, summary, highlights, report_text, feedback_text):
        """Save only annotated video and feedback/report files"""
        with open("analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report_text)
        with open("coach_feedback.txt", "w", encoding="utf-8") as f:
            f.write(feedback_text)
        print("✅ Results saved:")
        print("  - annotated_output.mp4 (video with overlays)")
        print("  - analysis_report.txt (technical report)")
        print("  - coach_feedback.txt (coaching feedback)")

# =========================
# Main Execution Functions
# =========================

def run_local_analysis(video_path, gemini_api_key=None):
    """Function to run analysis locally (Windows/VS Code)"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Set Gemini API key if provided
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    
    analyzer = InterviewAnalyzer()
    
    print(f"🎬 Analyzing video: {video_path}")
    video_duration = analyzer.process_video(video_path)
    
    highlights = analyzer.generate_highlights()
    summary = analyzer.generate_summary(video_duration)
    
    report_text = analyzer.generate_report(summary, highlights, video_duration)
    feedback_text = analyzer.generate_gemini_feedback(summary, highlights, analyzer.timeline)
    
    analyzer.save_results(summary, highlights, report_text, feedback_text)
    
    print("\n" + "="*60)
    print(report_text)
    print("\n" + "="*60)
    print(feedback_text)
    
    return analyzer, summary, highlights

def main():
    """Main function for interactive execution"""
    print("Enhanced Interview Analyzer - Windows/VS Code Version")
    print("====================================================")

    print("Select input method:")
    print("1. Upload video file")
    print("2. Use webcam")
    choice = input("Enter 1 or 2: ").strip()

    video_path = None
    temp_webcam_path = "test.mp4"

    if choice == "1":
        video_path = input("Enter the path to your interview video file: ").strip()
    elif choice == "2":
        print("\nRecording from webcam. Press 'q' to stop recording.")
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_webcam_path, fourcc, 25.0, (640, 480))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Webcam - Press q to stop', frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        video_path = temp_webcam_path
    else:
        print("Invalid choice. Exiting.")
        return

    # Use hardcoded Gemini API key (backend)
    api_key = get_gemini_api_key()

    try:
        analyzer, summary, highlights = run_local_analysis(video_path, api_key)
        print("\n✅ Analysis complete! Check the generated files for detailed results.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your video file path and try again.")

# Example usage:
if __name__ == "__main__":
    # For direct execution
    main()
    
    # Alternative: Direct function call
    # analyzer, summary, highlights = run_local_analysis("path/to/your/video.mp4", "your_gemini_api_key")