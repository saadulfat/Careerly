import google.generativeai as genai
import json
import re
import os
import cv2
import math
import time
import numpy as np
from collections import deque, Counter
from fer import FER
import mediapipe as mp
import csv
import uuid
from datetime import datetime
from typing import Dict, List
try:
    import whisper
except Exception:
    whisper = None

# Configure your Gemini API key
genai.configure(api_key="GEMINI_API_KEY")

EVAL_METRICS = ["Clarity", "Knowledge", "Conciseness", "Confidence", "Structure"]

class MasterAgent:
    """
    Master agent class that handles all Gemini configuration and provides centralized access to all agents.
    This eliminates the need to repeatedly configure Gemini and create model instances.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the master agent with optional API key.
        If no API key is provided, uses the default one configured above.
        """
        if api_key:
            genai.configure(api_key=api_key)
        
        # Initialize the Gemini model once
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Initialize all agents
        self.role_validator = RoleValidatorAgent(self.model)
        self.question_generator = QuestionGeneratorAgent(self.model)
        self.answer_evaluator = AnswerEvaluatorAgent(self.model)
        self.video_analyzer = VideoAnalyzerAgent(self.model)
    
    def validate_role(self, role):
        """Validate a job role using the RoleValidatorAgent."""
        return self.role_validator.validate_role(role)
    
    def generate_questions(self, role, interview_type="Job", level="Beginner"):
        """Generate interview questions using the QuestionGeneratorAgent."""
        return self.question_generator.generate_questions(role, interview_type, level)
    
    def evaluate_answer(self, question, answer, role, audio_metrics=None):
        """Evaluate an answer using the AnswerEvaluatorAgent."""
        return self.answer_evaluator.evaluate_answer(question, answer, role, audio_metrics)
    
    def analyze_video(self, video_path):
        """Analyze a video using the VideoAnalyzerAgent."""
        return self.video_analyzer.analyze_video(video_path)
    
    def get_model(self):
        """Get the configured Gemini model instance."""
        return self.model

# Integrated Whisper audio analyzer (moved from main.py)
WHISPER_MODEL = None
try:
    if whisper is not None:
        WHISPER_MODEL = whisper.load_model("base")
        print("Whisper model loaded successfully (base).")
except Exception as e:
    print(f"Error loading Whisper model: {e}. Audio transcription will be unavailable.")

def analyze_audio_for_hesitation(audio_path, whisper_model):
    if not whisper_model:
        return None
    result = whisper_model.transcribe(audio_path, word_timestamps=True, language="en")
    transcribed_text = result["text"]
    if not result.get("segments") or not result["segments"][0].get("words"):
        return {
            "full_transcription": transcribed_text,
            "filler_word_count": 0,
            "total_filler_duration": 0.0,
            "average_filler_duration": 0.0,
            "total_pause_duration": 0.0,
            "num_pauses": 0,
            "average_pause_duration": 0.0,
            "speech_rate_wpm": 0.0,
            "total_speech_duration": 0.0,
            "word_details": []
        }
    word_segments = []
    for segment in result["segments"]:
        if "words" in segment:
            word_segments.extend(segment["words"])
    filler_words = ["um", "uh", "like", "you know", "i mean", "so", "actually"]
    filler_word_data = []
    total_filler_duration = 0.0
    total_speech_duration = 0.0
    total_pause_duration = 0.0
    num_pauses = 0
    previous_word_end_time = None
    if word_segments:
        total_speech_duration = word_segments[-1]["end"] - word_segments[0]["start"]
        for word_info in word_segments:
            word_text = word_info["word"].lower().strip(".,!?")
            start_time = word_info["start"]
            end_time = word_info["end"]
            duration = end_time - start_time
            if word_text in filler_words:
                filler_word_data.append({
                    "word": word_text,
                    "start": start_time,
                    "end": end_time,
                    "duration": duration
                })
                total_filler_duration += duration
            if previous_word_end_time is not None:
                pause_duration = start_time - previous_word_end_time
                if pause_duration > 0.1:
                    total_pause_duration += pause_duration
                    num_pauses += 1
            previous_word_end_time = end_time
    num_words = len(word_segments)
    speech_rate_wpm = (num_words / total_speech_duration) * 60 if total_speech_duration > 0 else 0
    return {
        "full_transcription": transcribed_text,
        "filler_word_count": len(filler_word_data),
        "total_filler_duration": total_filler_duration,
        "average_filler_duration": total_filler_duration / len(filler_word_data) if filler_word_data else 0,
        "total_pause_duration": total_pause_duration,
        "num_pauses": num_pauses,
        "average_pause_duration": total_pause_duration / num_pauses if num_pauses > 0 else 0,
        "speech_rate_wpm": speech_rate_wpm,
        "total_speech_duration": total_speech_duration,
        "word_details": word_segments
    }

#AGENT 1-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RoleValidatorAgent:
    def __init__(self, model=None):
        """Initialize with an optional pre-configured model instance."""
        self.model = model
    
    def validate_role(self, role):
        """
        Enhanced LLM-driven role validation with better meaningless input detection.
        Returns True if the role is meaningful and valid, False otherwise.
        """
        # Pre-filter obviously invalid inputs
        if not role or len(role.strip()) < 2:
            return False
        
        # Check for common meaningless patterns
        meaningless_patterns = [
            r'^[_\-\.]+$',  # Only underscores, dashes, or dots
            r'^[0-9]+$',    # Only numbers
            r'^[^a-zA-Z]*$', # No alphabetic characters
            r'^(.)\1{2,}$'   # Repeated single character (aaa, bbb, etc.)
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, role.strip()):
                return False
        
        prompt = f"""You are an expert HR professional and career counselor with 20+ years of experience in recruitment across all industries.

TASK: Determine if the following text represents a valid, real-world job role or position.

INPUT ROLE: "{role}"

VALIDATION CRITERIA:
✅ VALID if the role is:
- A real job title (e.g., "Software Engineer", "Marketing Manager", "Teacher")
- A recognizable profession (e.g., "Doctor", "Lawyer", "Chef") 
- A legitimate business role (e.g., "Sales Representative", "Data Analyst")
- Contains meaningful job-related keywords
- Could reasonably exist in any company or industry

❌ INVALID if the role is:
- Meaningless symbols, numbers, or characters (e.g., "123", "___", "...", "abc")
- Random words unrelated to work (e.g., "banana", "purple sky")
- Gibberish or nonsensical text (e.g., "asdfgh", "xyzxyz")
- Empty, whitespace, or single characters
- Offensive, inappropriate, or unprofessional content
- Clearly a test input or placeholder

INSTRUCTIONS:
1. Analyze the input carefully
2. Consider if this could realistically be posted on a job board
3. Think about whether someone would genuinely apply for this role
4. Respond with ONLY one word: "Valid" or "Invalid"
5. Do not provide explanations, reasoning, or additional text

RESPONSE: """

        try:
            # Use the provided model or create a new one
            model = self.model if self.model else genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            text = response.text.strip().lower()
            return "valid" in text and "invalid" not in text
        except Exception:
            return False

#AGENT 2-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class QuestionGeneratorAgent:
    def __init__(self, model=None):
        """Initialize with an optional pre-configured model instance."""
        self.model = model
    
    def generate_questions(self, role, interview_type="Job", level="Beginner"):
        """
        Generate 5 high-quality, realistic interview questions based on role, type, and level.
        """
        prompt = f"""You are a senior hiring manager with 15+ years of experience in talent acquisition and professional interviews across diverse industries.

TASK: Generate exactly 5 high-quality, realistic interview questions for the role of "{role}".

CONTEXT:
- Interview Type: {interview_type} (Academic / Internship / Job)
- Difficulty Level: {level} (Beginner / Intermediate / Advanced)

QUESTION REQUIREMENTS:
✅ Each question must be directly relevant to the "{role}" position.
✅ Adjust the complexity and focus based on the interview type and difficulty level.
✅ Include a mix of:
   - Behavioral questions (past experiences)
   - Technical or role-specific questions
   - Problem-solving scenarios
   - Motivation and cultural fit questions
❌ Avoid:
   - Generic questions that apply to any role
   - Yes/no questions
   - Overly complex or confusing questions
   - Personal or inappropriate questions

FORMAT:
- Number each question (1. 2. 3. 4. 5.)
- One question per line
- No additional explanations or answers
- Keep questions concise but comprehensive

EXAMPLE FORMAT:
1. [Behavioral question about past experience]
2. [Technical question relevant to the role]
3. [Problem-solving or scenario question]
4. [Motivation / cultural fit question]
5. [Career goals / role-specific question]

Generate exactly 5 interview questions for the role "{role}" considering the interview type "{interview_type}" and difficulty level "{level}"."""

        try:
            # Use the provided model or create a new one
            model = self.model if self.model else genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)

            questions = []
            for line in response.text.strip().split("\n"):
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    question = re.sub(r'^\d+\.\s*', '', line).strip()
                    if question and len(question) > 10:
                        questions.append(question)
            return questions[:5] if len(questions) >= 5 else questions
        except Exception:
            return []

#AGENT 3-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class AnswerEvaluatorAgent:
    def __init__(self, model=None):
        """Initialize with an optional pre-configured model instance."""
        self.model = model
    
    # Add audio_metrics parameter with a default value of None
    def evaluate_answer(self, question, answer, role, audio_metrics=None):
        """
        Enhanced answer evaluation with robust meaningless input detection and detailed feedback.
        Now accepts optional audio_metrics for more comprehensive evaluation.
        """
        # Pre-screening for obviously meaningless answers
        if not answer or len(answer.strip()) < 3:
            return {
                "scores": {metric: 0 for metric in EVAL_METRICS},
                "strengths": ["N/A - Answer too short or empty"],
                "areas": ["Provide a complete, detailed response to the question", "Address the specific question being asked"],
                "comment": "<strong>Strengths:</strong> N/A - Answer too short or empty<br><br><strong>Areas for Improvement:</strong> Provide a complete, detailed response to the question • Address the specific question being asked"
            }
        
        # Check for meaningless patterns
        meaningless_patterns = [
            r'^[_\-\.\s]+$',          # Only symbols and spaces
            r'^(.)\1{3,}$',           # Repeated characters (aaaa, bbbb)
            r'^[^a-zA-Z0-9\s]*$',     # No alphanumeric content
            r'^\w{1,2}$'              # Single or two characters only
        ]
        
        for pattern in meaningless_patterns:
            if re.match(pattern, answer.strip()):
                return {
                    "scores": {metric: 0 for metric in EVAL_METRICS},
                    "strengths": ["N/A - Meaningless input detected"],
                    "areas": ["Provide a genuine, thoughtful response", "Answer the question with relevant content"],
                    "comment": "<strong>Strengths:</strong> N/A - Meaningless input detected<br><br><strong>Areas for Improvement:</strong> Provide a genuine, thoughtful response • Answer the question with relevant content"
                }

        # Build prompt, incorporating audio_metrics if provided
        prompt_parts = [
            f"""You are an expert interview coach and HR professional with extensive experience in candidate assessment across all industries.

CONTEXT:
- Job Role: {role}
- Interview Question: {question}
- Candidate Answer: {answer}

TASK: Evaluate this interview answer comprehensively and provide actionable feedback.

EVALUATION PROCESS:

STEP 1 - MEANINGFULNESS CHECK (CRITICAL):
First, determine if the answer is meaningful and relevant:
- Does it contain actual words and sentences?
- Does it attempt to address the question?
- Is it more than just symbols, single words, or gibberish?

⚠️ IMPORTANT: If the answer is ANY of the following, assign ALL scores as 0:
- Single words or very short responses (under 10 characters)
- Repetitive characters (xxx, aaa, 123, etc.)
- Nonsensical text or gibberish
- Completely unrelated to the question
- Just symbols, punctuation, or meaningless input

If meaningless, set all scores to 0 and explain why in strengths/areas.

STEP 2 - DETAILED SCORING (only if answer is meaningful):
Rate each metric from 1-10 considering the role "{role}":

🎯 CLARITY (1-10):
- How clear and understandable is the communication?
- Is the message easy to follow and well-articulated?
- Are ideas expressed in a logical, coherent manner?

🧠 KNOWLEDGE (1-10):
- Does the answer demonstrate relevant knowledge for "{role}"?
- Are technical concepts, industry terms, or role-specific skills mentioned appropriately?
- Is there evidence of understanding the field/industry?

⚡ CONCISENESS (1-10):
- Is the answer appropriately detailed without being verbose?
- Does it avoid unnecessary repetition or tangents?
- Is information presented efficiently?
"""
        ]

        # Add audio metrics to the prompt if available
        if audio_metrics:
            prompt_parts.append(f"""
Speaking Metrics (for Confidence and Fluency assessment):
- Total speech duration (excluding long pauses): {audio_metrics.get("total_speech_duration", 0):.2f} seconds
- Number of filler words detected: {audio_metrics.get("filler_word_count", 0)}
- Total duration of filler words: {audio_metrics.get("total_filler_duration", 0):.2f} seconds
- Number of significant pauses (>0.1s): {audio_metrics.get("num_pauses", 0)}
- Total duration of these pauses: {audio_metrics.get("total_pause_duration", 0):.2f} seconds
- Estimated speech rate: {audio_metrics.get("speech_rate_wpm", 0):.0f} words per minute
""")
            prompt_parts.append("""
💪 CONFIDENCE (1-10):
- Does the candidate sound sure of their response?
- Are statements made with appropriate conviction?
- Is there evidence of self-assurance without arrogance?
- **Consider the provided speaking metrics**: High filler word count or long/frequent pauses might indicate lower confidence. A steady speech rate with minimal hesitation might indicate higher confidence.

🏗️ STRUCTURE (1-10):
- Is the answer well-organized with logical flow?
- Are points presented in a coherent sequence?
- Is there a clear beginning, middle, and end?
- **Consider speaking metrics**: Disorganized speech flow might correlate with more pauses or hesitations.
""")
        else: # Original Confidence and Structure if no audio metrics
            prompt_parts.append("""
💪 CONFIDENCE (1-10):
- Does the candidate sound sure of their response?
- Are statements made with appropriate conviction?
- Is there evidence of self-assurance without arrogance?

🏗️ STRUCTURE (1-10):
- Is the answer well-organized with logical flow?
- Are points presented in a coherent sequence?
- Is there a clear beginning, middle, and end?
""")

        prompt_parts.append(f"""
STEP 3 - FEEDBACK GENERATION (CONCISE):
Provide exactly 2 strengths and 2 improvement points.

STYLE RULES (STRICT):
- Keep each bullet ultra-brief (max 12 words), concrete, and punchy
- Start with a strong noun or verb, no filler words
- Focus on the most impactful observation only

If the answer is meaningless, set strengths to ["N/A - Invalid response"], and areas should instruct to provide a real, relevant answer.

OUTPUT FORMAT (return ONLY valid JSON, no markdown or extra text):
{{
  "scores": {{
    "Clarity": <integer 0-10>,
    "Knowledge": <integer 0-10>,
    "Conciseness": <integer 0-10>,
    "Confidence": <integer 0-10>,
    "Structure": <integer 0-10>
  }},
  "strengths": ["specific, brief strength 1", "specific, brief strength 2"],
  "areas": ["brief, actionable improvement 1", "brief, actionable improvement 2"]
}}""")

        prompt = "".join(prompt_parts)

        try:
            # Use the provided model or create a new one
            model = self.model if self.model else genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            raw = response.text.strip()
            
            # Extract JSON from response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            
            if start == -1 or end == 0:
                raise ValueError("No valid JSON found in response")
            
            feedback = json.loads(raw[start:end])
            
            # Validate and clamp scores
            if "scores" not in feedback:
                feedback["scores"] = {}
                
            for metric in EVAL_METRICS:
                score = feedback["scores"].get(metric, 0)
                # Ensure score is integer between 0-10
                try:
                    score = int(float(score))
                    feedback["scores"][metric] = max(0, min(10, score))
                except (ValueError, TypeError):
                    feedback["scores"][metric] = 0
            
            # Ensure strengths and areas exist and are lists
            if "strengths" not in feedback or not isinstance(feedback["strengths"], list):
                feedback["strengths"] = ["Response provided", "Attempted to answer question"]
                
            if "areas" not in feedback or not isinstance(feedback["areas"], list):
                feedback["areas"] = ["Provide more specific examples", "Elaborate on key points"]
            
            # Ensure exactly 2 items in each list
            feedback["strengths"] = feedback["strengths"][:2]
            feedback["areas"] = feedback["areas"][:2]
            
            # Compress items to be brief (max 12 words)
            def _compress_items(items):
                compact = []
                for text in items:
                    try:
                        words = str(text).split()
                        compact.append(" ".join(words[:12]))
                    except Exception:
                        compact.append(str(text))
                return compact
            feedback["strengths"] = _compress_items(feedback["strengths"])
            feedback["areas"] = _compress_items(feedback["areas"])
            
            # Fill in if less than 2 items
            while len(feedback["strengths"]) < 2:
                feedback["strengths"].append("Provided a response to the question")
                
            while len(feedback["areas"]) < 2:
                feedback["areas"].append("Consider providing more detailed examples")
            
            # Add backward compatibility comment field
            strengths_text = " • ".join(feedback["strengths"])
            areas_text = " • ".join(feedback["areas"])
            feedback["comment"] = f"<strong>Strengths:</strong> {strengths_text}<br><br><strong>Areas for Improvement:</strong> {areas_text}"
            
            return feedback
        
        except Exception as e:
            # Fallback response in case of any errors
            print(f"Error in AnswerEvaluatorAgent: {e}") # Debugging
            return {
                "scores": {metric: 1 for metric in EVAL_METRICS},
                "strengths": ["Attempted to provide an answer", "Engaged with the question"],
                "areas": ["Provide more detailed responses", "Include specific examples and experiences"],
                "comment": "<strong>Strengths:</strong> Attempted to provide an answer • Engaged with the question<br><br><strong>Areas for Improvement:</strong> Provide more detailed responses • Include specific examples and experiences"
            }

#AGENT 4-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class VideoAnalyzerAgent:
    def __init__(self, model=None):
        """Initialize with an optional pre-configured model instance."""
        self.model = model
        
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

    def analyze_video(self, video_path):
        """Main video processing function"""
        try:
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

            # Generate results
            highlights = self.generate_highlights()
            summary = self.generate_summary(duration)
            
            return {
                "video_summary": summary,
                "video_highlights": highlights,
                "video_report": f"Video analysis completed. Processed {len(self.timeline)} samples over {duration:.1f} seconds.",
                "duration": duration,
                "output_path": output_path
            }

        except Exception as e:
            print(f"Error in video analysis: {e}")
            return {
                "video_summary": {"error": str(e)},
                "video_report": f"Video analysis failed: {str(e)}",
                "duration": 0,
                "output_path": None
            }

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

# Integrated feedback system (from feedback_system.py)
class CSVFeedbackSystem:
    """Simple CSV-based feedback system for storing and learning from user feedback"""
    
    def __init__(self, data_dir: str = "feedback_data"):
        self.data_dir = data_dir
        self.sessions_file = os.path.join(data_dir, "interview_sessions.csv")
        self.question_feedback_file = os.path.join(data_dir, "question_feedback.csv")
        self.session_feedback_file = os.path.join(data_dir, "session_feedback.csv")
        self.learning_insights_file = os.path.join(data_dir, "learning_insights.csv")
        os.makedirs(data_dir, exist_ok=True)
        self._init_csv_files()
    
    def _init_csv_files(self):
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'session_id', 'role', 'interview_type', 'level', 'interview_mode',
                    'created_at', 'completed_at', 'total_questions', 'overall_score'
                ])
        if not os.path.exists(self.question_feedback_file):
            with open(self.question_feedback_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'session_id', 'question_number', 'question_text', 'user_answer',
                    'ai_feedback', 'question_rating', 'feedback_accuracy_rating',
                    'question_relevance_rating', 'user_corrections', 'created_at'
                ])
        if not os.path.exists(self.session_feedback_file):
            with open(self.session_feedback_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'session_id', 'overall_experience_rating', 'ai_helpfulness_rating',
                    'feature_ratings', 'improvement_suggestions', 'would_recommend', 'created_at'
                ])
        if not os.path.exists(self.learning_insights_file):
            with open(self.learning_insights_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'role', 'insight_type', 'insight_data', 'confidence_score',
                    'sample_size', 'last_updated'
                ])
    
    def create_session(self, role: str, interview_type: str, level: str, interview_mode: str) -> str:
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        with open(self.sessions_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id, role, interview_type, level, interview_mode,
                timestamp, '', '', ''
            ])
        return session_id
    
    def update_session_completion(self, session_id: str, total_questions: int, overall_score: float):
        timestamp = datetime.now().isoformat()
        sessions = []
        with open(self.sessions_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['session_id'] == session_id:
                    row['completed_at'] = timestamp
                    row['total_questions'] = str(total_questions)
                    row['overall_score'] = str(overall_score)
                sessions.append(row)
        with open(self.sessions_file, 'w', newline='', encoding='utf-8') as f:
            if sessions:
                fieldnames = sessions[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sessions)
    
    def add_question_feedback(self, session_id: str, question_number: int, question_text: str,
                            user_answer: str, ai_feedback: str, question_rating: int = None,
                            feedback_accuracy_rating: int = None, question_relevance_rating: int = None,
                            user_corrections: Dict = None):
        timestamp = datetime.now().isoformat()
        with open(self.question_feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id, question_number, question_text, user_answer, ai_feedback,
                question_rating or '', feedback_accuracy_rating or '', question_relevance_rating or '',
                json.dumps(user_corrections) if user_corrections else '', timestamp
            ])
    
    def add_session_feedback(self, session_id: str, overall_experience_rating: int = None,
                           ai_helpfulness_rating: int = None, feature_ratings: Dict = None,
                           improvement_suggestions: str = None, would_recommend: bool = None):
        timestamp = datetime.now().isoformat()
        with open(self.session_feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                session_id, overall_experience_rating or '', ai_helpfulness_rating or '',
                json.dumps(feature_ratings) if feature_ratings else '', improvement_suggestions or '',
                would_recommend or '', timestamp
            ])
    
    def get_role_feedback_data(self, role: str, days: int = 30) -> Dict:
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        sessions = []
        with open(self.sessions_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['role'] == role:
                    try:
                        session_time = datetime.fromisoformat(row['created_at']).timestamp()
                        if session_time >= cutoff_date:
                            sessions.append(row)
                    except:
                        continue
        question_feedback = []
        session_ids = [s['session_id'] for s in sessions]
        with open(self.question_feedback_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['session_id'] in session_ids:
                    question_feedback.append(row)
        session_feedback = []
        with open(self.session_feedback_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['session_id'] in session_ids:
                    session_feedback.append(row)
        return {
            'sessions': sessions,
            'question_feedback': question_feedback,
            'session_feedback': session_feedback
        }
    
    def save_learning_insights(self, role: str, insight_type: str, insight_data: Dict, confidence_score: float = 0.5):
        timestamp = datetime.now().isoformat()
        existing_insights = []
        insight_exists = False
        try:
            with open(self.learning_insights_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['role'] == role and row['insight_type'] == insight_type:
                        row['insight_data'] = json.dumps(insight_data)
                        row['confidence_score'] = str(confidence_score)
                        row['sample_size'] = str(int(row.get('sample_size', 1)) + 1)
                        row['last_updated'] = timestamp
                        insight_exists = True
                    existing_insights.append(row)
        except FileNotFoundError:
            pass
        if not insight_exists:
            existing_insights.append({
                'role': role,
                'insight_type': insight_type,
                'insight_data': json.dumps(insight_data),
                'confidence_score': str(confidence_score),
                'sample_size': '1',
                'last_updated': timestamp
            })
        with open(self.learning_insights_file, 'w', newline='', encoding='utf-8') as f:
            if existing_insights:
                fieldnames = existing_insights[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_insights)
    
    def get_learning_insights(self, role: str, insight_type: str = None) -> Dict:
        insights = {}
        try:
            with open(self.learning_insights_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['role'] == role:
                        if insight_type is None or row['insight_type'] == insight_type:
                            try:
                                data = json.loads(row['insight_data'])
                                insights[row['insight_type']] = {
                                    'data': data,
                                    'confidence': float(row['confidence_score']),
                                    'sample_size': int(row['sample_size']),
                                    'last_updated': row['last_updated']
                                }
                            except:
                                continue
        except FileNotFoundError:
            pass
        return insights

# Integrated learning agent and adaptive components (from learning_agent.py)
class LearningAgent:
    """Learning agent that analyzes feedback and improves the system"""
    
    def __init__(self, feedback_system: CSVFeedbackSystem = None):
        self.feedback_system = feedback_system or CSVFeedbackSystem()
        self.model = genai.GenerativeModel("gemini-2.5-flash")
    
    def analyze_feedback_and_learn(self, role: str, days: int = 30):
        print(f"🤖 Learning Agent: Analyzing feedback for role '{role}'...")
        feedback_data = self.feedback_system.get_role_feedback_data(role, days)
        if not feedback_data['sessions']:
            print(f"📊 No feedback data found for role '{role}' in the last {days} days")
            return
        question_insights = self._analyze_question_quality(feedback_data, role)
        if question_insights:
            self.feedback_system.save_learning_insights(
                role, 'question_quality', question_insights,
                confidence_score=min(len(feedback_data['question_feedback']) / 20, 1.0)
            )
            print(f"✅ Question quality insights saved for '{role}'")
        accuracy_insights = self._analyze_feedback_accuracy(feedback_data, role)
        if accuracy_insights:
            self.feedback_system.save_learning_insights(
                role, 'feedback_accuracy', accuracy_insights,
                confidence_score=min(len(feedback_data['question_feedback']) / 15, 1.0)
            )
            print(f"✅ Feedback accuracy insights saved for '{role}'")
        pattern_insights = self._analyze_overall_patterns(feedback_data, role)
        if pattern_insights:
            self.feedback_system.save_learning_insights(
                role, 'overall_patterns', pattern_insights,
                confidence_score=min(len(feedback_data['sessions']) / 10, 1.0)
            )
            print(f"✅ Overall pattern insights saved for '{role}'")
    
    def _analyze_question_quality(self, feedback_data: Dict, role: str) -> Dict:
        question_feedback = feedback_data['question_feedback']
        if not question_feedback:
            return {}
        high_rated = []
        low_rated = []
        common_issues = []
        for qf in question_feedback:
            try:
                question_rating = int(qf.get('question_rating', 0)) if qf.get('question_rating') else 0
                relevance_rating = int(qf.get('question_relevance_rating', 0)) if qf.get('question_relevance_rating') else 0
                avg_rating = (question_rating + relevance_rating) / 2 if (question_rating and relevance_rating) else 0
                if avg_rating >= 4:
                    high_rated.append({'question': qf.get('question_text', ''), 'rating': avg_rating})
                elif avg_rating <= 2 and avg_rating > 0:
                    low_rated.append({'question': qf.get('question_text', ''), 'rating': avg_rating})
                corrections = qf.get('user_corrections', '')
                if corrections:
                    try:
                        corrections_data = json.loads(corrections)
                        if isinstance(corrections_data, dict):
                            common_issues.extend(corrections_data.get('issues', []))
                    except:
                        pass
            except (ValueError, TypeError):
                continue
        if high_rated or low_rated:
            return self._generate_question_insights(role, high_rated, low_rated, common_issues)
        return {}
    
    def _analyze_feedback_accuracy(self, feedback_data: Dict, role: str) -> Dict:
        question_feedback = feedback_data['question_feedback']
        if not question_feedback:
            return {}
        high_accuracy = []
        low_accuracy = []
        common_corrections = []
        for qf in question_feedback:
            try:
                accuracy_rating = int(qf.get('feedback_accuracy_rating', 0)) if qf.get('feedback_accuracy_rating') else 0
                if accuracy_rating >= 4:
                    high_accuracy.append({'question': qf.get('question_text', ''), 'answer': qf.get('user_answer', ''), 'feedback': qf.get('ai_feedback', ''), 'rating': accuracy_rating})
                elif accuracy_rating <= 2 and accuracy_rating > 0:
                    low_accuracy.append({'question': qf.get('question_text', ''), 'answer': qf.get('user_answer', ''), 'feedback': qf.get('ai_feedback', ''), 'rating': accuracy_rating})
                corrections = qf.get('user_corrections', '')
                if corrections:
                    try:
                        corrections_data = json.loads(corrections)
                        if isinstance(corrections_data, dict):
                            common_corrections.extend(corrections_data.get('corrections', []))
                    except:
                        pass
            except (ValueError, TypeError):
                continue
        if high_accuracy or low_accuracy:
            return self._generate_accuracy_insights(role, high_accuracy, low_accuracy, common_corrections)
        return {}
    
    def _analyze_overall_patterns(self, feedback_data: Dict, role: str) -> Dict:
        session_feedback = feedback_data['session_feedback']
        sessions = feedback_data['sessions']
        if not session_feedback:
            return {}
        experience_ratings = []
        helpfulness_ratings = []
        recommendations = []
        suggestions = []
        for sf in session_feedback:
            try:
                exp_rating = int(sf.get('overall_experience_rating', 0)) if sf.get('overall_experience_rating') else 0
                help_rating = int(sf.get('ai_helpfulness_rating', 0)) if sf.get('ai_helpfulness_rating') else 0
                recommend = sf.get('would_recommend', '').lower() == 'true'
                suggestion = sf.get('improvement_suggestions', '')
                if exp_rating > 0:
                    experience_ratings.append(exp_rating)
                if help_rating > 0:
                    helpfulness_ratings.append(help_rating)
                if recommend is not None:
                    recommendations.append(recommend)
                if suggestion:
                    suggestions.append(suggestion)
            except (ValueError, TypeError):
                continue
        avg_experience = sum(experience_ratings) / len(experience_ratings) if experience_ratings else 0
        avg_helpfulness = sum(helpfulness_ratings) / len(helpfulness_ratings) if helpfulness_ratings else 0
        recommendation_rate = sum(recommendations) / len(recommendations) if recommendations else 0
        return {
            'avg_experience_rating': round(avg_experience, 2),
            'avg_helpfulness_rating': round(avg_helpfulness, 2),
            'recommendation_rate': round(recommendation_rate * 100, 1),
            'total_sessions': len(sessions),
            'improvement_suggestions': suggestions[:5]
        }
    
    def _generate_question_insights(self, role: str, high_rated: List, low_rated: List, issues: List) -> Dict:
        try:
            prompt = f"""Analyze question feedback for the role of "{role}".

HIGH-RATED QUESTIONS (4+ stars):
{json.dumps(high_rated[:5], indent=2)}

LOW-RATED QUESTIONS (2 stars or less):
{json.dumps(low_rated[:5], indent=2)}

COMMON ISSUES:
{json.dumps(issues[:10], indent=2)}

Provide insights in this JSON format:
{{
    "successful_patterns": ["pattern1", "pattern2"],
    "problematic_patterns": ["pattern1", "pattern2"],
    "recommendations": ["recommendation1", "recommendation2"],
    "question_templates": ["template1", "template2"]
}}"""
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ Error generating question insights: {e}")
            return {
                "successful_patterns": [],
                "problematic_patterns": [],
                "recommendations": ["Continue collecting feedback to improve question quality"],
                "question_templates": []
            }
    
    def _generate_accuracy_insights(self, role: str, high_accuracy: List, low_accuracy: List, corrections: List) -> Dict:
        try:
            prompt = f"""Analyze feedback accuracy for the role of "{role}".

HIGH ACCURACY CASES (4+ stars):
{json.dumps(high_accuracy[:3], indent=2)}

LOW ACCURACY CASES (2 stars or less):
{json.dumps(low_accuracy[:3], indent=2)}

COMMON CORRECTIONS:
{json.dumps(corrections[:10], indent=2)}

Provide insights in this JSON format:
{{
    "accurate_patterns": ["pattern1", "pattern2"],
    "inaccurate_patterns": ["pattern1", "pattern2"],
    "common_misunderstandings": ["misunderstanding1", "misunderstanding2"],
    "improvement_recommendations": ["recommendation1", "recommendation2"]
}}"""
            response = self.model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(f"❌ Error generating accuracy insights: {e}")
            return {
                "accurate_patterns": [],
                "inaccurate_patterns": [],
                "common_misunderstandings": [],
                "improvement_recommendations": ["Continue collecting feedback to improve accuracy"]
            }

class AdaptiveQuestionGenerator:
    def __init__(self, master_agent: MasterAgent, learning_agent: LearningAgent):
        self.master_agent = master_agent
        self.learning_agent = learning_agent
    
    def generate_adaptive_questions(self, role: str, interview_type: str, level: str) -> List[str]:
        insights = self.learning_agent.feedback_system.get_learning_insights(role, 'question_quality')
        if not insights or insights.get('question_quality', {}).get('confidence', 0) < 0.3:
            return self.master_agent.generate_questions(role, interview_type, level)
        question_data = insights['question_quality']['data']
        successful_patterns = question_data.get('successful_patterns', [])
        recommendations = question_data.get('recommendations', [])
        enhanced_prompt = self._create_enhanced_prompt(
            role, interview_type, level, successful_patterns, recommendations
        )
        try:
            response = self.learning_agent.model.generate_content(enhanced_prompt)
            questions = []
            for line in response.text.strip().split("\n"):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or any(line.startswith(f"{i}.") for i in range(1, 10))):
                    question = line.split('.', 1)[1].strip() if '.' in line else line
                    if question and len(question) > 10:
                        questions.append(question)
            return questions[:5] if questions else self.master_agent.generate_questions(role, interview_type, level)
        except Exception as e:
            print(f"❌ Error in adaptive question generation: {e}")
            return self.master_agent.generate_questions(role, interview_type, level)
    
    def _create_enhanced_prompt(self, role: str, interview_type: str, level: str, 
                              successful_patterns: List, recommendations: List) -> str:
        import random
        import time
        variety_seed = int(time.time()) % 1000
        random.seed(variety_seed)
        question_starters = [
            "Generate exactly 5 unique, high-quality questions",
            "Create exactly 5 diverse, professional questions", 
            "Develop exactly 5 comprehensive, engaging questions",
            "Formulate exactly 5 well-crafted, relevant questions"
        ]
        starter = random.choice(question_starters)
        base_prompt = f"""You are a senior hiring manager generating interview questions for "{role}".

CONTEXT:
- Interview Type: {interview_type}
- Difficulty Level: {level}
- Variety Seed: {variety_seed} (ensure questions are different from previous sessions)

LEARNED INSIGHTS FROM USER FEEDBACK:
- Successful patterns: {', '.join(successful_patterns) if successful_patterns else 'None available yet'}
- Recommendations: {', '.join(recommendations) if recommendations else 'None available yet'}

{starter} that:
✅ Are directly relevant to "{role}"
✅ Match the {level} difficulty level
✅ Incorporate successful patterns from user feedback
✅ Include behavioral, technical, and problem-solving questions
✅ Are DIFFERENT from questions you've generated before (use variety seed for uniqueness)
❌ Avoid patterns that users have rated poorly

IMPORTANT: Make sure these questions are fresh and different from previous interview sessions. Use the variety seed to ensure uniqueness.

Format: Number each question (1. 2. 3. 4. 5.) with one question per line."""
        return base_prompt

class AdaptiveAnswerEvaluator:
    def __init__(self, master_agent: MasterAgent, learning_agent: LearningAgent):
        self.master_agent = master_agent
        self.learning_agent = learning_agent
    
    def evaluate_answer_adaptive(self, question: str, answer: str, role: str, audio_metrics: Dict = None) -> Dict:
        insights = self.learning_agent.feedback_system.get_learning_insights(role, 'feedback_accuracy')
        if not insights or insights.get('feedback_accuracy', {}).get('confidence', 0) < 0.3:
            return self.master_agent.evaluate_answer(question, answer, role, audio_metrics)
        standard_feedback = self.master_agent.evaluate_answer(question, answer, role, audio_metrics)
        accuracy_data = insights['feedback_accuracy']['data']
        accurate_patterns = accuracy_data.get('accurate_patterns', [])
        recommendations = accuracy_data.get('improvement_recommendations', [])
        if accurate_patterns or recommendations:
            learning_note = "This feedback has been enhanced using insights from user feedback for this role."
            if standard_feedback.get('comment'):
                standard_feedback['comment'] += f"<br><br><em>{learning_note}</em>"
            else:
                standard_feedback['comment'] = learning_note
        return standard_feedback
    
    def generate_combined_report(self, video_summary, video_highlights, audio_metrics, video_duration):
        """Generate AI-powered combined video and audio analysis report"""
        try:
            # Use the provided model or create a new one
            model = self.model if self.model else genai.GenerativeModel("gemini-2.5-flash")
            
            # Create comprehensive prompt for Gemini
            prompt = f"""You are an expert interview coach analyzing a candidate's performance from both video and audio perspectives. Based on the following data, provide a comprehensive, actionable feedback report.

VIDEO ANALYSIS SUMMARY:
- Overall Performance Score: {video_summary.get('overall_performance_score', 'N/A')}/100
- Eye Contact (Good): {video_summary.get('eye_contact_good_percent', 'N/A')}%
- Posture (Upright): {video_summary.get('posture_upright_percent', 'N/A')}%
- Hand Visibility: {video_summary.get('hands_visible_percent', 'N/A')}%
- Dominant Emotion: {video_summary.get('dominant_emotion', 'N/A')}
- Fidgeting Events: {video_summary.get('fidget_events', 'N/A')}
- Nervous Behavior Score: {video_summary.get('nervous_ticks_total', 'N/A')}
- Video Duration: {video_duration:.1f} seconds

VIDEO EMOTION BREAKDOWN:
{json.dumps(video_summary.get('emotion_distribution', {}), indent=2)}

VIDEO PROBLEMATIC MOMENTS:
{json.dumps(video_highlights, indent=2)}

AUDIO ANALYSIS METRICS:
- Speech Duration: {audio_metrics.get('total_speech_duration', 0):.2f} seconds
- Filler Words: {audio_metrics.get('filler_word_count', 0)} (Duration: {audio_metrics.get('total_filler_duration', 0):.2f}s)
- Pauses: {audio_metrics.get('num_pauses', 0)} (Duration: {audio_metrics.get('total_pause_duration', 0):.2f}s)
- Speech Rate: {audio_metrics.get('speech_rate_wpm', 0):.0f} words per minute
- Average Pause Duration: {audio_metrics.get('average_pause_duration', 0):.2f} seconds

TRANSCRIPTION:
{audio_metrics.get('full_transcription', 'No transcription available')}

Please provide a comprehensive analysis that includes:
1. Overall assessment combining video and audio insights (3-4 sentences)
2. Top 3 strengths from both video and audio analysis
3. Top 3 areas for improvement with specific examples from both modalities
4. 5 actionable practice recommendations that address both visual and verbal aspects
5. Specific timestamps for concerning moments (if any)
6. Encouragement and next steps

Keep the tone supportive but honest. Use specific data points when referencing performance metrics.
"""

            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            return self.generate_fallback_combined_report(video_summary, video_highlights, audio_metrics, video_duration)
    
    def generate_fallback_combined_report(self, video_summary, video_highlights, audio_metrics, video_duration):
        """Generate local combined feedback when Gemini is unavailable"""
        video_score = video_summary.get('overall_performance_score', 50)
        
        # Analyze audio metrics
        audio_score = 100
        if audio_metrics.get('filler_word_count', 0) > 5:
            audio_score -= 20
        if audio_metrics.get('num_pauses', 0) > 10:
            audio_score -= 15
        if audio_metrics.get('speech_rate_wpm', 0) < 120 or audio_metrics.get('speech_rate_wpm', 0) > 200:
            audio_score -= 10
        
        combined_score = (video_score + audio_score) / 2
        
        feedback_lines = [
            "Combined Video & Audio Interview Analysis",
            "==========================================",
            "",
            f"Overall Performance: {combined_score:.1f}/100",
            f"Video Analysis Score: {video_score:.1f}/100",
            f"Audio Analysis Score: {audio_score:.1f}/100",
            "",
            "VIDEO ANALYSIS:",
            "---------------",
            f"• Eye Contact: {video_summary.get('eye_contact_good_percent', 'N/A')}% good/excellent",
            f"• Posture: {video_summary.get('posture_upright_percent', 'N/A')}% upright",
            f"• Hand Visibility: {video_summary.get('hands_visible_percent', 'N/A')}%",
            f"• Dominant Emotion: {video_summary.get('dominant_emotion', 'N/A')}",
            f"• Fidgeting Events: {video_summary.get('fidget_events', 'N/A')}",
            "",
            "AUDIO ANALYSIS:",
            "---------------",
            f"• Speech Duration: {audio_metrics.get('total_speech_duration', 0):.2f} seconds",
            f"• Filler Words: {audio_metrics.get('filler_word_count', 0)}",
            f"• Pauses: {audio_metrics.get('num_pauses', 0)}",
            f"• Speech Rate: {audio_metrics.get('speech_rate_wpm', 0):.0f} wpm",
            "",
            "STRENGTHS:",
            "----------"
        ]
        
        # Dynamic strengths
        strengths = []
        if video_summary.get('eye_contact_good_percent', 0) >= 70:
            strengths.append("• Good eye contact and engagement")
        if video_summary.get('posture_upright_percent', 0) >= 75:
            strengths.append("• Professional posture and body language")
        if audio_metrics.get('filler_word_count', 0) <= 3:
            strengths.append("• Clear speech with minimal filler words")
        if 120 <= audio_metrics.get('speech_rate_wpm', 0) <= 200:
            strengths.append("• Appropriate speaking pace")
        
        if not strengths:
            strengths.append("• Professional appearance and demeanor")
            strengths.append("• Willingness to practice and improve")
        
        feedback_lines.extend(strengths)
        feedback_lines.extend([
            "",
            "AREAS FOR IMPROVEMENT:",
            "---------------------"
        ])
        
        # Dynamic improvement areas
        improvements = []
        if video_summary.get('eye_contact_good_percent', 0) < 60:
            improvements.append("• Work on maintaining consistent eye contact")
        if video_summary.get('posture_upright_percent', 0) < 70:
            improvements.append("• Practice maintaining upright, confident posture")
        if audio_metrics.get('filler_word_count', 0) > 5:
            improvements.append("• Reduce use of filler words (um, uh, like)")
        if audio_metrics.get('num_pauses', 0) > 10:
            improvements.append("• Work on reducing unnecessary pauses")
        
        if not improvements:
            improvements.append("• Continue practicing to maintain excellent performance")
        
        feedback_lines.extend(improvements)
        
        # Highlight specific moments
        if video_highlights:
            feedback_lines.extend([
                "",
                "CONCERNING MOMENTS:",
                "-------------------"
            ])
            for h in video_highlights[:3]:  # Top 3 highlights
                feedback_lines.append(f"• {h.get('start_time', 'N/A')}-{h.get('end_time', 'N/A')}: {', '.join(h.get('reasons', []))}")
        
        feedback_lines.extend([
            "",
            "PRACTICE RECOMMENDATIONS:",
            "-------------------------",
            "• Record mock interviews and review both video and audio",
            "• Practice maintaining eye contact while speaking",
            "• Work on reducing filler words and improving speech clarity",
            "• Practice the STAR method for structured responses",
            "• Use breathing exercises to reduce nervous energy",
            "",
            "Keep practicing - improvement comes with consistent effort! 💪"
        ])
        
        return "\n".join(feedback_lines)


# ================== CAREER AUTOMATION SYSTEM ==================
# Integrated multi-agent career automation system
# Uses MasterAgent's Gemini model when available, otherwise falls back to local configuration

# Additional imports for career automation
import os
import json as _json
import time as _time
import random as _random
import logging as _logging
import re as _re
from datetime import datetime as _dt
from urllib.parse import urljoin as _urljoin, quote as _quote
import urllib.parse as _urllib_parse

# Document processing imports (guarded to prevent errors if not installed)
try:
    from docx import Document
except ImportError:
    Document = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Web scraping imports
try:
    import requests
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    requests = None
    BeautifulSoup = None
    webdriver = None
    By = None
    WebDriverWait = None
    EC = None
    TimeoutException = None
    NoSuchElementException = None

# Template rendering and PDF generation
try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    Environment = None
    FileSystemLoader = None

try:
    import pdfkit
except ImportError:
    pdfkit = None

try:
    from weasyprint import HTML
except ImportError:
    HTML = None

# Text similarity
try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

# Document processing
try:
    from docx import Document as _Document
except Exception:
    _Document = None
try:
    import PyPDF2 as _PyPDF2
except Exception:
    _PyPDF2 = None

# Web scraping
try:
    import requests as _requests
    from bs4 import BeautifulSoup as _BeautifulSoup
except Exception:
    _requests = None
    _BeautifulSoup = None
try:
    from selenium import webdriver as _webdriver
    from selenium.webdriver.common.by import By as _By
    from selenium.webdriver.support.ui import WebDriverWait as _WebDriverWait
    from selenium.webdriver.support import expected_conditions as _EC
except Exception:
    _webdriver = None
    _By = None
    _WebDriverWait = None
    _EC = None

# Template rendering
try:
    from jinja2 import Environment as _J2Env, FileSystemLoader as _J2Loader
except Exception:
    _J2Env = None
    _J2Loader = None

# Text similarity
try:
    from rapidfuzz import fuzz as _fuzz
except Exception:
    _fuzz = None

import google.generativeai as _genai

_logger = _logging.getLogger(__name__)


class CareerAutomationSystem:
    """
    Complete career automation system that integrates all agents:
    - Agent 1: CV Tailoring
    - Agent 2: Cover Letter Generation
    - Agent 3: Job Scraping
    - Agent 4: Strengths/Weaknesses Analysis
    - Agent 5: Course Suggestions
    - Agent 6: Weekly Learning Plan
    - Agent 7: Career Roadmap Planning
    """

    def __init__(self, gemini_api_key=None, model=None):
        """Initialize the system with Gemini API or an injected model.
        If 'model' is provided (e.g., MasterAgent().get_model()), it will be used.
        Otherwise, configure Gemini using provided api key or existing env config.
        """
        if model is not None:
            self.model = model
        else:
            if gemini_api_key:
                _genai.configure(api_key=gemini_api_key)
            # If no key is provided, assume already configured elsewhere
            self.model = _genai.GenerativeModel("gemini-2.5-flash")

        # Initialize web scraping session
        self.session = (_requests.Session() if _requests else None)
        if self.session:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            })

        # Setup Chrome options for Selenium
        self.driver = None
        if _webdriver:
            try:
                options = _webdriver.ChromeOptions()
                options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
                options.add_experimental_option("excludeSwitches", ["enable-automation"])
                options.add_experimental_option('useAutomationExtension', False)
                self.driver = _webdriver.Chrome(options=options)
                try:
                    self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                except Exception:
                    pass
            except Exception as _e:
                _logger.error(f"Selenium initialization failed: {_e}")
                self.driver = None

        # Pakistani job sites configuration (enhanced with more reliable sources)
        self.newspapers = {
            'Dawn': {
                'base_url': 'https://www.dawn.com',
                'jobs_url': 'https://www.dawn.com/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"], a[href*="employment"]',
                    'job_titles': 'h1, h2, h3, .title, .headline',
                    'job_details': '.content, .description, .summary'
                }
            },
            'The Express Tribune': {
                'base_url': 'https://tribune.com.pk',
                'jobs_url': 'https://tribune.com.pk/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'The News International': {
                'base_url': 'https://www.thenews.com.pk',
                'jobs_url': 'https://www.thenews.com.pk/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'Daily Times': {
                'base_url': 'https://dailytimes.com.pk',
                'jobs_url': 'https://dailytimes.com.pk/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'The Nation': {
                'base_url': 'https://nation.com.pk',
                'jobs_url': 'https://nation.com.pk/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'Business Recorder': {
                'base_url': 'https://www.brecorder.com',
                'jobs_url': 'https://www.brecorder.com/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'Pakistan Today': {
                'base_url': 'https://www.pakistantoday.com.pk',
                'jobs_url': 'https://www.pakistantoday.com.pk/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'The Frontier Post': {
                'base_url': 'https://thefrontierpost.com',
                'jobs_url': 'https://thefrontierpost.com/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'Daily Pakistan': {
                'base_url': 'https://en.dailypakistan.com.pk',
                'jobs_url': 'https://en.dailypakistan.com.pk/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            },
            'ARY News': {
                'base_url': 'https://arynews.tv',
                'jobs_url': 'https://arynews.tv/jobs',
                'selectors': {
                    'job_links': 'a[href*="job"], a[href*="career"]',
                    'job_titles': 'h1, h2, h3, .title',
                    'job_details': '.content, .description'
                }
            }
        }

    # ================== DOCUMENT PROCESSING ==================
    def extract_text_from_docx(self, cv_path):
        if not _Document:
            raise RuntimeError("python-docx is not installed")
        doc = _Document(cv_path)
        text = []
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text.strip())
        return "\n".join(text)

    def extract_text_from_pdf(self, cv_path):
        if not _PyPDF2:
            raise RuntimeError("PyPDF2 is not installed")
        text = []
        with open(cv_path, "rb") as f:
            reader = _PyPDF2.PdfReader(f)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text.append(content.strip())
        return "\n".join(text)

    def extract_cv_text(self, cv_path):
        ext = os.path.splitext(cv_path)[1].lower()
        if ext == ".docx":
            return self.extract_text_from_docx(cv_path)
        elif ext == ".pdf":
            return self.extract_text_from_pdf(cv_path)
        else:
            raise ValueError("Unsupported file format! Only PDF or DOCX allowed.")

    # ================== AGENT 1: CV TAILORING ==================
    def tailor_cv_with_gemini(self, cv_text, job_description):
        prompt = f"""
You are an expert ATS-optimized resume writer with 15+ years of experience. Your task is to strategically tailor the existing CV to maximize match with the job description while maintaining authenticity.

CRITICAL INSTRUCTIONS:
1. PRESERVE the person's actual experience and qualifications - DO NOT fabricate or exaggerate
2. OPTIMIZE language to match job description keywords and requirements
3. REORDER and EMPHASIZE experiences that best match the target role
4. QUANTIFY achievements where possible using metrics from the original CV
5. Use ATS-friendly formatting and industry-standard terminology

TARGET JOB DESCRIPTION:
{job_description}

ORIGINAL CV CONTENT:
{cv_text}

TAILORING STRATEGY:
- Extract and highlight the most relevant experiences first
- Mirror the job description's language and keywords naturally
- Emphasize transferable skills that match the role requirements
- Restructure bullet points to lead with impact and results
- Include industry-specific terminology from the job posting
- Ensure skills section directly addresses job requirements
- Optimize summary to position candidate as ideal fit

Return ONLY valid JSON in this exact format (no explanations, code blocks, or markdown):
{{
  "NAME": "Extract exact name from original CV",
  "CONTACT": "Extract and format: email • phone • LinkedIn • location",
  "SUMMARY": "Write 3-4 line compelling summary that positions candidate perfectly for this specific role, incorporating key terms from job description while reflecting their actual background and achievements",
  "EXPERIENCE": [
    "Most relevant experience first: Job Title at Company Name (Start-End dates): Action-oriented description highlighting quantified achievements and skills that directly match job requirements, using keywords from job posting",
    "Second most relevant experience: Focus on transferable skills and accomplishments that support candidacy for target role",
    "Additional relevant experiences in order of relevance to target position"
  ],
  "SKILLS": [
    "List skills in order of relevance to job posting",
    "Include exact skill names mentioned in job description",
    "Add relevant technical skills from original CV",
    "Include soft skills specifically mentioned in job requirements",
    "Ensure mix of hard and soft skills totaling 8-12 items"
  ],
  "EDUCATION": "Format education, certifications, and relevant training in order of relevance to target role, including graduation years and any honors/achievements that support candidacy"
}}
"""
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            _logger.error(f"Error querying Gemini model: {e}")
            raise e
            
        try:
            structured = _json.loads(response_text)
        except Exception:
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                structured = _json.loads(response_text[start:end])
            except Exception as e:
                raise ValueError(f"Gemini did not return valid JSON.\nResponse:\n{response_text}") from e

        required_keys = ["NAME", "CONTACT", "SUMMARY", "EXPERIENCE", "SKILLS", "EDUCATION"]
        for key in required_keys:
            if key not in structured:
                structured[key] = "" if key not in ["EXPERIENCE", "SKILLS"] else []
        return structured

    def render_template(self, data, template_path):
        if not template_path or not os.path.exists(template_path) or not _J2Env:
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Tailored CV</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .section { margin-bottom: 20px; }
                    .title { font-size: 24px; font-weight: bold; color: #333; }
                    .subtitle { font-size: 18px; font-weight: bold; color: #666; }
                    .content { margin-left: 20px; }
                </style>
            </head>
            <body>
                <div class="section">
                    <div class="title">{name}</div>
                    <div class="content">{contact}</div>
                </div>
                <div class="section">
                    <div class="subtitle">Summary</div>
                    <div class="content">{summary}</div>
                </div>
                <div class="section">
                    <div class="subtitle">Experience</div>
                    <div class="content">
                        {experience}
                    </div>
                </div>
                <div class="section">
                    <div class="subtitle">Skills</div>
                    <div class="content">
                        {skills}
                    </div>
                </div>
                <div class="section">
                    <div class="subtitle">Education</div>
                    <div class="content">{education}</div>
                </div>
            </body>
            </html>
            """
            return html_template.format(
                name=data.get('NAME', 'Name'),
                contact=data.get('CONTACT', 'Contact Information'),
                summary=data.get('SUMMARY', 'Professional Summary'),
                experience='<br>'.join(data.get('EXPERIENCE', [])),
                skills='<br>'.join(data.get('SKILLS', [])),
                education=data.get('EDUCATION', 'Education')
            )
        template_dir, template_file = os.path.split(template_path)
        env = _J2Env(loader=_J2Loader(template_dir or "./"))
        template = env.get_template(template_file)
        return template.render(data=data)

    def html_to_pdf(self, html_content, output_path):
        try:
            try:
                from weasyprint import HTML as _WHTML
                _WHTML(string=html_content).write_pdf(output_path)
                _logger.info(f"CV saved at: {output_path}")
                return output_path
            except Exception:
                pass
            try:
                import pdfkit as _pdfkit
                _pdfkit.from_string(html_content, output_path)
                _logger.info(f"CV saved at: {output_path}")
                return output_path
            except Exception:
                pass
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            _logger.info(f"PDF generation not available. HTML saved at: {html_path}")
            return html_path
        except Exception as e:
            _logger.error(f"PDF generation failed: {e}")
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            _logger.info(f"Fallback: HTML saved at: {html_path}")
            return html_path

    def agent1_cv_tailor(self, cv_path, job_description, template_path, output_dir="./"):
        cv_text = self.extract_cv_text(cv_path)
        structured_cv = self.tailor_cv_with_gemini(cv_text, job_description)
        if template_path:
            html_content = self.render_template(structured_cv, template_path)
            base_name = os.path.splitext(os.path.basename(cv_path))[0]
            output_pdf = os.path.join(output_dir, f"{base_name}_tailored.pdf")
            output_file = self.html_to_pdf(html_content, output_pdf)
            return {
                'output_file': output_file,
                'structured_cv': structured_cv
            }
        else:
            return {
                'output_file': None,
                'structured_cv': structured_cv
            }

    # ================== AGENT 2: COVER LETTER ==================
    def generate_cover_letter(self, cv_text, job_description):
        prompt = f"""
You are an expert career consultant.
Based on the following tailored CV and job description,
write a **professional, formal, and compelling cover letter**.

Instructions:
- Use a polite greeting (e.g., "Dear Hiring Manager").
- If the job description includes a company name, address the company directly (e.g., "I am excited to apply to [Company Name]").
- If no company name is mentioned, instead express interest in the position (e.g., "I am excited to apply for this Data Scientist position").
- Highlight skills and experiences from the CV that are most relevant to the job description.
- Keep it concise: 3–4 paragraphs, maximum 1 page.
- End with a polite closing (e.g., "Sincerely, [Your Name], [Email Address], [Contact Number]).

CV:
{cv_text}

Job Description:
{job_description}

Return only the final cover letter text.
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            _logger.error(f"Error generating cover letter: {e}")
            raise e

    def agent2_cover_letter(self, cv_path, job_description):
        cv_text = self.extract_cv_text(cv_path)
        cover_letter = self.generate_cover_letter(cv_text, job_description)
        return cover_letter

    # ================== AGENT 3: JOB SCRAPING ==================
    def clean_text(self, text):
        if not text:
            return ""
        text = _re.sub(r'\s+', ' ', text.strip())
        return text

    def is_relevant(self, text, query):
        """Enhanced relevance checking with better matching"""
        if not text or not query:
            return False
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Direct keyword matching
        query_words = query_lower.split()
        text_words = text_lower.split()
        
        # Check for exact matches first
        if query_lower in text_lower:
            return True
        
        # Check for individual word matches
        matched_words = sum(1 for word in query_words if word in text_lower)
        if matched_words >= len(query_words) * 0.6:  # At least 60% of query words must match
            return True
        
        # Use fuzzy matching as fallback if available
        if _fuzz:
            score = _fuzz.partial_ratio(query_lower, text_lower)
            return score >= 65  # Slightly higher threshold for better relevance
        
        return False

    def random_delay(self, min_sec=1, max_sec=3):
        _time.sleep(_random.uniform(min_sec, max_sec))

    def scrape_newspaper_jobs(self, newspaper_name, newspaper_info, query):
        jobs = []
        try:
            if not self.session or not _BeautifulSoup:
                return jobs
            _logger.info(f"Scraping jobs from {newspaper_name}...")
            response = self.session.get(newspaper_info['jobs_url'], timeout=15)
            if response.status_code == 200:
                soup = _BeautifulSoup(response.content, 'html.parser')
                jobs.extend(self.extract_jobs_from_page(soup, newspaper_info, newspaper_name, query))
            return jobs
        except Exception as e:
            _logger.error(f"Error scraping {newspaper_name}: {str(e)}")
            return []

    def extract_jobs_from_page(self, soup, newspaper_info, newspaper_name, query):
        jobs = []
        job_links = soup.select(newspaper_info['selectors']['job_links'])
        for link in job_links[:20]:
            href = link.get('href', '')
            title = link.get_text(strip=True)
            if href and title and self.is_relevant(title, query):
                if not href.startswith('http'):
                    href = _urljoin(newspaper_info['base_url'], href)
                job = {
                    'title': self.clean_text(title),
                    'link': href,
                    'newspaper': newspaper_name,
                    'source': newspaper_name,
                    'company': 'Not specified',
                    'location': 'Pakistan',
                    'scraped_at': _dt.now().isoformat()
                }
                jobs.append(job)
        return jobs

    def scrape_linkedin(self, query, location="Pakistan"):
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            loc_encoded = _urllib_parse.quote(location)
            url = f"https://www.linkedin.com/jobs/search/?keywords={query_encoded}&location={loc_encoded}&f_TPR=r2592000"
            self.driver.get(url)
            self.random_delay(2, 4)
            job_cards = _WebDriverWait(self.driver, 10).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "div.base-card"))
            )
            for card in job_cards[:20]:
                try:
                    title = card.find_element(_By.CSS_SELECTOR, "h3").text.strip()
                    company = card.find_element(_By.CSS_SELECTOR, "h4").text.strip()
                    link = card.find_element(_By.CSS_SELECTOR, "a").get_attribute("href")
                    try:
                        location_element = card.find_element(_By.CSS_SELECTOR, "span.job-search-card__location")
                        job_location = location_element.text.strip()
                    except Exception:
                        job_location = location
                    if self.is_relevant(title, query):
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": job_location,
                            "link": link,
                            "source": "LinkedIn",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"LinkedIn scraping failed: {e}")
            return []

    def scrape_bayt(self, query, location="Pakistan"):
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            url = f"https://www.bayt.com/en/pakistan/jobs/{query_encoded}-jobs/?postedWithin=30"
            self.driver.get(url)
            self.random_delay(2, 4)
            job_cards = _WebDriverWait(self.driver, 10).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "li.has-pointer-d"))
            )
            for card in job_cards[:20]:
                try:
                    title_element = card.find_element(_By.CSS_SELECTOR, "h2 a")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    if not link.startswith('http'):
                        link = "https://www.bayt.com" + link
                    try:
                        company_element = card.find_element(_By.CSS_SELECTOR, "b.m0")
                        company = company_element.text.strip()
                    except Exception:
                        company = "Not specified"
                    try:
                        location_elements = card.find_elements(_By.CSS_SELECTOR, "span.color-mid")
                        job_location = location_elements[0].text.strip() if location_elements else location
                    except Exception:
                        job_location = location
                    if self.is_relevant(title, query):
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": job_location,
                            "link": link,
                            "source": "Bayt.com",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"Bayt.com scraping failed: {e}")
            return []

    def scrape_mustakbil(self, query, location="Pakistan"):
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            url = f"https://mustakbil.com/jobs/search?q={query_encoded}&location=pakistan&posted_within=30"
            self.driver.get(url)
            self.random_delay(2, 4)
            job_cards = None
            selectors_to_try = [
                "div.job-item",
                "div.job-card",
                "div.listing",
                "article.job",
                "div.search-result"
            ]
            for selector in selectors_to_try:
                try:
                    job_cards = _WebDriverWait(self.driver, 8).until(
                        _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, selector))
                    )
                    if job_cards:
                        break
                except Exception:
                    continue
            if not job_cards:
                job_cards = self.driver.find_elements(_By.CSS_SELECTOR, "div")
                job_cards = [card for card in job_cards if card.text and len(card.text.strip()) > 20]
            for card in job_cards[:20]:
                try:
                    title = ""
                    company = ""
                    link = ""
                    try:
                        title_element = card.find_element(_By.CSS_SELECTOR, "h3 a")
                        title = title_element.text.strip()
                        link = title_element.get_attribute("href")
                    except Exception:
                        try:
                            title_element = card.find_element(_By.CSS_SELECTOR, "a[href*='job']")
                            title = title_element.text.strip()
                            link = title_element.get_attribute("href")
                        except Exception:
                            links = card.find_elements(_By.TAG_NAME, "a")
                            for link_elem in links:
                                href = link_elem.get_attribute("href") or ""
                                text = link_elem.text.strip()
                                if ("job" in href.lower() or "career" in href.lower()) and len(text) > 10:
                                    title = text
                                    link = href
                                    break
                    if not title or not link:
                        continue
                    try:
                        company_selectors = ["span.company", "div.company", "p.company", ".employer"]
                        for selector in company_selectors:
                            try:
                                company_element = card.find_element(_By.CSS_SELECTOR, selector)
                                company = company_element.text.strip()
                                break
                            except Exception:
                                continue
                        if not company:
                            company = "Not specified"
                    except Exception:
                        company = "Not specified"
                    if link and not link.startswith('http'):
                        link = "https://mustakbil.com" + link
                    if title and link and self.is_relevant(title, query):
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": location,
                            "link": link,
                            "source": "Mustakbil.com",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"Mustakbil.com scraping failed: {e}")
            return []

    def scrape_rozee(self, query, location="Pakistan"):
        """Scrape jobs from Rozee.pk - one of Pakistan's largest job portals"""
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            url = f"https://www.rozee.pk/job/jsearch/q/{query_encoded}/fpn/1"
            self.driver.get(url)
            self.random_delay(2, 4)
            
            job_cards = _WebDriverWait(self.driver, 10).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "div.job-item, article.job-post, div.job-listing"))
            )
            
            for card in job_cards[:20]:
                try:
                    title_element = card.find_element(_By.CSS_SELECTOR, "h3 a, h2 a, .job-title a")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    
                    try:
                        company_element = card.find_element(_By.CSS_SELECTOR, ".company-name, .employer, .company")
                        company = company_element.text.strip()
                    except Exception:
                        company = "Not specified"
                    
                    try:
                        location_element = card.find_element(_By.CSS_SELECTOR, ".location, .job-location")
                        job_location = location_element.text.strip()
                    except Exception:
                        job_location = location
                    
                    if self.is_relevant(title, query):
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": job_location,
                            "link": link,
                            "source": "Rozee.pk",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"Rozee.pk scraping failed: {e}")
            return []

    def scrape_jobs_pk(self, query, location="Pakistan"):
        """Scrape jobs from Jobs.pk"""
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            url = f"https://www.jobs.pk/jobs?q={query_encoded}&l=pakistan"
            self.driver.get(url)
            self.random_delay(2, 4)
            
            job_cards = _WebDriverWait(self.driver, 10).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "div.job-card, div.job-item, .job-listing"))
            )
            
            for card in job_cards[:20]:
                try:
                    title_element = card.find_element(_By.CSS_SELECTOR, "h3 a, h2 a, .job-title a")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    
                    try:
                        company_element = card.find_element(_By.CSS_SELECTOR, ".company, .employer-name")
                        company = company_element.text.strip()
                    except Exception:
                        company = "Not specified"
                    
                    if self.is_relevant(title, query):
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": location,
                            "link": link if link.startswith('http') else f"https://www.jobs.pk{link}",
                            "source": "Jobs.pk",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"Jobs.pk scraping failed: {e}")
            return []

    def scrape_olx_jobs(self, query, location="Pakistan"):
        """Scrape jobs from OLX Pakistan"""
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            url = f"https://www.olx.com.pk/jobs_q_{query_encoded}"
            self.driver.get(url)
            self.random_delay(2, 4)
            
            job_cards = _WebDriverWait(self.driver, 10).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "div[data-aut-id='itemBox'], .ads__item"))
            )
            
            for card in job_cards[:15]:
                try:
                    title_element = card.find_element(_By.CSS_SELECTOR, "h3 a, h2 a, [data-aut-id='itemTitle'] a")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    
                    try:
                        location_element = card.find_element(_By.CSS_SELECTOR, "[data-aut-id='item-location'], .ads__item__location")
                        job_location = location_element.text.strip()
                    except Exception:
                        job_location = location
                    
                    if self.is_relevant(title, query) and "job" in title.lower():
                        jobs.append({
                            "title": title,
                            "company": "Various Employers",
                            "location": job_location,
                            "link": link if link.startswith('http') else f"https://www.olx.com.pk{link}",
                            "source": "OLX Pakistan",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"OLX Pakistan scraping failed: {e}")
            return []

    def scrape_indeed_pk(self, query, location="Pakistan"):
        """Scrape jobs from Indeed Pakistan"""
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            loc_encoded = _urllib_parse.quote(location)
            url = f"https://pk.indeed.com/jobs?q={query_encoded}&l={loc_encoded}"
            self.driver.get(url)
            self.random_delay(2, 4)
            
            job_cards = _WebDriverWait(self.driver, 10).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "div.jobsearch-SerpJobCard, div.job_seen_beacon, .slider_container .slider_item"))
            )
            
            for card in job_cards[:20]:
                try:
                    title_element = card.find_element(_By.CSS_SELECTOR, "h2 a, .jobTitle a, [data-jk] h2 a")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    
                    try:
                        company_element = card.find_element(_By.CSS_SELECTOR, ".companyName, [data-testid='company-name'], .company")
                        company = company_element.text.strip()
                    except Exception:
                        company = "Not specified"
                    
                    try:
                        location_element = card.find_element(_By.CSS_SELECTOR, ".companyLocation, [data-testid='job-location'], .location")
                        job_location = location_element.text.strip()
                    except Exception:
                        job_location = location
                    
                    if self.is_relevant(title, query):
                        full_link = link if link.startswith('http') else f"https://pk.indeed.com{link}"
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": job_location,
                            "link": full_link,
                            "source": "Indeed Pakistan",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"Indeed Pakistan scraping failed: {e}")
            return []

    def scrape_pk_jobs_com(self, query, location="Pakistan"):
        """Scrape jobs from PkJobs.com"""
        jobs = []
        try:
            if not _webdriver:
                return jobs
            query_encoded = _urllib_parse.quote(query)
            url = f"https://www.pkjobs.com.pk/jobs?q={query_encoded}"
            self.driver.get(url)
            self.random_delay(2, 4)
            
            job_cards = _WebDriverWait(self.driver, 8).until(
                _EC.presence_of_all_elements_located((_By.CSS_SELECTOR, "div.job-item, .job-card, article.job"))
            )
            
            for card in job_cards[:15]:
                try:
                    title_element = card.find_element(_By.CSS_SELECTOR, "h3 a, h2 a, .job-title a")
                    title = title_element.text.strip()
                    link = title_element.get_attribute("href")
                    
                    try:
                        company_element = card.find_element(_By.CSS_SELECTOR, ".company, .employer")
                        company = company_element.text.strip()
                    except Exception:
                        company = "Not specified"
                    
                    if self.is_relevant(title, query):
                        jobs.append({
                            "title": title,
                            "company": company,
                            "location": location,
                            "link": link if link.startswith('http') else f"https://www.pkjobs.com.pk{link}",
                            "source": "PkJobs.com",
                            "scraped_at": _dt.now().isoformat()
                        })
                except Exception:
                    continue
            return jobs
        except Exception as e:
            _logger.error(f"PkJobs.com scraping failed: {e}")
            return []

    def agent3_job_scraper(self, user_query, location="Pakistan"):
        all_jobs = []
        
        # Scrape newspaper job sections
        for newspaper_name, newspaper_info in self.newspapers.items():
            jobs = self.scrape_newspaper_jobs(newspaper_name, newspaper_info, user_query)
            all_jobs.extend(jobs)
            _time.sleep(2)
        
        # Major job portals - prioritized by reliability
        platforms = [
            ("Bayt.com", lambda: self.scrape_bayt(user_query, location)),
            ("Rozee.pk", lambda: self.scrape_rozee(user_query, location)),
            ("Indeed Pakistan", lambda: self.scrape_indeed_pk(user_query, location)),
            ("LinkedIn", lambda: self.scrape_linkedin(user_query, location)),
            ("Mustakbil.com", lambda: self.scrape_mustakbil(user_query, location)),
            ("Jobs.pk", lambda: self.scrape_jobs_pk(user_query, location)),
            ("PkJobs.com", lambda: self.scrape_pk_jobs_com(user_query, location)),
            ("OLX Pakistan", lambda: self.scrape_olx_jobs(user_query, location))
        ]
        
        for site_name, scrape_func in platforms:
            try:
                _logger.info(f"Scraping {site_name}...")
                jobs = scrape_func()
                all_jobs.extend(jobs)
                _logger.info(f"Found {len(jobs)} jobs from {site_name}")
                self.random_delay(3, 5)  # Respectful delay between sites
            except Exception as e:
                _logger.error(f"{site_name} completely failed: {e}")
                continue
        
        return all_jobs

    # ================== AGENT 4: STRENGTHS/WEAKNESSES ==================
    def analyze_strengths_weaknesses(self, cv_text, job_description):
        prompt = f"""
Analyze this CV for the job position described in this job description.
Return **only very short bullet points**.

1. Strengths
2. Weaknesses  
3. Market comparison (skills/experience gaps)

Format:
Strengths:
- ...
Weaknesses:
- ...
Market:
- ...

CV:
{cv_text}

Job Description:
{job_description}
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            _logger.error(f"Error analyzing strengths/weaknesses: {e}")
            raise e

    def agent4_strengths_weaknesses(self, cv_path, job_description):
        cv_text = self.extract_cv_text(cv_path)
        analysis = self.analyze_strengths_weaknesses(cv_text, job_description)
        return analysis

    # ================== AGENT 5: COURSE SUGGESTIONS ==================
    def get_course_names_from_gemini(self, cv_text, job_description):
        prompt = f"""
You are an expert career coach.
Based on the following CV and job description, suggest 5-7 relevant courses
that help improve skills and close gaps.

Instructions:
- Return only course names, one per line.
- Keep it short and clear.

CV:
{cv_text}

Job Description:
{job_description}
"""
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            course_names = [line.strip() for line in response_text.splitlines() if line.strip()]
            return course_names
        except Exception as e:
            _logger.error(f"Error getting course suggestions: {e}")
            raise e

    def get_coursera_link(self, course_name):
        query = _quote(course_name)
        search_url = f"https://www.coursera.org/search?query={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            resp = _requests.get(search_url, headers=headers, timeout=10) if _requests else None
            if not resp:
                return "Link not found"
            soup = _BeautifulSoup(resp.text, "html.parser")
            a_tag = soup.select_one("a[data-click-key='search.search.click.search_card']")
            if a_tag and a_tag.get("href"):
                href = a_tag["href"]
                if href.startswith("/learn/") or href.startswith("/specializations/"):
                    return f"https://www.coursera.org{href}"
        except Exception:
            pass
        return "Link not found"

    def agent5_course_suggestions_with_links(self, cv_path, job_description):
        cv_text = self.extract_cv_text(cv_path)
        course_names = self.get_course_names_from_gemini(cv_text, job_description)
        course_list = []
        for course in course_names:
            link = self.get_coursera_link(course)
            course_list.append(f"- {course} : {link}")
        return "\n".join(course_list)

    # ================== AGENT 6: WEEKLY LEARNING PLAN ==================
    def get_weekly_learning_plan(self, cv_text, job_description):
        prompt = f"""
You are an expert career and learning coach.
Based on this CV and the job description, do the following:

1. Identify **one key skill** to focus on for the job.
2. Create a **1-week learning plan**, broken down by day, including:
   - Daily learning hours
   - Exercises or practical tasks

Instructions:
- Keep it short and actionable.
- Use bullet points for clarity.
- Each day should include 1-2 exercises or practical tasks.
- Example format:

Skill to focus: <Skill Name>
Weekly Plan:
- Day 1 (3 hrs): Task 1, Task 2
- Day 2 (3 hrs): Task 1, Task 2
...

CV:
{cv_text}

Job Description:
{job_description}
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            _logger.error(f"Error generating weekly learning plan: {e}")
            raise e

    def agent6_weekly_learning_scheduler(self, cv_path, job_description):
        cv_text = self.extract_cv_text(cv_path)
        weekly_plan = self.get_weekly_learning_plan(cv_text, job_description)
        return weekly_plan

    # ================== AGENT 7: CAREER ROADMAP ==================
    def get_career_roadmap_from_description(self, job_description, years=3):
        prompt = f"""
You are an expert career mentor.
Based on the following job description, create a **long-term career roadmap**.

Instructions:
- Suggest a **1-{years} year plan** for career growth.
- Guess the next steps in the career automatically (e.g., Junior → Mid → Senior).
- For each step, list:
    - Key skills to acquire
    - Suggested courses/projects (short list)
    - Milestones/timeline
- Keep it bullet-pointed, clear, and actionable.
- Output format:

Step 1: <Role/Title> (<Timeline>)
- Skills: ...
- Courses/Projects: ...
- Milestones: ...

Return only the roadmap in bullet points.

Job Description:
{job_description}
"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            _logger.error(f"Error generating career roadmap: {e}")
            raise e

    def agent7_career_path_planner(self, job_description, years=3):
        roadmap = self.get_career_roadmap_from_description(job_description, years)
        return roadmap

    # ================== UTILITY METHODS ==================
    def save_jobs_to_file(self, jobs, user_query, location, filename=None):
        if not filename:
            safe_query = "".join(c for c in user_query if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_query = safe_query.replace(' ', '_')
            filename = f"jobs_{safe_query}_{len(jobs)}.json"
        results = {
            "search_query": user_query,
            "location": location,
            "total_jobs": len(jobs),
            "scraped_at": _dt.now().isoformat(),
            "sites_scraped": list(self.newspapers.keys()) + ["Bayt.com", "Rozee.pk", "Indeed Pakistan", "LinkedIn", "Mustakbil.com", "Jobs.pk", "PkJobs.com", "OLX Pakistan"],
            "date_filter": "Past month only for platforms",
            "jobs": jobs
        }
        with open(filename, "w", encoding="utf-8") as f:
            _json.dump(results, f, ensure_ascii=False, indent=2)
        _logger.info(f"Results saved to: {filename}")
        return filename

    def display_job_summary(self, jobs, user_query, location):
        print("\n" + "=" * 60)
        print("SCRAPING SUMMARY")
        print("=" * 60)
        sources = {}
        for job in jobs:
            source = job['source']
            sources[source] = sources.get(source, 0) + 1
        total_jobs = len(jobs)
        print(f"Total Jobs Found: {total_jobs}")
        print("\nBreakdown by source:")
        for source, count in sources.items():
            print(f"  • {source}: {count} jobs")
        print(f"\nSearch Query: '{user_query}'")
        print(f"Location: {location}")
        print("Filter: Past month only for platforms")
        if total_jobs > 0:
            print(f"\nSample jobs found:")
            for i, job in enumerate(jobs[:8], 1):
                print(f"  {i:2d}. {job['title'][:50]:<50} | {job['source']:<15}")
                print(f"      Company: {job['company'][:40]}")
                print()
        else:
            print("\nNo jobs found. Try different keywords.")

    def close(self):
        if getattr(self, 'driver', None):
            try:
                self.driver.quit()
            except Exception:
                pass

    def extract_job_title_from_description(self, job_description):
        """Extract job title from job description using Gemini"""
        prompt = f"""
Extract the exact job title from this job description.
Return only the job title, nothing else.

Job Description:
{job_description}

Examples:
- If description mentions "Software Engineer position", return "Software Engineer"
- If description mentions "Marketing Manager role", return "Marketing Manager"
- If description mentions "Data Scientist opening", return "Data Scientist"

Return only the job title:
"""
        try:
            response = self.model.generate_content(prompt)
            job_title = response.text.strip()
            return job_title if job_title else "Software Engineer"
        except Exception as e:
            _logger.error(f"Error extracting job title: {e}")
            raise e
