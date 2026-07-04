from flask import Flask, render_template, request, session, jsonify, redirect, url_for, send_file
from flask_cors import CORS
from flask_session import Session
from flask_login import LoginManager, login_required, current_user
from database import db
from models import User, InterviewSession
from auth_routes import auth_bp
from agents import RoleValidatorAgent, QuestionGeneratorAgent, AnswerEvaluatorAgent, analyze_audio_for_hesitation, WHISPER_MODEL, CSVFeedbackSystem
import os
import whisper
import tempfile
import re
from agents import MasterAgent, VideoAnalyzerAgent  # Add this import
from agents import CareerAutomationSystem

app = Flask(__name__)
CORS(app)
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
app.config["SECRET_KEY"] = "careerly_secret_key_change_in_production"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'mysql+pymysql://root:123abc@localhost/careerly'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
Session(app)
db.init_app(app)

# Initialize database
with app.app_context():
    db.create_all()







# Additional API configuration for career automation
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize agents
master = MasterAgent()
# Use the master agent directly - no need to create individual agent instances

# Initialize CareerAutomationSystem lazily, reusing MasterAgent model
career_system = None

def get_career_system():
    """Get or create career automation system instance"""
    global career_system
    if career_system is None:
        # Use the same API key as MasterAgent
        career_system = CareerAutomationSystem(gemini_api_key="GEMINI_API_KEY")
    return career_system

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'doc', 'docx'}

# Initialize feedback system
feedback_system = CSVFeedbackSystem()

# --- Whisper Model Initialization moved to agents.py ---


# ---------- HELPER: Audio Analysis ----------
 


# ---------- ROUTES ----------
@app.route("/")
def home():
    # If user is not logged in, redirect to login
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return render_template("home.html")

@app.route("/interview")
@login_required
def interview():
    return render_template("interview.html")

@app.route("/analysis")
@login_required
def analysis():
    return render_template("analysis.html")

# Redirect old login/signup routes to new auth routes
@app.route("/login")
def login_redirect():
    return redirect(url_for('auth.login'))

@app.route("/signup")
def signup_redirect():
    return redirect(url_for('auth.signup'))


@app.route("/start_interview", methods=["POST"])
@login_required
def start_interview():
    data = request.get_json()
    role = data.get("role", "").strip()
    interview_type = data.get("interview_type", "").strip()
    level = data.get("level", "").strip()
    interview_mode = data.get("interview_mode", "text").strip()

    if not role or not master.validate_role(role):
        return jsonify({"error": "Invalid role entered. Please provide a valid job role."}), 400

    if not interview_type or not level:
        return jsonify({"error": "Please select interview type and difficulty level."}), 400

    questions = master.generate_questions(role, interview_type, level)

    # Create interview session in database
    try:
        interview_session = InterviewSession.create_session(
            user_id=current_user.id,
            role=role,
            interview_type=interview_type,
            level=level,
            interview_mode=interview_mode
        )
        interview_session.update_questions(questions)
        
        # Store session data in Flask session for compatibility
        session["db_session_id"] = interview_session.session_id
    except Exception as e:
        print(f"Error creating interview session: {e}")
        # Continue with old session-based approach if DB fails
        pass

    session["questions"] = questions
    session["answers"] = []
    session["role"] = role
    session["interview_type"] = interview_type
    session["level"] = level
    session["interview_mode"] = interview_mode
    session["current_question"] = 0

    return jsonify({"success": True, "questions": questions, "total_questions": len(questions)})


@app.route("/submit_answer", methods=["POST"])
@login_required
def submit_answer():
    data = request.get_json()
    q_index = data.get("q_index", 0)
    answer = data.get("answer", "").strip()
    interview_mode = session.get("interview_mode", "text")
    audio_metrics = None

    if "questions" not in session or q_index >= len(session["questions"]):
        return jsonify({"error": "Invalid question index"}), 400

    question = session["questions"][q_index]
    role = session.get("role", "Unknown Role")

    if interview_mode == "audio":
        audio_metrics = session.pop("current_audio_metrics", None)
    elif interview_mode == "video":
        # Get video data from session
        video_data = session.get("current_video_data", {})
        audio_metrics = video_data.get("audio_metrics")
        video_analysis = video_data.get("video_analysis", {})
        # Clear video data after use
        session.pop("current_video_data", None)

    feedback = master.evaluate_answer(question, answer, role, audio_metrics)

    answer_data = {
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "question_number": q_index + 1,
        "mode": interview_mode,
        "audio_metrics": audio_metrics
    }
    
    session["answers"].append(answer_data)
    session["current_question"] = q_index + 1
    
    # Update database session if exists
    try:
        db_session_id = session.get("db_session_id")
        if db_session_id:
            interview_session = InterviewSession.get_by_session_id(db_session_id)
            if interview_session:
                interview_session.add_answer(answer_data)
    except Exception as e:
        print(f"Error updating interview session: {e}")

    return jsonify({
        "success": True,
        "feedback": feedback,
        "is_last_question": q_index >= len(session["questions"]) - 1
    })


@app.route("/upload_audio_answer", methods=["POST"])
@login_required
def upload_audio_answer():
    print("=== AUDIO UPLOAD DEBUG START ===")
    
    if "audio_file" not in request.files:
        print("ERROR: No audio file in request")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_file"]
    print(f"Received file: {audio_file.filename}")
    print(f"Content type: {audio_file.content_type}")
    print(f"File size: {len(audio_file.read())} bytes")
    audio_file.seek(0)  # Reset file pointer after reading size
    
    if audio_file.filename == "":
        print("ERROR: Empty filename")
        return jsonify({"error": "No selected file"}), 400

    if not WHISPER_MODEL:
        print("ERROR: Whisper model not loaded")
        return jsonify({"error": "Audio transcription service not available."}), 503

    try:
        # Determine file extension
        content_type = audio_file.content_type or ''
        print(f"Content type: {content_type}")
        
        file_extension = None
        if 'webm' in content_type:
            file_extension = '.webm'
        elif 'wav' in content_type:
            file_extension = '.wav'
        elif 'mp3' in content_type:
            file_extension = '.mp3'
        elif 'mp4' in content_type or 'm4a' in content_type:
            file_extension = '.m4a'
        else:
            file_extension = os.path.splitext(audio_file.filename)[1] or '.webm'
        
        print(f"Using file extension: {file_extension}")
        
        # Create static/audio directory if it doesn't exist
        audio_dir = os.path.join("static", "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        import time
        timestamp = str(int(time.time()))
        saved_filename = f"interview_answer_{timestamp}{file_extension}"
        saved_path = os.path.join(audio_dir, saved_filename)
        
        # Save the audio file permanently
        audio_file.save(saved_path)
        
        print(f"Audio saved to: {saved_path}")
        print(f"Saved file size: {os.path.getsize(saved_path)} bytes")

        # Try transcription
        try:
            print("Starting Whisper transcription...")
            analysis_results = analyze_audio_for_hesitation(saved_path, WHISPER_MODEL)
            print(f"Full transcription: '{analysis_results['full_transcription']}'")
            
        except Exception as whisper_error:
            print(f"Whisper detailed error: {whisper_error}")
            # Try basic transcription
            try:
                print("Trying basic Whisper transcription...")
                result = WHISPER_MODEL.transcribe(saved_path, fp16=False, language="en")
                print(f"Basic transcription: '{result['text']}'")
                
                analysis_results = {
                    "full_transcription": result["text"],
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
            except Exception as final_error:
                print(f"Final Whisper error: {final_error}")
                raise Exception(f"Unable to transcribe audio: {str(final_error)}")

        # Don't delete the file - keep it for inspection
        # os.remove(saved_path)  # REMOVED THIS LINE
        
        if not analysis_results.get("full_transcription"):
            print("WARNING: Empty transcription returned")
            return jsonify({"error": "No speech detected in audio file"}), 400

        # Store both transcription and file path for debugging
        session["current_audio_transcription"] = analysis_results["full_transcription"]
        session["current_audio_file_path"] = saved_path  # Store path for debugging
        session["current_audio_metrics"] = {
            k: v for k, v in analysis_results.items()
            if k not in ["full_transcription", "word_details"]
        }

        print(f"=== AUDIO UPLOAD SUCCESS: File saved at {saved_path} ===")
        return jsonify({
            "success": True,
            "transcription": analysis_results["full_transcription"],
            "message": "Audio processed successfully. Submit to get feedback.",
            "debug_file_path": saved_path  # Include path in response for debugging
        })

    except Exception as e:
        print(f"=== AUDIO UPLOAD ERROR: {e} ===")
        return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    
@app.route("/submit_video_interview", methods=["POST"])
@login_required
def submit_video_interview():
    if "video_file" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video_file = request.files["video_file"]
    if video_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save video
        video_dir = os.path.join("static", "video")
        os.makedirs(video_dir, exist_ok=True)
        import time
        timestamp = str(int(time.time()))

        # Accept any video format and save with original extension
        original_ext = os.path.splitext(video_file.filename)[1]
        if not original_ext:
            original_ext = '.mp4'  # fallback
        saved_filename = f"interview_video_{timestamp}{original_ext}"
        saved_path = os.path.join(video_dir, saved_filename)
        video_file.save(saved_path)

        print(f"Video saved to: {saved_path}")

        # Run real video analysis using the integrated VideoAnalyzerAgent
        try:
            print("Starting comprehensive video analysis...")
            video_analysis = master.analyze_video(saved_path)
            print("Video analysis completed successfully")
            
            # Extract audio from video for additional analysis
            audio_path = None
            try:
                from moviepy.editor import VideoFileClip
                video_clip = VideoFileClip(saved_path)
                audio_path = saved_path.replace(original_ext, ".wav")
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                video_clip.close()
                print(f"Audio extracted to: {audio_path}")
            except Exception as e:
                print(f"Audio extraction failed: {e}")
                # Continue without audio if extraction fails

            # Run audio analysis if audio was extracted
            audio_metrics = None
            if audio_path and os.path.exists(audio_path):
                try:
                    audio_metrics = analyze_audio_for_hesitation(audio_path, WHISPER_MODEL)
                    print("Audio analysis completed")
                except Exception as e:
                    print(f"Audio analysis failed: {e}")
                    audio_metrics = None
            else:
                print("No audio available for analysis")
                audio_metrics = {
                    "full_transcription": "Audio analysis not available",
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

            # Store comprehensive video analysis data in session
            session["current_video_data"] = {
                "video_path": saved_path,
                "video_analysis": video_analysis,
                "audio_metrics": audio_metrics,
                "timestamp": timestamp
            }

            # Clean up temporary audio file
            if audio_path and os.path.exists(audio_path) and audio_path != saved_path:
                try:
                    os.remove(audio_path)
                except:
                    pass  # Don't fail if cleanup fails

            return jsonify({
                "success": True,
                "message": "Video analysis completed successfully. Submit to get comprehensive feedback.",
                "transcription": audio_metrics.get("full_transcription", "No transcription available") if audio_metrics else "No audio available",
                "video_summary": video_analysis.get("video_summary", {}),
                "analysis_status": "completed"
            })

        except Exception as video_error:
            print(f"Video analysis failed: {video_error}")
            # Fallback to basic processing if video analysis fails
            return jsonify({
                "success": True,
                "message": "Video saved but analysis failed. Basic processing completed.",
                "transcription": "Video analysis unavailable",
                "analysis_status": "failed"
            })

    except Exception as e:
        print(f"Video processing error: {e}")
        return jsonify({"error": f"Failed to process video: {str(e)}"}), 500


@app.route("/final_report", methods=["GET"])
@login_required
def final_report():
    if "questions" not in session or "answers" not in session:
        return jsonify({"error": "No interview data found"}), 400

    questions = session.get("questions", [])
    answers = session.get("answers", [])
    role = session.get("role", "Unknown Role")

    report_answers = []
    for idx, question in enumerate(questions):
        answer_obj = next((a for a in answers if a.get("question_number") == idx + 1), None)
        if answer_obj:
            answer = answer_obj.get("answer", "")
            feedback = answer_obj.get("feedback", {})
            mode = answer_obj.get("mode", "text")
            audio_metrics = answer_obj.get("audio_metrics")
        else:
            answer = "<No answer provided>"
            feedback = {"scores": {}, "comment": "No feedback available."}
            mode = "text"
            audio_metrics = None

        report_answers.append({
            "question_number": idx + 1,
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "mode": mode,
            "audio_metrics": audio_metrics,
            "video_analysis": answer_obj.get("video_analysis") if answer_obj and answer_obj.get("mode") == "video" else None
        })

    total_scores = {m: 0 for m in ["Clarity", "Knowledge", "Conciseness", "Confidence", "Structure"]}
    count = 0
    for ans in answers:
        if ans.get("feedback") and ans["feedback"].get("scores"):
            for key in total_scores:
                total_scores[key] += ans["feedback"]["scores"].get(key, 0)
            count += 1
    avg_scores = {k: round(v / count, 1) if count else 0 for k, v in total_scores.items()}
    
    # Calculate overall score
    overall_score = round(sum(avg_scores.values()) / len(avg_scores), 1) if avg_scores else 0.0
    
    # Update database session if exists
    try:
        db_session_id = session.get("db_session_id")
        if db_session_id:
            interview_session = InterviewSession.get_by_session_id(db_session_id)
            if interview_session:
                interview_session.complete_session(avg_scores, overall_score)
    except Exception as e:
        print(f"Error completing interview session: {e}")

    return jsonify({
        "success": True,
        "role": role,
        "total_questions": len(questions),
        "answers": report_answers,
        "average_scores": avg_scores,
        "overall_score": overall_score
    })


@app.route("/submit_session_feedback", methods=["POST"])
def submit_session_feedback():
    data = request.get_json() or {}

    # Ensure we have or create a CSV session id
    csv_session_id = session.get("csv_session_id")
    if not csv_session_id:
        role = session.get("role", "Unknown Role")
        interview_type = session.get("interview_type", "")
        level = session.get("level", "")
        interview_mode = session.get("interview_mode", "text")
        csv_session_id = feedback_system.create_session(role, interview_type, level, interview_mode)
        session["csv_session_id"] = csv_session_id

    # Parse payload safely
    def to_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "y")
        return None

    overall_experience_rating = to_int(data.get("overall_experience_rating"))
    ai_helpfulness_rating = to_int(data.get("ai_helpfulness_rating"))
    feature_ratings = data.get("feature_ratings") if isinstance(data.get("feature_ratings"), dict) else None
    improvement_suggestions = data.get("improvement_suggestions") or None
    would_recommend = to_bool(data.get("would_recommend"))

    # Save session feedback
    feedback_system.add_session_feedback(
        csv_session_id,
        overall_experience_rating=overall_experience_rating,
        ai_helpfulness_rating=ai_helpfulness_rating,
        feature_ratings=feature_ratings,
        improvement_suggestions=improvement_suggestions,
        would_recommend=would_recommend,
    )

    # Update completion metrics if available
    try:
        total_questions = len(session.get("questions", []))
        # Compute a simple overall score from recorded answers if present
        answers = session.get("answers", [])
        score_sum = 0
        score_count = 0
        for ans in answers:
            scores = (ans.get("feedback") or {}).get("scores") or {}
            for m in ["Clarity", "Knowledge", "Conciseness", "Confidence", "Structure"]:
                if isinstance(scores.get(m), (int, float)):
                    score_sum += float(scores[m])
                    score_count += 1
        overall_score = round((score_sum / score_count) * 10, 1) if score_count else 0.0  # scale to 100
        feedback_system.update_session_completion(csv_session_id, total_questions, overall_score)
    except Exception:
        pass

    return jsonify({"success": True, "message": "Feedback submitted successfully."})


# ================== Career Automation API (New) ==================
@app.route('/api/upload-cv', methods=['POST'])
def upload_cv():
    try:
        if 'cv_file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['cv_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = os.path.basename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return jsonify({'success': True, 'filename': filename, 'filepath': filepath, 'message': 'CV uploaded successfully'})
        return jsonify({'error': 'Invalid file type. Only PDF, DOC, and DOCX files are allowed.'}), 400
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/tailor-cv', methods=['POST'])
def tailor_cv():
    try:
        if request.content_type and 'multipart/form-data' in request.content_type:
            cv_path = None
            if 'cv' in request.files:
                file = request.files['cv']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                if file and allowed_file(file.filename):
                    filename = os.path.basename(file.filename)
                    cv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(cv_path)
                else:
                    return jsonify({'error': 'Invalid file type. Only PDF, DOC, and DOCX files are allowed.'}), 400
            elif 'cv_path' in request.form:
                cv_path = request.form.get('cv_path')
                if not os.path.exists(cv_path):
                    return jsonify({'error': 'CV file not found'}), 404
            else:
                return jsonify({'error': 'Either CV file or CV path is required'}), 400
            job_description = request.form.get('job_description', '')
            job_title = request.form.get('job_title', '')
            job_link = request.form.get('job_link', '')
            enhanced_description = (f"Job Title: {job_title}\nJob Link: {job_link}\n\nJob Description:\n{job_description}" if job_title and job_link else job_description)
        else:
            data = request.get_json() or {}
            cv_path = data.get('cv_path')
            job_description = data.get('job_description', '')
            if not cv_path or not job_description:
                return jsonify({'error': 'CV path and job description are required'}), 400
            if not os.path.exists(cv_path):
                return jsonify({'error': 'CV file not found'}), 404
            enhanced_description = job_description
        if not (enhanced_description or '').strip():
            return jsonify({'error': 'Job description is required'}), 400
        system = get_career_system()
        
        # Use the template file we just created
        template_path = 'enhanced_font_cv_template.html'
        
        try:
            result = system.agent1_cv_tailor(cv_path, enhanced_description, template_path, app.config['OUTPUT_FOLDER'])
            
            if result and result.get('structured_cv'):
                response_data = {
                    'success': True, 
                    'structured_cv': result.get('structured_cv'), 
                    'message': 'CV tailored successfully'
                }
                
                # Add download URL if PDF was generated
                if result.get('output_file'):
                    output_filename = os.path.basename(result['output_file'])
                    download_url = f'/api/download-cv/{output_filename}'
                    response_data['download_url'] = download_url
                
                return jsonify(response_data)
            else:
                return jsonify({'error': 'Failed to generate tailored CV - no structured data returned'}), 500
                
        except Exception as api_error:
            if "quota" in str(api_error).lower() or "429" in str(api_error):
                return jsonify({
                    'error': 'Gemini API quota exceeded. Please try again later or upgrade your API plan.',
                    'details': 'The free tier has a limit of 50 requests per day.'
                }), 429
            else:
                raise api_error
    except Exception as e:
        return jsonify({'error': f'CV tailoring failed: {str(e)}'}), 500


@app.route('/api/generate-cover-letter', methods=['POST'])
def api_generate_cover_letter():
    try:
        data = request.get_json() or {}
        cv_path = data.get('cv_path')
        job_description = data.get('job_description')
        if not cv_path or not job_description:
            return jsonify({'error': 'CV path and job description are required'}), 400
        if not os.path.exists(cv_path):
            return jsonify({'error': 'CV file not found'}), 404
        system = get_career_system()
        cover_letter = system.agent2_cover_letter(cv_path, job_description)
        return jsonify({'success': True, 'cover_letter': cover_letter, 'message': 'Cover letter generated successfully'})
    except Exception as e:
        return jsonify({'error': f'Cover letter generation failed: {str(e)}'}), 500


@app.route('/api/scrape-jobs', methods=['POST'])
def api_scrape_jobs():
    try:
        data = request.get_json() or {}
        job_description = data.get('job_description')
        location = data.get('location', 'Pakistan')
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        system = get_career_system()
        job_title = system.extract_job_title_from_description(job_description)
        jobs = system.agent3_job_scraper(job_title, location)
        return jsonify({'success': True, 'jobs': jobs, 'total_jobs': len(jobs), 'job_title': job_title, 'location': location, 'message': f'Found {len(jobs)} jobs'})
    except Exception as e:
        return jsonify({'error': f'Job scraping failed: {str(e)}'}), 500


@app.route('/api/analyze-strengths-weaknesses', methods=['POST'])
def api_analyze_strengths_weaknesses():
    try:
        data = request.get_json() or {}
        cv_path = data.get('cv_path')
        job_description = data.get('job_description')
        if not cv_path or not job_description:
            return jsonify({'error': 'CV path and job description are required'}), 400
        if not os.path.exists(cv_path):
            return jsonify({'error': 'CV file not found'}), 404
        system = get_career_system()
        analysis = system.agent4_strengths_weaknesses(cv_path, job_description)
        return jsonify({'success': True, 'analysis': analysis, 'message': 'Analysis completed successfully'})
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/api/suggest-courses', methods=['POST'])
def api_suggest_courses():
    try:
        data = request.get_json() or {}
        cv_path = data.get('cv_path')
        job_description = data.get('job_description')
        if not cv_path or not job_description:
            return jsonify({'error': 'CV path and job description are required'}), 400
        if not os.path.exists(cv_path):
            return jsonify({'error': 'CV file not found'}), 404
        system = get_career_system()
        courses = system.agent5_course_suggestions_with_links(cv_path, job_description)
        return jsonify({'success': True, 'courses': courses, 'message': 'Course suggestions generated successfully'})
    except Exception as e:
        return jsonify({'error': f'Course suggestions failed: {str(e)}'}), 500


@app.route('/api/weekly-learning-plan', methods=['POST'])
def api_weekly_learning_plan():
    try:
        data = request.get_json() or {}
        cv_path = data.get('cv_path')
        job_description = data.get('job_description')
        if not cv_path or not job_description:
            return jsonify({'error': 'CV path and job description are required'}), 400
        if not os.path.exists(cv_path):
            return jsonify({'error': 'CV file not found'}), 404
        system = get_career_system()
        plan = system.agent6_weekly_learning_scheduler(cv_path, job_description)
        return jsonify({'success': True, 'plan': plan, 'message': 'Weekly learning plan generated successfully'})
    except Exception as e:
        return jsonify({'error': f'Learning plan generation failed: {str(e)}'}), 500


@app.route('/api/career-roadmap', methods=['POST'])
def api_career_roadmap():
    try:
        data = request.get_json() or {}
        job_description = data.get('job_description')
        years = data.get('years', 3)
        if not job_description:
            return jsonify({'error': 'Job description is required'}), 400
        system = get_career_system()
        roadmap = system.agent7_career_path_planner(job_description, years)
        return jsonify({'success': True, 'roadmap': roadmap, 'years': years, 'message': 'Career roadmap generated successfully'})
    except Exception as e:
        return jsonify({'error': f'Career roadmap generation failed: {str(e)}'}), 500


@app.route('/api/complete-workflow', methods=['POST'])
def api_complete_workflow():
    try:
        data = request.get_json() or {}
        cv_path = data.get('cv_path')
        job_description = data.get('job_description')
        location = data.get('location', 'Pakistan')
        if not cv_path or not job_description:
            return jsonify({'error': 'CV path and job description are required'}), 400
        if not os.path.exists(cv_path):
            return jsonify({'error': 'CV file not found'}), 404
        system = get_career_system()
        
        try:
            results = system.complete_career_workflow(cv_path=cv_path, job_description=job_description, template_path=None, location=location)
            return jsonify({'success': True, 'results': results, 'message': 'Complete workflow executed successfully'})
        except Exception as api_error:
            if "quota" in str(api_error).lower() or "429" in str(api_error):
                return jsonify({
                    'error': 'Gemini API quota exceeded. Please try again later or upgrade your API plan.',
                    'details': 'The free tier has a limit of 50 requests per day.'
                }), 429
            else:
                raise api_error
    except Exception as e:
        return jsonify({'error': f'Complete workflow failed: {str(e)}'}), 500


@app.route('/api/health')
def api_health():
    return jsonify({'status': 'healthy', 'message': 'Careerly API is running'})


@app.route('/api/download-cv/<filename>')
def api_download_cv(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500


# Register authentication blueprint
app.register_blueprint(auth_bp)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(int(user_id))

if __name__ == "__main__":
    # The database tables are already created when the app starts
    print("Starting Careerly Flask server...")
    app.run(host="127.0.0.1", port=5002, debug=True)
