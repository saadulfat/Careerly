// main.js
document.addEventListener('DOMContentLoaded', () => {
  let questions = [];
  let currentQuestionIndex = 0;
  let interviewMode = 'text'; // 'text' | 'audio' | 'video'

  // Audio state
  let mediaRecorder = null;
  let audioStream = null;
  let audioChunks = [];
  let audioBlob = null;
  let isRecording = false;

  // Video state
  let videoBlob = null;

  // Add webcam recording support for video interview
  let webcamStream = null;
  let webcamRecorder = null;
  let webcamChunks = [];

  /* ----------------------- Helpers ----------------------- */

  const $ = (id) => document.getElementById(id);

  const sections = [
    'landingSection',
    'roleSection',
    'loadingSection',
    'interviewSection',
    'questionFeedbackSection',
    'reportSection',
    'feedbackSection',
    'errorSection',
  ];

  /* ----------------------- Visual FX: Parallax & Spotlight ----------------------- */
  const parallaxLayers = Array.from(document.querySelectorAll('.parallax-layer'));
  const spotlight = document.getElementById('spotlight');
  function onMouseMove(e) {
    const { innerWidth: w, innerHeight: h } = window;
    const x = (e.clientX - w / 2) / (w / 2);
    const y = (e.clientY - h / 2) / (h / 2);
    parallaxLayers.forEach((layer, idx) => {
      const depth = (idx + 1) * 8; // varying depth
      layer.style.transform = `translate(${x * depth}px, ${y * depth}px)`;
    });
    if (spotlight) {
      spotlight.style.left = `${e.clientX}px`;
      spotlight.style.top = `${e.clientY}px`;
    }
  }
  window.addEventListener('mousemove', onMouseMove);

  // Tilt effect utility
  function attachTilt(el) {
    if (!el) return;
    const strength = 8;
    el.addEventListener('mousemove', (e) => {
      const rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (e.clientX - cx) / (rect.width / 2);
      const dy = (e.clientY - cy) / (rect.height / 2);
      el.style.transform = `rotateX(${(-dy * strength).toFixed(2)}deg) rotateY(${(dx * strength).toFixed(2)}deg)`;
    });
    el.addEventListener('mouseleave', () => {
      el.style.transform = 'rotateX(0deg) rotateY(0deg)';
    });
  }
  Array.from(document.querySelectorAll('.tilt')).forEach(attachTilt);

  function showOnly(sectionId) {
    sections.forEach((id) => {
      const el = $(id);
      if (!el) return;
      if (id === sectionId) el.classList.remove('hidden');
      else el.classList.add('hidden');
    });
  }

  function showError(message) {
    showOnly('errorSection');
    const errorMessage = $('errorMessage');
    if (errorMessage) errorMessage.textContent = message || 'Something went wrong.';
    toast('Error', message || 'Something went wrong.', 'error');
  }

  // Toast notifications
  function toast(title, message, type = 'info', timeout = 3500) {
    const container = $('toastContainer');
    if (!container) return;
    const el = document.createElement('div');
    el.className = `toast toast-${type === 'success' ? 'success' : type === 'error' ? 'error' : 'info'} glass`;
    el.innerHTML = `
      <div>
        <div class="toast-title">${title}</div>
        <div class="text-sm opacity-90">${message}</div>
      </div>
      <button class="toast-close" aria-label="Close">✕</button>
    `;
    el.querySelector('.toast-close').addEventListener('click', () => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(-8px)';
      setTimeout(() => el.remove(), 180);
    });
    container.appendChild(el);
    setTimeout(() => {
      if (el.isConnected) {
        el.style.opacity = '0';
        el.style.transform = 'translateY(-8px)';
        setTimeout(() => el.remove(), 180);
      }
    }, timeout);
  }

  /* ----------------------- Sidebar & Timer ----------------------- */
  let sessionTimerInterval = null;
  let sessionSeconds = 0;
  function startSessionTimer() {
    stopSessionTimer();
    sessionSeconds = 0;
    sessionTimerInterval = setInterval(() => {
      sessionSeconds += 1;
      const m = String(Math.floor(sessionSeconds / 60)).padStart(2, '0');
      const s = String(sessionSeconds % 60).padStart(2, '0');
      const sbTimer = $('sbTimer');
      if (sbTimer) sbTimer.textContent = `${m}:${s}`;
    }, 1000);
  }
  function stopSessionTimer() {
    if (sessionTimerInterval) clearInterval(sessionTimerInterval);
    sessionTimerInterval = null;
  }
  function updateSidebar() {
    const sbRole = $('sbRole');
    const sbMode = $('sbMode');
    const sbQ = $('sbQ');
    if (sbRole) sbRole.textContent = $('roleInput')?.value || (window?.sessionStorage?.getItem('role') || '-');
    if (sbMode) sbMode.textContent = interviewMode || '-';
    if (sbQ) sbQ.textContent = `${currentQuestionIndex + 1}/${questions.length || 0}`;
  }
  const sidebarPanel = $('sidebarPanel');
  const sidebarToggle = $('sidebarToggle');
  const sbGrid = $('sbGrid');
  sidebarToggle?.addEventListener('click', () => {
    if (!sbGrid) return;
    const isHidden = sbGrid.classList.toggle('hidden');
    sidebarToggle.textContent = isHidden ? 'Show' : 'Hide';
  });

  function calculateAverageScore(scores) {
    if (!scores || typeof scores !== 'object') return 'N/A';
    const vals = Object.values(scores);
    if (!vals.length) return 'N/A';
    const sum = vals.reduce((a, b) => a + (Number(b) || 0), 0);
    return (sum / vals.length).toFixed(1);
  }

  function updateInterviewModeUI() {
    const answerInput = $('answerInput');
    const audioControls = $('audioControls');
    const videoControls = $('videoControls');
    const wordCount = $('wordCount');
    const submitAnswer = $('submitAnswer');
    const recordButton = $('recordButton');
    const audioUploadInput = $('audioUploadInput');
    const uploadAudioButton = $('uploadAudioButton');
    const audioTranscriptionPreview = $('audioTranscriptionPreview');

    if (!answerInput || !audioControls || !videoControls || !wordCount || !submitAnswer || !recordButton || !audioUploadInput || !uploadAudioButton || !audioTranscriptionPreview) return;

    if (interviewMode === 'audio') {
      audioControls.classList.remove('hidden');
      videoControls.classList.add('hidden');
      answerInput.classList.add('hidden');
      answerInput.readOnly = true;
      answerInput.value = '';
      wordCount.classList.add('hidden');
      submitAnswer.textContent = 'Submit Answer'; // Submit will now process whatever audio is ready (recorded or uploaded)

      // Reset audio specific UI
      audioTranscriptionPreview.textContent = 'Record your answer or upload an audio file below.';
      recordButton.classList.remove('recording');
      $('recordingStatus').classList.add('hidden');
      $('audioWaveform').classList.add('hidden');
      audioUploadInput.value = null; // Clear selected file

      // Disable submit initially until an audio source is ready
      submitAnswer.disabled = true;

         } else if (interviewMode === 'video') {
       videoControls.classList.remove('hidden');
       audioControls.classList.add('hidden');
       answerInput.classList.add('hidden');
       wordCount.classList.add('hidden');
      submitAnswer.textContent = 'Submit Video';
       submitAnswer.disabled = !videoBlob;
       
       // Reset video preview and webcam state
       if ($('videoPreview')) {
         $('videoPreview').textContent = 'Record your answer or upload a video file below.';
       }
       
       // Reset webcam buttons to initial state
       if ($('startWebcamButton')) $('startWebcamButton').disabled = false;
       if ($('stopWebcamButton')) $('stopWebcamButton').disabled = true;
       
       // Clear any existing webcam stream
       if (webcamStream) {
         webcamStream.getTracks().forEach(track => track.stop());
         webcamStream = null;
       }
       if (webcamRecorder) {
         webcamRecorder = null;
       }
       webcamChunks = [];
     } else {
      audioControls.classList.add('hidden');
      videoControls.classList.add('hidden');
      answerInput.classList.remove('hidden');
      answerInput.readOnly = false;
      wordCount.classList.remove('hidden');
      submitAnswer.textContent = 'Submit Answer';
      if (isRecording) stopRecording(); // safety
      submitAnswer.disabled = false; // Always enabled for text mode
    }
    updateSidebar();
  }

  function loadQuestion() {
    if (!questions.length || currentQuestionIndex >= questions.length) {
      showError('No more questions available.');
      return;
    }
    showOnly('interviewSection');

    $('questionBox').textContent = questions[currentQuestionIndex];
    $('progressText').textContent = `Question ${currentQuestionIndex + 1} of ${questions.length}`;
    $('progressBar').style.width = `${((currentQuestionIndex + 1) / questions.length) * 100}%`;

         $('answerInput').value = '';
     $('wordCount').textContent = 'Word count: 0/200';
     $('audioTranscriptionPreview').textContent = '';
     $('nextQuestion').classList.add('hidden');
     $('viewReport').classList.add('hidden');
     $('submitAnswer').disabled = false;

     audioBlob = null;
     audioChunks = [];
     videoBlob = null; // Reset video for new question
     
     // Reset video mode UI for new question
     if ($('videoPreview')) {
       $('videoPreview').textContent = 'Record your answer or upload a video file below.';
     }
     
     // Reset webcam state
     if (webcamStream) {
       webcamStream.getTracks().forEach(track => track.stop());
       webcamStream = null;
     }
     if (webcamRecorder) {
       webcamRecorder = null;
     }
     webcamChunks = [];
     
     // Reset webcam buttons
     if ($('startWebcamButton')) $('startWebcamButton').disabled = false;
     if ($('stopWebcamButton')) $('stopWebcamButton').disabled = true;
     
     // Reset video upload input for new question
     if (typeof resetVideoUploadInput === 'function') {
       resetVideoUploadInput();
     }
     
     updateInterviewModeUI();
    toast('New Question', 'Focus on examples and structure your answer.', 'info', 2200);
    updateSidebar();
  }

  function renderFeedback(feedback) {
    const scoresDiv = $('feedbackScores');
    const contentDiv = $('feedbackContent');
    if (!scoresDiv || !contentDiv) return;

    const scores = feedback?.scores || {};
    const avg = calculateAverageScore(scores);

    let scoresHTML = `
      <div class="bg-[#1f251d] border border-[#42513e] rounded-xl p-6">
        <div class="flex justify-between items-center mb-4">
          <h3 class="text-lg font-bold text-white">Question Score</h3>
          <p class="text-2xl font-bold text-[#54d22d]">${avg} / 10</p>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
    `;
    for (const [metric, val] of Object.entries(scores)) {
      scoresHTML += `
        <div class="text-center p-2 bg-[#2d372a] rounded-lg">
          <p class="text-sm text-[#a5b6a0]">${metric}</p>
          <p class="text-lg font-bold text-white">${val} / 10</p>
        </div>
      `;
    }
    scoresHTML += `</div></div>`;
    scoresDiv.innerHTML = scoresHTML;

    const strengthsHTML = (feedback?.strengths || [])
      .map((s) => `<li class="flex items-start gap-3"><span class="text-[#54d22d] mt-1">✓</span><span class="text-[#a5b6a0]">${s}</span></li>`)
      .join('');
    const areasHTML = (feedback?.areas || [])
      .map((a) => `<li class="flex items-start gap-3"><span class="text-yellow-400 mt-1">⚠</span><span class="text-[#a5b6a0]">${a}</span></li>`)
      .join('');

    let contentHTML = '';
    if (strengthsHTML) {
      contentHTML += `<div class="flex flex-col gap-2"><p class="text-white text-base font-bold">Strengths</p><ul class="list-none pl-0 space-y-2">${strengthsHTML}</ul></div>`;
    }
    if (areasHTML) {
      contentHTML += `<div class="flex flex-col gap-2 mt-4"><p class="text-white text-base font-bold">Areas for Improvement</p><ul class="list-none pl-0 space-y-2">${areasHTML}</ul></div>`;
    }

    contentDiv.innerHTML = `<div class="bg-[#1f251d] border border-[#42513e] rounded-xl p-6">${contentHTML}</div>`;
    toast('Feedback Ready', 'Review your strengths and areas to improve.', 'success', 2600);
    updateSidebar();
  }

  function renderReport(data) {
    const summaryDiv = $('reportSummary');
    const reportDiv = $('report');
    if (!summaryDiv || !reportDiv) return;

    // Use overall_score from backend if available, otherwise calculate
    const overall = data?.overall_score !== undefined ? data.overall_score : calculateAverageScore(data?.average_scores);
    const avg = data?.average_scores || {};
    
    summaryDiv.innerHTML = `
      <div class="text-center mb-6">
        <div class="bg-[#1f251d] border border-[#42513e] rounded-xl p-6 mb-4">
          <p class="text-white tracking-light text-3xl font-bold leading-tight mb-2">Final Score: ${overall} / 10</p>
          <p class="text-[#a5b6a0] text-sm font-normal">Overall Performance Rating</p>
        </div>
        <div class="bg-[#1f251d] border border-[#42513e] rounded-xl p-4">
          <h4 class="text-lg font-semibold mb-4 text-white">Individual Category Scores</h4>
          <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div class="text-center p-3 bg-[#2d372a] rounded-lg">
              <div class="text-2xl font-bold text-[#54d22d]">${avg.Clarity ?? 0}</div>
              <div class="text-xs text-[#a5b6a0] mt-1">Clarity</div>
            </div>
            <div class="text-center p-3 bg-[#2d372a] rounded-lg">
              <div class="text-2xl font-bold text-[#54d22d]">${avg.Knowledge ?? 0}</div>
              <div class="text-xs text-[#a5b6a0] mt-1">Knowledge</div>
            </div>
            <div class="text-center p-3 bg-[#2d372a] rounded-lg">
              <div class="text-2xl font-bold text-[#54d22d]">${avg.Conciseness ?? 0}</div>
              <div class="text-xs text-[#a5b6a0] mt-1">Conciseness</div>
            </div>
            <div class="text-center p-3 bg-[#2d372a] rounded-lg">
              <div class="text-2xl font-bold text-[#54d22d]">${avg.Confidence ?? 0}</div>
              <div class="text-xs text-[#a5b6a0] mt-1">Confidence</div>
            </div>
            <div class="text-center p-3 bg-[#2d372a] rounded-lg">
              <div class="text-2xl font-bold text-[#54d22d]">${avg.Structure ?? 0}</div>
              <div class="text-xs text-[#a5b6a0] mt-1">Structure</div>
            </div>
          </div>
        </div>
      </div>
    `;

    let html = '';
    (data?.answers || []).forEach((item, i) => {
      const qAvg = calculateAverageScore(item?.feedback?.scores || {});
      let audioMetricsHTML = '';
      if (item?.mode === 'audio' && item?.audio_metrics) {
        const m = item.audio_metrics;
        audioMetricsHTML = `
          <div class="mt-3 p-3 bg-[#1f251d] rounded-lg border border-[#42513e]">
            <p class="font-bold text-white mb-1">Speaking Metrics:</p>
            <ul class="list-disc list-inside text-sm text-[#a5b6a0]">
              <li>Speech Duration: ${Number(m.total_speech_duration)?.toFixed(2) || 'N/A'}s</li>
              <li>Filler Words: ${m.filler_word_count || 0} (Total Duration: ${Number(m.total_filler_duration)?.toFixed(2) || 'N/A'}s)</li>
              <li>Pauses: ${m.num_pauses || 0} (Total Duration: ${Number(m.total_pause_duration)?.toFixed(2) || 'N/A'}s)</li>
              <li>Speech Rate: ${Number(m.speech_rate_wpm)?.toFixed(0) || 'N/A'} wpm</li>
            </ul>
          </div>
        `;
      }

      html += `
        <details class="flex flex-col rounded-xl bg-[#2d372a] px-4 py-2 group" ${i === 0 ? 'open' : ''}>
          <summary class="flex cursor-pointer items-center justify-between gap-6 py-2 list-none">
            <p class="text-white text-sm font-medium leading-normal flex-1">Q${i + 1}: ${item.question}</p>
            <div class="flex items-center gap-2 text-right">
              <span class="text-sm font-bold text-[#54d22d]">Score: ${qAvg}/10</span>
              <div class="text-white group-open:rotate-180 transition-transform"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 256 256"><path d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z"></path></svg></div>
            </div>
          </summary>
          <div class="border-t border-[#42513e] mt-2 pt-3 text-[#a5b6a0] text-sm font-normal pb-2">
            <p class="font-bold text-white mb-1">Your Answer (${item.mode} mode):</p>
            <p class="mb-3">${item.answer || 'No answer provided.'}</p>
            ${audioMetricsHTML}
            ${item?.feedback?.strengths?.length ? `<p class="font-bold text-white mb-1 mt-3">Strengths:</p><ul class="list-disc list-inside mb-3">${item.feedback.strengths.map((s)=>`<li>${s}</li>`).join('')}</ul>` : ''}
            ${item?.feedback?.areas?.length ? `<p class="font-bold text-white mb-1">Areas for Improvement:</p><ul class="list-disc list-inside">${item.feedback.areas.map((a)=>`<li>${a}</li>`).join('')}</ul>` : ''}
          </div>
        </details>
      `;
    });
    reportDiv.innerHTML = html;
    toast('Report Ready', 'Your interview summary has been generated.', 'success', 2800);
    updateSidebar();
  }

  function renderVideoReport(report) {
    const summaryDiv = $('reportSummary');
    const reportDiv = $('report');
    if (!summaryDiv || !reportDiv) return;
    
    // Handle both old and new report formats
    if (typeof report === 'string') {
      // New format: report is a string (the combined analysis)
      summaryDiv.innerHTML = `<div class="bg-[#1f251d] border border-[#42513e] rounded-xl p-6">
        <h3 class="text-lg font-bold text-white mb-2">Video Interview Analysis</h3>
        <div class="text-[#a5b6a0] whitespace-pre-line">${report}</div>
      </div>`;
      reportDiv.innerHTML = '';
    } else {
      // Old format: report is an object with separate fields
      summaryDiv.innerHTML = `<div class="bg-[#1f251d] border border-[#42513e] rounded-xl p-6">
        <h3 class="text-lg font-bold text-white mb-2">Video Interview Analysis</h3>
        <p class="text-white">${report.video_report || 'Analysis completed'}</p>
        <h3 class="text-lg font-bold text-white mt-4 mb-2">Audio Metrics</h3>
        <pre class="text-[#a5b6a0]">${JSON.stringify(report.audio_metrics || {}, null, 2)}</pre>
      </div>`;
      reportDiv.innerHTML = '';
    }
    toast('Video Analysis', 'Combined analysis is ready to review.', 'success', 2600);
    updateSidebar();
  }

  /* ----------------------- Audio Recording ----------------------- */

// Replace the startRecording function with better format detection
async function startRecording() {
  try {
    audioChunks = [];
    audioBlob = null;

    audioStream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 44100
      }
    });

    // Try different formats in order of preference for Whisper
    const supportedTypes = [
      'audio/wav',
      'audio/mp4',
      'audio/webm;codecs=opus',
      'audio/webm'
    ];

    let selectedType = 'audio/webm'; // fallback
    for (const type of supportedTypes) {
      if (MediaRecorder.isTypeSupported(type)) {
        selectedType = type;
        break;
      }
    }

    mediaRecorder = new MediaRecorder(audioStream, { 
      mimeType: selectedType,
      audioBitsPerSecond: 128000
    });

    // Rest of your existing code...
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      // Create blob with proper type
      audioBlob = new Blob(audioChunks, { type: selectedType });
      isRecording = false;
      $('recordButton').classList.remove('recording');
      $('recordingStatus').classList.add('hidden');
      $('audioWaveform').classList.add('hidden');
      $('submitAnswer').disabled = false;

      if (audioStream) audioStream.getTracks().forEach((t) => t.stop());
    };

      mediaRecorder.start();
      isRecording = true;

      $('recordButton').classList.add('recording');
      $('recordingStatus').classList.remove('hidden');
      $('audioWaveform').classList.remove('hidden');
      $('submitAnswer').disabled = true;
      $('audioTranscriptionPreview').textContent = 'Recording your answer...';
    // Rest of existing startRecording code...
  } catch (err) {
    console.error('Mic error:', err);
    showError('Microphone access denied or failed. Please check browser permissions.');
  }
}

  function stopRecording() {
    try {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
      if (audioStream) {
        audioStream.getTracks().forEach((t) => t.stop());
      }
    } catch (e) {
      console.warn('stopRecording error:', e);
    }
  }

async function uploadAudioAndGetTranscription() {
  console.log("=== FRONTEND AUDIO DEBUG START ===");
  
  if (!audioBlob) throw new Error('No audio recorded. Please record your answer first.');
  
  console.log("Audio blob type:", audioBlob.type);
  console.log("Audio blob size:", audioBlob.size);
  
  const fd = new FormData();
  
  if (audioBlob instanceof File) {
    console.log("Uploading File object:", audioBlob.name);
    fd.append('audio_file', audioBlob);
  } else {
    let filename = 'answer.webm';
    if (audioBlob.type.includes('wav')) filename = 'answer.wav';
    else if (audioBlob.type.includes('mp3')) filename = 'answer.mp3';
    else if (audioBlob.type.includes('mp4')) filename = 'answer.m4a';
    
    console.log("Uploading Blob with filename:", filename);
    fd.append('audio_file', audioBlob, filename);
  }

  const res = await fetch('/upload_audio_answer', {
    method: 'POST',
    body: fd,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error || 'Failed to process audio.');
  }

  const data = await res.json();
  console.log("Backend response:", data);
  console.log("Audio file saved at:", data.debug_file_path);
  
  if (data?.transcription) {
    console.log("Transcription received:", data.transcription);
    return data.transcription;
  }
  throw new Error('No transcription returned.');
}

  /* ----------------------- Event Listeners ----------------------- */

  // Word count
  $('answerInput')?.addEventListener('input', (e) => {
    const text = e.target.value || '';
    // count words (split on whitespace)
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    $('wordCount').textContent = `Word count: ${words}/200`;
    // Optional: soft limit (do not enforce hard cap)
  });

  // Start/Stop recording toggle
  $('recordButton')?.addEventListener('click', () => {
    if (interviewMode !== 'audio') return;
    if (!isRecording) startRecording();
    else stopRecording();
  });

    // Handle audio file upload
// Handle audio file upload
$('audioUploadInput')?.addEventListener('change', (event) => {
  if (interviewMode !== 'audio') return;
  const file = event.target.files[0];
  if (file) {
    if (isRecording) stopRecording(); // Stop any active recording
    
    // Store the file properly for upload
    audioBlob = file;
    $('audioTranscriptionPreview').textContent = `File selected: ${file.name}. Click "Upload Audio" to transcribe.`;
    $('uploadAudioButton').disabled = false;
    $('submitAnswer').disabled = false;
  } else {
    audioBlob = null;
    $('audioTranscriptionPreview').textContent = 'Record your answer or upload an audio file below.';
    $('uploadAudioButton').disabled = true;
    $('submitAnswer').disabled = true;
  }
});

  // Handle explicit upload & transcribe button click
  $('uploadAudioButton')?.addEventListener('click', async () => {
    if (interviewMode !== 'audio' || !audioBlob) return;

    // Prevent re-uploading the same file if already processed
    if ($('audioTranscriptionPreview').textContent.startsWith('Transcription:')) {
      alert('Audio already transcribed. You can submit your answer.');
      return;
    }

    $('uploadAudioButton').disabled = true;
    $('submitAnswer').disabled = true;
    $('audioTranscriptionPreview').textContent = 'Uploading and transcribing audio...';
    toast('Uploading', 'Your audio is being transcribed...', 'info', 1800);

    try {
      await uploadAudioAndGetTranscription();
      $('audioTranscriptionPreview').textContent = `Transcription: ${$('audioTranscriptionPreview').textContent}`;
      $('submitAnswer').disabled = false; // Enable submit after transcription
      toast('Transcribed', 'Audio transcription complete. Review and submit.', 'success', 2200);
    } catch (error) {
      console.error('Upload and transcribe failed:', error);
      $('audioTranscriptionPreview').textContent = `Error: ${error.message}. Please try again.`;
      toast('Transcription Failed', error.message || 'Please try again.', 'error');
      $('uploadAudioButton').disabled = false; // Re-enable upload button
    } finally {
        $('uploadAudioButton').disabled = false;
    }
  });

  // Role form submit -> start interview
  $('roleForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    showOnly('loadingSection');
    toast('Generating', 'Creating personalized questions...', 'info', 1800);

    try {
      const role = $('roleInput')?.value?.trim() || '';
      const interview_type = $('typeSelect')?.value || '';
      const level = $('levelSelect')?.value || '';
      const selectedMode = document.querySelector('input[name="interviewMode"]:checked');
      interviewMode = selectedMode ? selectedMode.value : 'text';

      if (!role) throw new Error('Please enter a role to proceed.');
      if (!interview_type || !level) throw new Error('Please select interview type and level.');

      const res = await fetch('/start_interview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role,
          interview_type,
          level,
          interview_mode: interviewMode,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error || 'Failed to start the interview.');
      }

      const data = await res.json();
      questions = data?.questions || [];
      currentQuestionIndex = 0;

      if (!questions.length) throw new Error('No questions generated. Please try again.');

      loadQuestion();
      toast('Ready', 'First question is ready. Good luck!', 'success', 2000);
      // start sidebar & timer
      if ($('sidebarPanel')) $('sidebarPanel').classList.remove('hidden');
      updateSidebar();
      startSessionTimer();
    } catch (err) {
      showError(err.message || 'Failed to start the interview.');
    }
  });

  // Career Automation API helpers
  async function apiUploadCV(file) {
    const fd = new FormData();
    fd.append('cv_file', file);
    const res = await fetch('/api/upload-cv', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Upload failed');
    return data.filepath;
  }

  async function apiTailorCV({ cvPath, jobDescription, file, jobTitle, jobLink }) {
    let body;
    let headers;
    let url = '/api/tailor-cv';
    if (file) {
      const fd = new FormData();
      fd.append('cv', file);
      fd.append('job_description', jobDescription || '');
      if (jobTitle) fd.append('job_title', jobTitle);
      if (jobLink) fd.append('job_link', jobLink);
      body = fd;
    } else {
      headers = { 'Content-Type': 'application/json' };
      body = JSON.stringify({ cv_path: cvPath, job_description: jobDescription });
    }
    const res = await fetch(url, { method: 'POST', headers, body });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'CV tailoring failed');
    return data;
  }

  async function apiCoverLetter(cvPath, jobDescription) {
    const res = await fetch('/api/generate-cover-letter', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cv_path: cvPath, job_description: jobDescription }) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Cover letter failed');
    return data.cover_letter;
  }

  async function apiScrapeJobs(jobDescription, location) {
    const res = await fetch('/api/scrape-jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_description: jobDescription, location }) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Job scraping failed');
    return data;
  }

  async function apiStrengths(cvPath, jobDescription) {
    const res = await fetch('/api/analyze-strengths-weaknesses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cv_path: cvPath, job_description: jobDescription }) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Analysis failed');
    return data.analysis;
  }

  async function apiCourses(cvPath, jobDescription) {
    const res = await fetch('/api/suggest-courses', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cv_path: cvPath, job_description: jobDescription }) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Courses failed');
    return data.courses;
  }

  async function apiWeeklyPlan(cvPath, jobDescription) {
    const res = await fetch('/api/weekly-learning-plan', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cv_path: cvPath, job_description: jobDescription }) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Plan failed');
    return data.plan;
  }

  async function apiRoadmap(jobDescription, years) {
    const res = await fetch('/api/career-roadmap', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job_description: jobDescription, years }) });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error || 'Roadmap failed');
    return data.roadmap;
  }

  // Replace placeholder handlers with prompt-driven flows
  const analysisButtons = [
    'btnTailorCV','btnCoverLetter','btnCVAnalysis','btnFindJobs','btnRoadmap','btnWeeklyPlan','btnCourses'
  ];
  analysisButtons.forEach(id => {
    const el = $(id);
    el?.addEventListener('click', async () => {
      try {
        if (id === 'btnTailorCV') {
          const useUpload = confirm('Upload a CV file? Click Cancel to use a server path.');
          let result;
          if (useUpload) {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pdf,.doc,.docx';
            input.onchange = async () => {
              const file = input.files[0];
              const jd = prompt('Paste job description:') || '';
              const jt = prompt('Job title (optional):') || '';
              const jl = prompt('Job link (optional):') || '';
              toast('Tailoring', 'Generating tailored CV...', 'info');
              const data = await apiTailorCV({ file, jobDescription: jd, jobTitle: jt, jobLink: jl });
              toast('Done', 'CV tailored. Download from: ' + data.download_url, 'success');
            };
            input.click();
            return;
          } else {
            const cvp = prompt('Enter server CV path:') || '';
            const jd = prompt('Paste job description:') || '';
            toast('Tailoring', 'Generating tailored CV...', 'info');
            result = await apiTailorCV({ cvPath: cvp, jobDescription: jd });
            toast('Done', 'CV tailored. Download from: ' + result.download_url, 'success');
          }
        } else if (id === 'btnCoverLetter') {
          const cvp = prompt('Enter server CV path:') || '';
          const jd = prompt('Paste job description:') || '';
          toast('Generating', 'Creating cover letter...', 'info');
          const letter = await apiCoverLetter(cvp, jd);
          alert(letter);
        } else if (id === 'btnCVAnalysis') {
          const cvp = prompt('Enter server CV path:') || '';
          const jd = prompt('Paste job description:') || '';
          toast('Analyzing', 'Evaluating strengths and weaknesses...', 'info');
          const text = await apiStrengths(cvp, jd);
          alert(text);
        } else if (id === 'btnFindJobs') {
          const jd = prompt('Paste job description:') || '';
          const loc = prompt('Location (default Pakistan):') || 'Pakistan';
          toast('Searching', 'Scraping jobs...', 'info');
          const data = await apiScrapeJobs(jd, loc);
          alert(`Found ${data.total_jobs} jobs for ${data.job_title} in ${data.location}`);
        } else if (id === 'btnRoadmap') {
          const jd = prompt('Paste job description:') || '';
          const yrs = parseInt(prompt('Years (default 3):') || '3', 10);
          toast('Planning', 'Generating roadmap...', 'info');
          const rm = await apiRoadmap(jd, yrs);
          alert(rm);
        } else if (id === 'btnWeeklyPlan') {
          const cvp = prompt('Enter server CV path:') || '';
          const jd = prompt('Paste job description:') || '';
          toast('Planning', 'Generating weekly plan...', 'info');
          const plan = await apiWeeklyPlan(cvp, jd);
          alert(plan);
        } else if (id === 'btnCourses') {
          const cvp = prompt('Enter server CV path:') || '';
          const jd = prompt('Paste job description:') || '';
          toast('Recommending', 'Finding relevant courses...', 'info');
          const courses = await apiCourses(cvp, jd);
          alert(typeof courses === 'string' ? courses : JSON.stringify(courses, null, 2));
        }
      } catch (e) {
        showError(e.message || 'Operation failed');
      }
    });
  });

  // Submit answer (text or audio)
  $('submitAnswer')?.addEventListener('click', async () => {
    try {
      $('submitAnswer').disabled = true;

      let answerText = '';

      if (interviewMode === 'audio') {
        if (isRecording) {
          // If user presses while recording, stop first; UI reenables after stop.
          stopRecording();
          // Prevent accidental immediate submit; user must click again after stop.
          $('submitAnswer').disabled = false;
          return;
        }
        if (!audioBlob) {
          $('submitAnswer').disabled = false;
          alert('No audio recorded or uploaded. Please record your answer or upload an audio file first.');
          return;
        }

        // If audioBlob exists and hasn't been transcribed yet, transcribe it now.
        // The uploadAudioAndGetTranscription function already sets audioTranscriptionPreview
        // and stores metrics in the session.
        if (!$('audioTranscriptionPreview').textContent.startsWith('Transcription:')) {
            // This means an audio file was uploaded or recorded but not yet transcribed for display
            // We need to call it here before submitting the answer
            $('audioTranscriptionPreview').textContent = 'Processing audio for submission...';
            toast('Processing', 'Transcribing your audio...', 'info', 1800);
            answerText = await uploadAudioAndGetTranscription();
        } else {
            // Audio was already transcribed (either from prior 'Upload Audio' click or a previous recording)
            answerText = $('audioTranscriptionPreview').textContent.replace('Transcription: ', '');
        }

      } else if (interviewMode === 'video') {
        if (!videoBlob) {
          $('submitAnswer').disabled = false;
          alert('No video recorded or uploaded. Please record your answer or upload a video file first.');
          return;
        }

        $('submitAnswer').textContent = 'Processing video...';
        toast('Uploading', 'Processing your video answer...', 'info', 2000);

        // Submit video blob for processing
        const fd = new FormData();
        fd.append('video_file', videoBlob);

        const res = await fetch('/submit_video_interview', {
          method: 'POST',
          body: fd,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err?.error || 'Failed to process video.');
        }

        const data = await res.json();
        
        // Use the transcription as the answer text
        answerText = data.transcription || 'Video answer submitted';
        
        // Reset video state for next question
        videoBlob = null;
        if ($('videoPreview')) {
          $('videoPreview').textContent = 'Record your answer or upload a video file below.';
        }
        
        // Reset webcam state if it was used
        if (webcamStream) {
          webcamStream.getTracks().forEach(track => track.stop());
          webcamStream = null;
        }
        if (webcamRecorder) {
          webcamRecorder = null;
        }
        webcamChunks = [];
        
        // Reset submit button for next question
        $('submitAnswer').textContent = 'Submit Video';
        $('submitAnswer').disabled = true; // Will be enabled when new video is ready
        
        // Reset webcam buttons
        if ($('startWebcamButton')) $('startWebcamButton').disabled = false;
        if ($('stopWebcamButton')) $('stopWebcamButton').disabled = true;
        
        // Reset video upload input
        if (videoUploadInput) {
          videoUploadInput.value = '';
        }
      } else {
        answerText = $('answerInput')?.value?.trim() || '';
      }

      const res = await fetch('/submit_answer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q_index: currentQuestionIndex,
          answer: answerText,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error || 'Failed to submit answer.');
      }

      const data = await res.json();
      const { feedback, is_last_question } = data;


      // Show feedback
      showOnly('questionFeedbackSection');
      renderFeedback(feedback);

      // Buttons
      if (is_last_question) {
        $('viewReport').classList.remove('hidden');
        $('nextQuestion').classList.add('hidden');
        toast('Final Question', 'View your full report to finish.', 'info', 2200);
      } else {
        $('nextQuestion').classList.remove('hidden');
        $('viewReport').classList.add('hidden');
        toast('Saved', 'Answer submitted. Proceed to the next question.', 'success', 1800);
      }
      updateSidebar();
    } catch (err) {
      console.error(err);
      showError(err.message || 'Submission failed.');
    } finally {
      $('submitAnswer').disabled = false;
    }
  });

  // Next question
  $('nextQuestion')?.addEventListener('click', () => {
    currentQuestionIndex += 1;
    if (currentQuestionIndex >= questions.length) {
      // Safety: if somehow exceeded, go to report
      $('viewReport').click();
      return;
    }
    loadQuestion();
    showOnly('interviewSection');
  });

  // View report
  $('viewReport')?.addEventListener('click', async () => {
    try {
      const res = await fetch('/final_report', { method: 'GET' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error || 'Failed to fetch final report.');
      }
      const data = await res.json();
      renderReport(data);
      showOnly('reportSection');
      toast('Great Work', 'Review your report and share feedback.', 'success', 2400);
      stopSessionTimer();
      updateSidebar();
    } catch (err) {
      showError(err.message || 'Could not load final report.');
    }
  });

  // Restart
  $('restartBtn')?.addEventListener('click', () => {
    // Show feedback collection instead of restarting immediately
    showOnly('feedbackSection');
    toast('One more step', 'Share quick feedback to help us improve.', 'info', 2400);
  });

  // Error "Try again"
  $('tryAgain')?.addEventListener('click', () => {
    window.location.reload();
  });

          // Video upload input change
     const videoUploadInput = $('videoUploadInput');
     const videoPreview = $('videoPreview');
     const submitAnswerBtn = $('submitAnswer');

     if (videoUploadInput) {
       videoUploadInput.addEventListener('change', function(e) {
         const file = e.target.files && e.target.files[0];
         if (file) {
           videoBlob = file;
           videoPreview.textContent = file.name;
           submitAnswerBtn.disabled = false;
           
           // Show preview message
           if (interviewMode === 'video') {
             $('videoPreview').textContent = `File selected: ${file.name}. Click "Submit Video" to process.`;
           }
         } else {
           videoBlob = null;
           videoPreview.textContent = 'No video selected.';
           submitAnswerBtn.disabled = true;
         }
       });
     }
     
     // Function to reset video upload input for new questions
     function resetVideoUploadInput() {
       if (videoUploadInput) {
         videoUploadInput.value = '';
       }
       videoBlob = null;
       if (videoPreview) {
         videoPreview.textContent = 'Record your answer or upload a video file below.';
       }
       if (submitAnswerBtn) {
         submitAnswerBtn.disabled = true;
         submitAnswerBtn.textContent = 'Submit Video';
       }
     }

  // Webcam recording controls
  $('startWebcamButton')?.addEventListener('click', async () => {
    try {
      webcamChunks = [];
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      webcamRecorder = new MediaRecorder(webcamStream, { mimeType: 'video/webm' });
      webcamRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) webcamChunks.push(e.data);
      };
      webcamRecorder.onstop = () => {
        videoBlob = new Blob(webcamChunks, { type: 'video/webm' });
        $('videoPreview').textContent = 'Webcam recording complete.';
        $('submitAnswer').disabled = false;
      };
      webcamRecorder.start();
      $('startWebcamButton').disabled = true;
      $('stopWebcamButton').disabled = false;
      $('videoPreview').textContent = 'Recording from webcam...';
    } catch (err) {
      showError('Webcam access denied or failed.');
    }
  });

  $('stopWebcamButton')?.addEventListener('click', () => {
    if (webcamRecorder && webcamRecorder.state === 'recording') {
      webcamRecorder.stop();
      webcamStream.getTracks().forEach(track => track.stop());
      $('startWebcamButton').disabled = false;
      $('stopWebcamButton').disabled = true;
    }
  });

  // Show upload progress line for video
function showVideoUploadProgress(msg) {
  let progressLine = $('videoUploadProgress');
  if (!progressLine) {
    progressLine = document.createElement('div');
    progressLine.id = 'videoUploadProgress';
    progressLine.className = 'text-sm text-[#54d22d] mt-2';
    $('videoControls').appendChild(progressLine);
  }
  progressLine.textContent = msg;
}

  /* ----------------------- Feedback Collection ----------------------- */

  // Star rating functionality
  function initializeStarRatings() {
    const starRatings = document.querySelectorAll('.star-rating');
    
    starRatings.forEach(rating => {
      const stars = rating.querySelectorAll('.star');
      let currentRating = 0;
      
      stars.forEach((star, index) => {
        star.addEventListener('click', () => {
          currentRating = index + 1;
          updateStarDisplay(stars, currentRating);
        });
        
        star.addEventListener('mouseenter', () => {
          updateStarDisplay(stars, index + 1);
        });
      });
      
      rating.addEventListener('mouseleave', () => {
        updateStarDisplay(stars, currentRating);
      });
      
      // Store rating in data attribute
      rating.addEventListener('click', () => {
        rating.dataset.rating = currentRating;
      });
    });
  }
  
  function updateStarDisplay(stars, rating) {
    stars.forEach((star, index) => {
      if (index < rating) {
        star.classList.add('active');
      } else {
        star.classList.remove('active');
      }
    });
  }
  
  function getStarRating(ratingId) {
    const rating = $(ratingId);
    return rating ? parseInt(rating.dataset.rating) || 0 : 0;
  }

  // Submit feedback
  $('submitFeedbackBtn')?.addEventListener('click', async () => {
    try {
      $('submitFeedbackBtn').disabled = true;
      $('submitFeedbackBtn').textContent = 'Submitting...';
      toast('Submitting', 'Sending your feedback...', 'info', 2000);
      
      // Collect feedback data
      const feedbackData = {
        overall_experience_rating: getStarRating('overallRating'),
        ai_helpfulness_rating: getStarRating('aiHelpfulnessRating'),
        feature_ratings: {
          question_quality: getStarRating('questionQualityRating'),
          audio_video: getStarRating('audioVideoRating'),
          ui_experience: getStarRating('uiRating')
        },
        improvement_suggestions: $('improvementSuggestions')?.value || '',
        would_recommend: document.querySelector('input[name="recommendation"]:checked')?.value === 'true'
      };
      
      // Submit to backend
      const res = await fetch('/submit_session_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
      });
      
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error || 'Failed to submit feedback.');
      }
      
      // Show thank you message
      $('submitFeedbackBtn').classList.add('hidden');
      $('thankYouMessage').classList.remove('hidden');
      toast('Thank you!', 'Your feedback helps us improve Careerly.', 'success', 3000);
      stopSessionTimer();
    } catch (err) {
      console.error('Feedback submission error:', err);
      alert('Failed to submit feedback. Please try again.');
      $('submitFeedbackBtn').disabled = false;
      $('submitFeedbackBtn').textContent = 'Submit Feedback & Help Us Learn! 🎯';
      toast('Failed', err.message || 'Could not submit feedback.', 'error');
    }
  });

  // Final restart button
  $('finalRestartBtn')?.addEventListener('click', () => {
    window.location.reload();
  });

  // Initialize star ratings when DOM is loaded
  initializeStarRatings();

  /* ----------------------- Tips Widget ----------------------- */
  const tipsToggle = $('tipsToggle');
  const tipsPanel = $('tipsPanel');
  const tipsContent = $('tipsContent');
  tipsToggle?.addEventListener('click', () => {
    tipsPanel?.classList.toggle('hidden');
  });
  function updateTipsByRole(role) {
    if (!tipsContent) return;
    const base = [
      'Keep answers structured: Situation, Task, Action, Result.',
      'Use specific examples and quantify impact where possible.',
      'Pause briefly to think; clarity beats speed.'
    ];
    const roleHints = {
      'developer': ['Mention design trade-offs and performance considerations.', 'Reference testing strategy and CI/CD.'],
      'data': ['Highlight data quality checks and reproducibility.', 'Discuss metrics and validation.'],
      'product': ['Tie answers to user outcomes and KPIs.', 'Demonstrate prioritization and stakeholder alignment.']
    };
    const r = (role || '').toLowerCase();
    let extra = [];
    if (r.includes('engineer') || r.includes('developer')) extra = roleHints.developer;
    else if (r.includes('data') || r.includes('analyst') || r.includes('science')) extra = roleHints.data;
    else if (r.includes('product') || r.includes('pm')) extra = roleHints.product;
    const items = [...base, ...extra];
    tipsContent.innerHTML = items.map(t => `<li>${t}</li>`).join('');
  }

  // Update tips and sidebar role when role changes
  $('roleInput')?.addEventListener('input', (e) => {
    updateTipsByRole(e.target.value);
    updateSidebar();
  });

  /* ----------------------- Keyboard Shortcuts ----------------------- */
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      $('submitAnswer')?.click();
    }
    if (e.key === 'n' && e.shiftKey) {
      $('nextQuestion')?.click();
    }
    if (e.key === 'r' && e.shiftKey) {
      $('viewReport')?.click();
    }
  });

});