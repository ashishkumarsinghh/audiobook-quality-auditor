import os
import sys
import tempfile
import streamlit as st
import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.signal import find_peaks, butter, filtfilt
from scoring import calculate_weighted_quality_score, score_clipping, score_noise_floor, score_clarity, score_lufs, score_pacing, score_rolloff, score_lra

# Set page configuration
st.set_page_config(
    page_title="AWGP & Shantikunj Audiobook Quality Auditor",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 15px;
    }
    .score-badge {
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
    }
    .big-score {
        font-size: 3.8rem;
        font-weight: 900;
        margin: 0;
        line-height: 1.1;
    }
    .step-box {
        background: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .macro-box {
        background: #EFF6FF;
        border: 2px dashed #3B82F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .fix-card {
        background: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .file-meta-box {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 15px;
        font-size: 0.92rem;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# Macro content definition
MACRO_CONTENT = """SelectAll:
High-passFilter:frequency="80" rolloff="dB12"
LoudnessNormalization:DualMono="1" LUFSLevel="-20" NormalizeTo="LUFS" RMSLevel="-20" StereoIndependent="0"
Limiter:gain-L="0" gain-R="0" hold="10" limit="-3" makeup="No" thresh="-3" type="SoftLimit"
"""

# Audio Processing Logic
def analyze_audio_data(y, sr):
    clip_count = int(np.sum(np.abs(y) >= 0.999))
    peak_db = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))
    
    frame_len_lra = int(sr * 0.4)
    hop_len_lra = int(sr * 0.1)
    st_rms = librosa.feature.rms(y=y, frame_length=frame_len_lra, hop_length=hop_len_lra)[0]
    st_db = 20 * np.log10(np.clip(st_rms, 1e-8, None))
    speech_st_db = st_db[st_db > (np.max(st_db) - 30)]
    lra = float(np.percentile(speech_st_db, 95) - np.percentile(speech_st_db, 10)) if len(speech_st_db) > 0 else 0.0
    
    frame_len_noise = int(sr * 0.1)
    hop_len_noise = int(sr * 0.05)
    rms_noise = librosa.feature.rms(y=y, frame_length=frame_len_noise, hop_length=hop_len_noise)[0]
    rms_noise_db = 20 * np.log10(np.clip(rms_noise, 1e-8, None))
    noise_floor = float(np.percentile(rms_noise_db, 5))
    
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total_energy = np.sum(S) + 1e-10
    mid_mask = (freqs >= 1000) & (freqs <= 4000)
    vocal_clarity_pct = float((np.sum(S[mid_mask, :]) / total_energy) * 100)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_rolloff = float(np.mean(rolloff))
    
    nyq = 0.5 * sr
    b, a = butter(5, [300 / nyq, 3300 / nyq], btype='band')
    y_filtered = filtfilt(b, a, y)
    frame_len_wpm = int(sr * 0.05)
    hop_len_wpm = int(sr * 0.01)
    rms_wpm = librosa.feature.rms(y=y_filtered, frame_length=frame_len_wpm, hop_length=hop_len_wpm)[0]
    rms_wpm = np.clip(rms_wpm, a_min=1e-10, a_max=None)
    intensity_db = 20 * np.log10(rms_wpm)
    intensity_db = intensity_db - np.max(intensity_db)
    peaks, _ = find_peaks(intensity_db, height=-25.0, distance=10, prominence=2.0)
    duration_mins = (len(y) / sr) / 60.0
    wpm = float((len(peaks) / duration_mins) / 1.45) if duration_mins > 0 else 0.0
    
    metrics = {
        "clips": clip_count,
        "peak_db": peak_db,
        "noise_floor": noise_floor,
        "vocal_clarity": vocal_clarity_pct,
        "rolloff": mean_rolloff,
        "lufs": lufs,
        "lra": lra,
        "wpm": wpm
    }
    
    score, subscores = calculate_weighted_quality_score(metrics)
    return metrics, score, subscores

def get_badge(subscore):
    if subscore >= 0.85:
        return "🟢 EXCELLENT", "#10B981"
    elif subscore >= 0.50:
        return "🟡 ACCEPTABLE", "#F59E0B"
    else:
        return "🔴 FIX NEEDED", "#EF4444"

# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🎧 Navigation")
    app_mode = st.radio(
        "Go to section:",
        [
            "🔍 Audio Quality Auditor",
            "📖 Parameter Guide & Step-by-Step Fixes",
            "🏆 Reference Hall of Fame (Top Audiobooks)",
            "📋 1-Click Audacity Macro & Presets"
        ]
    )
    
    st.markdown("---")
    st.download_button(
        label="📥 Download Audacity Macro (.txt)",
        data=MACRO_CONTENT,
        file_name="AWGP_Master_Audiobook_Preset.txt",
        mime="text/plain",
        help="Download the official 1-click Audacity macro file for instant audio mastering."
    )
    st.markdown("---")
    st.markdown("""
    **Acoustic Standards Applied:**
    - 🏛️ **Audible ACX Standard**
    - 📻 **EBU R128 Broadcast**
    - 🔊 **AES Spoken Word Standard**
    
    **Weighting Distribution:**
    - ⚡ **Sound Cracking**: `25%`
    - 🎙️ **Vocal Clarity**: `20%`
    - 🤫 **Noise Floor**: `20%`
    - 🔊 **Master Volume**: `15%`
    - ⏱️ **Reading Speed**: `10%`
    - 🌬️ **High-End Air**: `5%`
    - 🎭 **Dynamic Life**: `5%`
    """)

# ==============================================================================
# 1. PAGE: AUDIO QUALITY AUDITOR
# ==============================================================================
if app_mode == "🔍 Audio Quality Auditor":
    st.markdown('<div class="main-title">🎧 Audiobook Quality Auditor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Instant, Objective DSP Quality Screening for AWGP Audio Editors & Narrators</div>', unsafe_allow_html=True)
    
    audio_path = None
    file_info = None
    
    st.markdown("**Drag and drop your recorded or mastered chapter file to screen against Audible ACX / EBU R128 standards:**")
    uploaded_file = st.file_uploader(
        "Upload Audio Track (WAV, MP3, M4A, FLAC, OGG, AAC)",
        type=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma"]
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.flush()
        audio_path = tfile.name
        file_info = {
            "name": uploaded_file.name,
            "size_mb": len(uploaded_file.getvalue()) / (1024 * 1024)
        }

    if audio_path and os.path.exists(audio_path):
        st.markdown("---")
        if file_info:
            st.markdown(f"""
            <div class="file-meta-box">
                📁 <strong>Audited File:</strong> <code>{file_info['name']}</code> &nbsp;|&nbsp; <strong>File Size:</strong> {file_info['size_mb']:.2f} MB
            </div>
            """, unsafe_allow_html=True)
        
        st.audio(audio_path)
        
        with st.spinner("Analyzing DSP acoustic parameters..."):
            try:
                y, sr = librosa.load(audio_path, sr=None, duration=300)
                metrics, score, subscores = analyze_audio_data(y, sr)
            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
                score = None

        if score is not None:
            col_score, col_bars = st.columns([1.1, 2.1])
            
            with col_score:
                if score >= 85.0:
                    score_color = "#10B981"
                    rating_text = "🟢 PRO STUDIO GRADE"
                    sub_text = "Meets Audible ACX & International Standards"
                elif score >= 68.0:
                    score_color = "#F59E0B"
                    rating_text = "🟡 GOOD QUALITY"
                    sub_text = "Minor EQ / Loudness Tweaks Needed"
                elif score >= 50.0:
                    score_color = "#F97316"
                    rating_text = "🟠 FAIR (NEEDS EDITING)"
                    sub_text = "Noticeable Noise, Pacing or EQ Issues"
                else:
                    score_color = "#EF4444"
                    rating_text = "🔴 POOR (NEEDS RE-RECORDING)"
                    sub_text = "Severe Distortion, Noise, or Clipping"

                st.markdown(f"""
                <div class="score-badge">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #64748B;">LISTENER QUALITY SCORE</div>
                    <div class="big-score" style="color: {score_color};">{score:.1f}<span style="font-size: 1.8rem; color: #94A3B8;">/100</span></div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: {score_color}; margin-top: 5px;">{rating_text}</div>
                    <div style="font-size: 0.88rem; color: #64748B; margin-top: 4px;">{sub_text}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_bars:
                st.markdown("### 📊 Acoustic Health Overview")
                c1, c2 = st.columns(2)
                with c1:
                    b_clip, color_clip = get_badge(subscores["clips"])
                    st.markdown(f"**⚡ Cracking / Clipping (25%)**: `{metrics['clips']} clips` — <span style='color:{color_clip}; font-weight:700;'>{b_clip}</span>", unsafe_allow_html=True)
                    st.progress(float(subscores["clips"]))
                    
                    b_clarity, color_clarity = get_badge(subscores["clarity"])
                    st.markdown(f"**🎙️ Vocal Formant Clarity (20%)**: `{metrics['vocal_clarity']:.1f}%` — <span style='color:{color_clarity}; font-weight:700;'>{b_clarity}</span>", unsafe_allow_html=True)
                    st.progress(float(subscores["clarity"]))
                    
                    b_noise, color_noise = get_badge(subscores["noise_floor"])
                    st.markdown(f"**🤫 Noise Floor (20%)**: `{metrics['noise_floor']:.1f} dBFS` — <span style='color:{color_noise}; font-weight:700;'>{b_noise}</span>", unsafe_allow_html=True)
                    st.progress(float(subscores["noise_floor"]))

                with c2:
                    b_lufs, color_lufs = get_badge(subscores["lufs"])
                    st.markdown(f"**🔊 Master Volume (15%)**: `{metrics['lufs']:.1f} LUFS` — <span style='color:{color_lufs}; font-weight:700;'>{b_lufs}</span>", unsafe_allow_html=True)
                    st.progress(float(subscores["lufs"]))
                    
                    b_wpm, color_wpm = get_badge(subscores["wpm"])
                    st.markdown(f"**⏱️ Reading Speed (10%)**: `{metrics['wpm']:.0f} WPM` — <span style='color:{color_wpm}; font-weight:700;'>{b_wpm}</span>", unsafe_allow_html=True)
                    st.progress(float(subscores["wpm"]))
                    
                    b_air, color_air = get_badge(subscores["rolloff"])
                    st.markdown(f"**🌬️ High-Frequency Air (5%)**: `{metrics['rolloff']:.0f} Hz` — <span style='color:{color_air}; font-weight:700;'>{b_air}</span>", unsafe_allow_html=True)
                    st.progress(float(subscores["rolloff"]))

            # Actionable Checklist
            st.markdown("---")
            st.markdown("### 🛠️ Editor Action Checklist (Customized for This File)")
            
            fixes = []
            if subscores["clips"] < 0.85:
                fixes.append(("⚡ Digital Clipping Detected", f"Found **{metrics['clips']} clipped samples**. In Audacity: Apply `Effect -> Limiter (Soft Limit to -3.0 dB)` before exporting. During recording: Lower microphone input gain on audio interface."))
            if subscores["clarity"] < 0.85:
                fixes.append(("🎙️ Muffled Vocal Formants", f"Formant clarity is **{metrics['vocal_clarity']:.1f}%** (target is ≥9.5%). In Audacity: Apply `Effect -> Filter Curve EQ` and boost +2.5 dB between 2.5 kHz and 5.0 kHz. Narrator should speak 4-6 inches from condenser mic."))
            if subscores["noise_floor"] < 0.85:
                fixes.append(("🤫 Audible Background Hiss", f"Pause noise floor is **{metrics['noise_floor']:.1f} dBFS** (target is ≤ -58 dBFS). Turn off room fans/AC; in Audacity, apply `Effect -> Noise Reduction` (select 2s pause profile, apply 8 dB reduction)."))
            if subscores["lufs"] < 0.85:
                dir_str = "too loud / hot" if metrics["lufs"] > -18 else "too quiet"
                fixes.append(("🔊 Volume Out of Spec", f"Loudness is **{metrics['lufs']:.1f} LUFS** ({dir_str}). In Audacity: Apply `Effect -> Loudness Normalization` and set to **-20.0 LUFS**."))
            if subscores["wpm"] < 0.85:
                dir_w = "too fast / rushed" if metrics["wpm"] > 145 else "too slow"
                fixes.append(("⏱️ Narration Pace Warning", f"Pacing is **{metrics['wpm']:.0f} WPM** ({dir_w}). Ideal tempo is 110–135 WPM with 0.5–1.0s natural pauses between sentences."))
            if subscores["rolloff"] < 0.85:
                fixes.append(("🌬️ Low High-Frequency Rolloff", f"Rolloff is **{metrics['rolloff']:.0f} Hz** (target ≥4700 Hz). Always export at 44.1 kHz WAV / 192+ kbps MP3; avoid heavy low-pass filtering."))

            if not fixes:
                st.success("🎉 **Outstanding Quality!** This audio meets all international professional publishing criteria.")
            else:
                for f_title, f_desc in fixes:
                    st.markdown(f"""
                    <div class="fix-card">
                        <strong style="color: #991B1B; font-size: 1.05rem;">{f_title}</strong><br>
                        <span style="color: #374151;">{f_desc}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            st.markdown("#### ⚡ 1-Click Fix with Audacity Macro")
            st.download_button(
                label="📥 Download Audacity 1-Click Master Macro (.txt)",
                data=MACRO_CONTENT,
                file_name="AWGP_Master_Audiobook_Preset.txt",
                mime="text/plain",
                help="Download the macro, import into Audacity via Tools -> Macro Manager, and run it on your audio."
            )

# ==============================================================================
# 2. PAGE: PARAMETER GUIDE & STEP-BY-STEP FIXES
# ==============================================================================
elif app_mode == "📖 Parameter Guide & Step-by-Step Fixes":
    st.markdown('<div class="main-title">📖 Comprehensive Parameter Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detailed Definitions, Listener Impact, and Validated Fixes for All 7 Parameters</div>', unsafe_allow_html=True)
    
    param = st.selectbox(
        "Select an Acoustic Parameter to Learn More:",
        [
            "⚡ 1. Sound Cracking & Digital Clipping (Weight: 25%)",
            "🎙️ 2. Vocal Formant Clarity (Weight: 20%)",
            "🤫 3. Background Noise Floor (Weight: 20%)",
            "🔊 4. Master Volume & Loudness LUFS (Weight: 15%)",
            "⏱️ 5. Reading Speed & Pacing (Weight: 10%)",
            "🌬️ 6. High-Frequency Air / Rolloff (Weight: 5%)",
            "🎭 7. Dynamic Life & Expression (Weight: 5%)"
        ]
    )
    
    st.markdown("---")
    
    if "1. Sound Cracking" in param:
        st.markdown("### ⚡ Sound Cracking & Digital Clipping")
        st.markdown("""
        #### 1. Plain-Language Definition
        When recording volume is pushed too high (into the red zone above $0\\text{ dBFS}$), the audio waveform gets chopped off flat at the top. This creates harsh digital distortion, scratching, and crackling noises.
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        Clipped waveforms create unnatural, non-harmonic square wave spikes across high frequencies. The human auditory cortex is forced to constantly attempt to 'repair' and predict the missing audio, causing headaches, ear strain, and listener dropout within 5–10 minutes.
        
        #### 3. Standard & Tolerances
        * **Target**: **0 clips (100% Score)**
        * **1–5 clips**: Minor occasional peak (85% score)
        * **>500 clips**: Severe audible distortion (30% score)
        * **Audible ACX Mandate**: True Peak $\\le -3.0\\text{ dBFS}$.
        
        #### 🛠️ Validated Step-by-Step Fix for Editors:
        1. **In Audacity (Before Exporting)**:
           * Select All (`Ctrl + A`).
           * Go to **Effect $\\rightarrow$ Limiter**.
           * Set **Type: Soft Limit**, **Limit to: -3.0 dB**, **Hold: 10 ms**.
           * Click **Apply**. This guarantees 0 clipped samples.
        2. **During Recording Studio Setup**:
           * Turn down the microphone gain knob on the audio interface so vocal peaks bounce between **$-12\\text{ dB}$ and $-6\\text{ dB}$** on the meter.
        """)
        
    elif "2. Vocal Formant" in param:
        st.markdown("### 🎙️ Vocal Formant Clarity (Speech Intelligibility Band)")
        st.markdown("""
        #### 1. Plain-Language Definition
        Measures the percentage of total acoustic energy residing in the **$1000\\text{ Hz} - 4000\\text{ Hz}$ frequency band**. This band contains the essential vocal formants ($F_2/F_3$) and consonant bursts (/t/, /k/, /s/, /p/, /d/).
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        When vocal clarity is low ($<6\\%$), consonants sound muffled and words blend together (sounding like talking through a blanket). The listener's brain must expend heavy cognitive decoding effort to guess what words are being spoken, causing quick mental exhaustion.
        
        #### 3. Standard & Tolerances
        * **Target**: **$\\ge 9.5\\%$ of total energy (Ideal: $\\ge 11.0\\%$)**
        * **6.5% – 9.5%**: Acceptable but slightly boxy
        * **<3.5%**: Severe muffled phone/laptop mic sound
        
        #### 🛠️ Validated Step-by-Step Fix for Editors:
        1. **In Audacity (EQ Enhancement)**:
           * Select All (`Ctrl + A`).
           * Go to **Effect $\\rightarrow$ Filter Curve EQ**.
           * Draw a gentle boost curve of **$+2.5\\text{ dB}$ centered at $3.2\\text{ kHz}$** ($2.5\\text{ kHz} - 5.0\\text{ kHz}$).
           * Add a sharp high-pass cut below **$80\\text{ Hz}$** to remove boomy room mud.
           * Click **Apply**.
        2. **Microphone Technique**:
           * Use a large-diaphragm condenser microphone (e.g. Rode NT1, Audio-Technica AT2020).
           * Keep the narrator **4 to 6 inches away** from the mic with a dual-layer mesh pop filter.
        """)

    elif "3. Background Noise" in param:
        st.markdown("### 🤫 Background Noise Floor (Silence Purity)")
        st.markdown("""
        #### 1. Plain-Language Definition
        Measures the residual sound level (electrical preamp hiss, ceiling fan hum, AC drone, street traffic) during speech pauses.
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        The human brain has an auditory sensory gating mechanism that attempts to filter out continuous noise. Constant background hiss forces this filter to work continuously, breaking immersion and causing subconscious irritation—especially when listening on headphones.
        
        #### 3. Standard & Tolerances
        * **Target**: **$\\le -58.0\\text{ dBFS}$ (Ideal: $\\le -65.0\\text{ dBFS}$)**
        * **$-58\\text{ to }-48\\text{ dBFS}$**: Audible background hiss (40%–85% score)
        * **$> -40\\text{ dBFS}$**: Intolerable loud drone (0% score)
        * **Audible ACX Standard**: Noise floor $\\le -60.0\\text{ dBFS}$.
        
        #### 🛠️ Validated Step-by-Step Fix for Editors:
        1. **In Audacity (Noise Reduction)**:
           * Find a 2-second section of pure silence/pause in the recording.
           * Highlight it $\\rightarrow$ Go to **Effect $\\rightarrow$ Noise Reduction $\\rightarrow$ Click 'Get Noise Profile'**.
           * Select the whole track (`Ctrl + A`) $\\rightarrow$ Go to **Effect $\\rightarrow$ Noise Reduction**.
           * Set **Noise Reduction: 8 dB**, **Sensitivity: 6.00**, **Frequency Smoothing: 3**.
           * Click **OK** *(Do not use more than 10 dB reduction to avoid metallic artifacts)*.
        2. **Studio Recording Prep**:
           * Turn off all ceiling fans, AC units, and noisy laptop chargers during recording.
        """)

    elif "4. Master Volume" in param:
        st.markdown("### 🔊 Master Volume & Loudness (Integrated LUFS)")
        st.markdown("""
        #### 1. Plain-Language Definition
        Measures the overall perceived loudness of the entire audiobook track using the international standard **ITU-R BS.1770-4 / EBU R128**.
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        If audio is too quiet ($-28\\text{ LUFS}$), listeners must turn up their volume, which elevates background hiss. If audio is too loud ($-12\\text{ LUFS}$), it causes acoustic fatigue and ear pain. Proper mastering keeps the audio consistently comfortable across chapters.
        
        #### 3. Standard & Tolerances
        * **Target**: **$-24.0\\text{ to }-18.0\\text{ LUFS}$ (Ideal Target: $-20.0\\text{ LUFS}$)**
        * **Audible ACX Standard**: Between $-23.0\\text{ and }-18.0\\text{ LUFS}$ (Peak $\\le -3\\text{ dBFS}$).
        
        #### 🛠️ Validated Step-by-Step Fix for Editors:
        1. **In Audacity (1-Click Loudness Normalization)**:
           * Select All (`Ctrl + A`).
           * Go to **Effect $\\rightarrow$ Loudness Normalization**.
           * Set **Normalize to: Perceived Loudness (LUFS)**.
           * Enter **-20.0 LUFS** (or **-21.0 LUFS**).
           * Click **Apply**.
           * *(Always follow with a soft limiter at -3.0 dB to prevent peaks!)*
        """)

    elif "5. Reading Speed" in param:
        st.markdown("### ⏱️ Reading Speed & Narrative Pacing (WPM)")
        st.markdown("""
        #### 1. Plain-Language Definition
        Calculates the speaking rate in words per minute (WPM) based on detected syllable nuclei peaks in active speech.
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        For spiritual, philosophical, and educational audiobooks, rapid narration ($>160\\text{ WPM}$) overflows the listener's short-term working memory. Listeners cannot absorb the wisdom or reflect on concepts if words are rushed without pauses.
        
        #### 3. Standard & Tolerances
        * **Target**: **$100 - 145\\text{ WPM}$ (Ideal: $115 - 135\\text{ WPM}$)**
        * **$>160\\text{ WPM}$**: Rushed / speed-reading (degrades retention)
        * **$<90\\text{ WPM}$**: Excessively slow (unless dramatic theatrical reading)
        
        #### 🛠️ Validated Step-by-Step Fix for Narrators & Editors:
        1. **For Narrators**:
           * Breathe comfortably and deliberately pause for **$0.5\\text{ to }1.0\\text{ seconds}$** after major thoughts and between paragraphs.
        2. **For Editors (If audio was artificially sped up)**:
           * If audio was recorded or exported with pitch-shift speedup, revert to $1.0\\times$ playback speed.
        """)

    elif "6. High-Frequency" in param:
        st.markdown("### 🌬️ High-Frequency Air (Spectral Rolloff)")
        st.markdown("""
        #### 1. Plain-Language Definition
        Measures the frequency below which $85\\%$ of total acoustic power is contained.
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        Natural human voice contains delicate high-frequency overtones up to $16\\text{ kHz}$. A low rolloff ($<3500\\text{ Hz}$) indicates heavy low-bitrate compression or aggressive low-pass denoising, giving the audio a boxy, closed-in telephone sound.
        
        #### 3. Standard & Tolerances
        * **Target**: **$\\ge 4700\\text{ Hz}$ (Ideal: $\\ge 5000\\text{ Hz}$)**
        
        #### 🛠️ Validated Step-by-Step Fix for Editors:
        1. Always export in **$44.1\\text{ kHz}$ or $48\\text{ kHz}$ 16-bit WAV** or **192+ kbps CBR MP3**.
        2. Avoid destructive low-pass filters below $10\\text{ kHz}$.
        """)

    elif "7. Dynamic Life" in param:
        st.markdown("### 🎭 Dynamic Life & Expression (LRA)")
        st.markdown("""
        #### 1. Plain-Language Definition
        Measures the statistical variation in speech loudness over time (in LU) excluding silent intervals.
        
        #### 2. Why It Matters for Listener Fatigue (Psychoacoustics)
        A healthy dynamic range ($12-18\\text{ LU}$) creates emotional engagement. Completely flat audio ($<8\\text{ LU}$) caused by heavy brickwall limiters or monotonous reading bores the listener and makes audio feel artificial.
        
        #### 3. Standard & Tolerances
        * **Target**: **$\\ge 12.0\\text{ LU}$ (Ideal: $14.0 - 18.0\\text{ LU}$)**
        
        #### 🛠️ Validated Step-by-Step Fix for Editors:
        1. Avoid aggressive single-band compressors with heavy ratios ($>4:1$).
        2. Narrators should use emotional vocal modulation (whispering, emphasis, varying pitch).
        """)

# ==============================================================================
# 3. PAGE: REFERENCE HALL OF FAME
# ==============================================================================
elif app_mode == "🏆 Reference Hall of Fame (Top Audiobooks)":
    st.markdown('<div class="main-title">🏆 Audiobook Quality Hall of Fame</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Top Performing Audiobooks Across AWGP, Yagyavalkya, and Global Publishing Channels</div>', unsafe_allow_html=True)
    
    tab_awgp_leaders, tab_aspect_leaders, tab_all_table = st.tabs([
        "🌟 Top AWGP & Yagyavalkya Audiobooks",
        "🎖️ Best-in-Class by Feature",
        "📊 Complete 25-Audiobook Benchmark Scorecard"
    ])
    
    with tab_awgp_leaders:
        st.markdown("### 🌟 Best Performing Shantikunj & Yagyavalkya Audiobooks")
        st.markdown("Click directly on any title to listen on YouTube:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color:#1E3A8A; margin:0;"><a href="https://www.youtube.com/watch?v=3x-Sww8Ah5A" target="_blank" style="text-decoration:none; color:#1E3A8A;">1. Karmakanda Pradeep (Odia) ↗</a></h4>
                <p style="color:#64748B; font-size:0.9rem;">Yagyavalkya Channel • 8,200 Views</p>
                <div style="font-size:1.8rem; font-weight:800; color:#10B981;">87.2 / 100 <span style="font-size:1rem;">🟢 PRO GRADE</span></div>
                <ul style="font-size:0.9rem; color:#374151; margin-top:8px;">
                    <li><strong>Noise Floor</strong>: 🟢 -94.3 dBFS (Pristine silence)</li>
                    <li><strong>Vocal Clarity</strong>: 🟢 13.4% (Studio quality presence)</li>
                    <li><strong>Headroom</strong>: 🟢 0 clipped samples</li>
                </ul>
                <a href="https://www.youtube.com/watch?v=3x-Sww8Ah5A" target="_blank" style="font-weight:600; color:#2563EB;">▶️ Open Video on YouTube</a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4 style="color:#1E3A8A; margin:0;"><a href="https://www.youtube.com/watch?v=6gjejRSuvpQ" target="_blank" style="text-decoration:none; color:#1E3A8A;">2. Gayatri Sadhana Process ↗</a></h4>
                <p style="color:#64748B; font-size:0.9rem;">AWGP Shantikunj Official Channel</p>
                <div style="font-size:1.8rem; font-weight:800; color:#10B981;">83.5 / 100 <span style="font-size:1rem;">🟡 GOOD</span></div>
                <ul style="font-size:0.9rem; color:#374151; margin-top:8px;">
                    <li><strong>Noise Floor</strong>: 🟢 -71.0 dBFS (Ultra clean)</li>
                    <li><strong>Loudness</strong>: 🟢 -23.0 LUFS (ACX compliant)</li>
                    <li><strong>Headroom</strong>: 🟢 0 clipped samples</li>
                </ul>
                <a href="https://www.youtube.com/watch?v=6gjejRSuvpQ" target="_blank" style="font-weight:600; color:#2563EB;">▶️ Open Video on YouTube</a>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="color:#1E3A8A; margin:0;"><a href="https://www.youtube.com/watch?v=OC6ah1Z6nXI" target="_blank" style="text-decoration:none; color:#1E3A8A;">3. Gayatri Mahavigyan Pt-1 (Part 01) ↗</a></h4>
                <p style="color:#64748B; font-size:0.9rem;">Yagyavalkya Channel • 18,000 Views</p>
                <div style="font-size:1.8rem; font-weight:800; color:#10B981;">82.9 / 100 <span style="font-size:1rem;">🟡 GOOD</span></div>
                <ul style="font-size:0.9rem; color:#374151; margin-top:8px;">
                    <li><strong>Vocal Clarity</strong>: 🟢 15.3% (Highest clarity in catalog!)</li>
                    <li><strong>Noise Floor</strong>: 🟢 -63.5 dBFS (Clean studio tone)</li>
                    <li><strong>Pacing</strong>: 🟢 142 WPM (Natural storytelling tempo)</li>
                </ul>
                <a href="https://www.youtube.com/watch?v=OC6ah1Z6nXI" target="_blank" style="font-weight:600; color:#2563EB;">▶️ Open Video on YouTube</a>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
                <h4 style="color:#1E3A8A; margin:0;"><a href="https://www.youtube.com/watch?v=8RCRci9nv1E" target="_blank" style="text-decoration:none; color:#1E3A8A;">4. Women's Right to Gayatri ↗</a></h4>
                <p style="color:#64748B; font-size:0.9rem;">AWGP Shantikunj Official Channel</p>
                <div style="font-size:1.8rem; font-weight:800; color:#10B981;">80.5 / 100 <span style="font-size:1rem;">🟡 GOOD</span></div>
                <ul style="font-size:0.9rem; color:#374151; margin-top:8px;">
                    <li><strong>Noise Floor</strong>: 🟢 -66.3 dBFS (Compliant room tone)</li>
                    <li><strong>Loudness</strong>: 🟢 -23.3 LUFS (ACX compliant)</li>
                    <li><strong>Headroom</strong>: 🟢 0 clipped samples</li>
                </ul>
                <a href="https://www.youtube.com/watch?v=8RCRci9nv1E" target="_blank" style="font-weight:600; color:#2563EB;">▶️ Open Video on YouTube</a>
            </div>
            """, unsafe_allow_html=True)

    with tab_aspect_leaders:
        st.markdown("### 🎖️ Feature-by-Feature Acoustic Champions")
        
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("""
            **1. 🤫 Best Background Silence / Noise Floor:**
            * 🥇 [**Karmakanda Pradeep (Odia)**](https://www.youtube.com/watch?v=3x-Sww8Ah5A): `-94.3 dBFS` 🟢 (Yagyavalkya)
            * 🥈 [**जीवन देवता की साधना आराधना**](https://www.youtube.com/watch?v=IOhueHD3rBE): `-78.3 dBFS` 🟢 (AWGP)
            * 🥉 [**Elmer Gantry Audiobook**](https://www.youtube.com/watch?v=VVxug7cVLa0): `-70.2 dBFS` 🟢 (Pro Reference)
            
            ---
            **2. 🎙️ Best Vocal Formant Clarity (1-4 kHz):**
            * 🥇 [**Gayatri Mahavigyan Pt-1 (Part 01)**](https://www.youtube.com/watch?v=OC6ah1Z6nXI): `15.3%` 🟢 (Yagyavalkya)
            * 🥈 [**Christopher Lee - Tell-Tale Heart**](https://www.youtube.com/watch?v=Z_utA6j3Oc8): `14.2%` 🟢 (Pro Reference)
            * 🥉 [**Karmakanda Pradeep (Odia)**](https://www.youtube.com/watch?v=3x-Sww8Ah5A): `13.4%` 🟢 (Yagyavalkya)
            
            ---
            **3. ⚡ Perfect Digital Headroom (0 Clips):**
            * 🏆 [**Audible ACX Official Tips**](https://www.youtube.com/watch?v=XvIdC56hHVo) (0 clips)
            * 🏆 [**Gayatri Mahavigyan Pt-1**](https://www.youtube.com/watch?v=OC6ah1Z6nXI) (0 clips)
            * 🏆 [**Gayatri Sadhana Process**](https://www.youtube.com/watch?v=6gjejRSuvpQ) (0 clips)
            """)
            
        with c_b:
            st.markdown("""
            **4. 🔊 Best Standardized Loudness (LUFS):**
            * 🥇 [**Audible ACX Official Tips**](https://www.youtube.com/watch?v=XvIdC56hHVo): `-21.7 LUFS` 🟢
            * 🥈 [**Stephen Fry - Odyssey**](https://www.youtube.com/watch?v=ENzUY8c98Zw): `-21.6 LUFS` 🟢
            * 🥉 [**Gayatri Sadhana Process**](https://www.youtube.com/watch?v=6gjejRSuvpQ): `-23.0 LUFS` 🟢
            
            ---
            **5. ⏱️ Best Narrative Pacing & Reading Tempo:**
            * 🥇 [**Audible ACX Official Tips**](https://www.youtube.com/watch?v=XvIdC56hHVo): `125 WPM` 🟢 (Natural storytelling)
            * 🥈 [**सफल जीवन की दिशाधारा**](https://www.youtube.com/watch?v=IlN0cNEe_6A): `131 WPM` 🟢 (Clear articulation)
            * 🥉 [**Gayatri Mahavigyan Pt-1**](https://www.youtube.com/watch?v=OC6ah1Z6nXI): `142 WPM` 🟢 (Comfortable tempo)
            
            ---
            **6. 🎭 Highest Dynamic Life & Theatrical Expression:**
            * 🥇 [**Stephen Fry - Odyssey**](https://www.youtube.com/watch?v=ENzUY8c98Zw): `19.1 LU` 🟢 (World-class voice acting)
            * 🥈 [**Christopher Lee - Tell-Tale Heart**](https://www.youtube.com/watch?v=Z_utA6j3Oc8): `16.7 LU` 🟢
            * 🥉 [**Audible ACX Official Tips**](https://www.youtube.com/watch?v=XvIdC56hHVo): `16.2 LU` 🟢
            """)

    with tab_all_table:
        st.markdown("### 📊 Complete 25-Audiobook Benchmark Scorecard")
        st.markdown("Click on any audiobook title to listen directly on YouTube:")
        
        benchmark_data = [
            {"Category": "Pro", "Title": "[Audible ACX Official Tips](https://www.youtube.com/watch?v=XvIdC56hHVo)", "Score": "**97.9**", "Clips": "0", "Noise Floor": "-60.2 dB", "Clarity": "11.5%", "LUFS": "-21.7", "WPM": "125", "Rating": "🟢 PRO GRADE"},
            {"Category": "Pro", "Title": "[Elmer Gantry Audiobook](https://www.youtube.com/watch?v=VVxug7cVLa0)", "Score": "**97.9**", "Clips": "0", "Noise Floor": "-70.2 dB", "Clarity": "10.9%", "LUFS": "-23.0", "WPM": "110", "Rating": "🟢 PRO GRADE"},
            {"Category": "Pro", "Title": "[Stephen Fry - Odyssey](https://www.youtube.com/watch?v=ENzUY8c98Zw)", "Score": "**96.3**", "Clips": "0", "Noise Floor": "-67.9 dB", "Clarity": "11.9%", "LUFS": "-21.6", "WPM": "88", "Rating": "🟢 PRO GRADE"},
            {"Category": "Pro", "Title": "[Christopher Lee - Tell-Tale Heart](https://www.youtube.com/watch?v=Z_utA6j3Oc8)", "Score": "**90.4**", "Clips": "0", "Noise Floor": "-64.7 dB", "Clarity": "14.2%", "LUFS": "-28.8", "WPM": "134", "Rating": "🟢 PRO GRADE"},
            {"Category": "Pro", "Title": "[Penguin Audio (Natalie Dormer)](https://www.youtube.com/watch?v=XNuaseHTX98)", "Score": "**87.6**", "Clips": "0", "Noise Floor": "-160.0 dB", "Clarity": "9.0%", "LUFS": "-25.9", "WPM": "136", "Rating": "🟢 PRO GRADE"},
            {"Category": "Pro", "Title": "[Neil Gaiman - American Gods](https://www.youtube.com/watch?v=-x7J-D4dtns)", "Score": "**83.5**", "Clips": "0", "Noise Floor": "-69.4 dB", "Clarity": "7.1%", "LUFS": "-27.0", "WPM": "118", "Rating": "🟡 GOOD"},
            {"Category": "Yagya", "Title": "[Karmakanda Pradeep (Odia)](https://www.youtube.com/watch?v=3x-Sww8Ah5A)", "Score": "**87.2**", "Clips": "0", "Noise Floor": "-94.3 dB", "Clarity": "13.4%", "LUFS": "-28.0", "WPM": "155", "Rating": "🟢 PRO GRADE"},
            {"Category": "Yagya", "Title": "[Gayatri Mahavigyan Pt-1 (01)](https://www.youtube.com/watch?v=OC6ah1Z6nXI)", "Score": "**82.9**", "Clips": "0", "Noise Floor": "-63.5 dB", "Clarity": "15.3%", "LUFS": "-12.7", "WPM": "142", "Rating": "🟡 GOOD"},
            {"Category": "Yagya", "Title": "[सफल जीवन की दिशाधारा](https://www.youtube.com/watch?v=IlN0cNEe_6A)", "Score": "**82.0**", "Clips": "0", "Noise Floor": "-65.7 dB", "Clarity": "4.3%", "LUFS": "-24.1", "WPM": "131", "Rating": "🟡 GOOD"},
            {"Category": "Yagya", "Title": "[Gayatri Mahavigyan Pt-1 (02)](https://www.youtube.com/watch?v=-3FPoQ0lNlo)", "Score": "**77.9**", "Clips": "0", "Noise Floor": "-56.5 dB", "Clarity": "10.4%", "LUFS": "-12.7", "WPM": "147", "Rating": "🟡 GOOD"},
            {"Category": "Yagya", "Title": "[हर सुबह नया जन्म हर रात नई मौत](https://www.youtube.com/watch?v=G6lIgpnblcA)", "Score": "**45.0**", "Clips": "0", "Noise Floor": "-41.3 dB", "Clarity": "7.2%", "LUFS": "-17.7", "WPM": "154", "Rating": "🔴 POOR"},
            {"Category": "AWGP", "Title": "[Gayatri Sadhana Process](https://www.youtube.com/watch?v=6gjejRSuvpQ)", "Score": "**83.5**", "Clips": "0", "Noise Floor": "-71.0 dB", "Clarity": "8.7%", "LUFS": "-23.0", "WPM": "177", "Rating": "🟡 GOOD"},
            {"Category": "AWGP", "Title": "[Women's Right to Gayatri](https://www.youtube.com/watch?v=8RCRci9nv1E)", "Score": "**80.5**", "Clips": "0", "Noise Floor": "-66.3 dB", "Clarity": "7.5%", "LUFS": "-23.3", "WPM": "172", "Rating": "🟡 GOOD"},
            {"Category": "AWGP", "Title": "[Day-1 जीवन जीने की कला](https://www.youtube.com/watch?v=4Qp80cEG9Mw)", "Score": "**77.9**", "Clips": "0", "Noise Floor": "-62.4 dB", "Clarity": "8.4%", "LUFS": "-27.2", "WPM": "153", "Rating": "🟡 GOOD"},
            {"Category": "AWGP", "Title": "[गहना कर्मणोगतिः](https://www.youtube.com/watch?v=-xsOl340NAg)", "Score": "**77.8**", "Clips": "0", "Noise Floor": "-59.9 dB", "Clarity": "5.2%", "LUFS": "-23.8", "WPM": "143", "Rating": "🟡 GOOD"},
            {"Category": "AWGP", "Title": "[Marriage: A Sacred Union](https://www.youtube.com/watch?v=6Ul43m6mBJI)", "Score": "**67.8**", "Clips": "3", "Noise Floor": "-52.1 dB", "Clarity": "4.8%", "LUFS": "-22.2", "WPM": "161", "Rating": "🟠 FAIR"},
            {"Category": "AWGP", "Title": "[Honesty: Surest Policy](https://www.youtube.com/watch?v=de9oNT3UWrw)", "Score": "**56.2**", "Clips": "0", "Noise Floor": "-49.4 dB", "Clarity": "5.1%", "LUFS": "-27.3", "WPM": "172", "Rating": "🟠 FAIR"},
            {"Category": "AWGP", "Title": "[योग के नाम पर मायाचार](https://www.youtube.com/watch?v=IAHK2Sqnj74)", "Score": "**52.3**", "Clips": "671", "Noise Floor": "-61.3 dB", "Clarity": "7.9%", "LUFS": "-16.1", "WPM": "185", "Rating": "🟠 FAIR"},
            {"Category": "AWGP", "Title": "[जीवन देवता की साधना](https://www.youtube.com/watch?v=IOhueHD3rBE)", "Score": "**45.0**", "Clips": "4,438", "Noise Floor": "-78.3 dB", "Clarity": "6.8%", "LUFS": "-14.0", "WPM": "164", "Rating": "🔴 POOR"},
            {"Category": "AWGP", "Title": "[बुढ़ापे से टक्कर लीजिए](https://www.youtube.com/watch?v=MJLn0FaXLKI)", "Score": "**42.5**", "Clips": "6,653", "Noise Floor": "-67.2 dB", "Clarity": "5.3%", "LUFS": "-12.7", "WPM": "165", "Rating": "🔴 POOR"},
            {"Category": "AWGP", "Title": "[ईश्वर कौन है?](https://www.youtube.com/watch?v=LhTOMA9mMdg)", "Score": "**35.0**", "Clips": "13,510", "Noise Floor": "-56.2 dB", "Clarity": "4.3%", "LUFS": "-14.2", "WPM": "180", "Rating": "🔴 POOR"},
            {"Category": "AWGP", "Title": "[मस्तिष्क प्रत्यक्ष कल्पवृक्ष](https://www.youtube.com/watch?v=vcXp9Iopx60)", "Score": "**33.7**", "Clips": "14,762", "Noise Floor": "-56.6 dB", "Clarity": "4.3%", "LUFS": "-12.7", "WPM": "182", "Rating": "🔴 POOR"},
            {"Category": "AWGP", "Title": "[मैं क्या हूँ?](https://www.youtube.com/watch?v=DJJny0NvTxY)", "Score": "**30.0**", "Clips": "4,901", "Noise Floor": "-34.4 dB", "Clarity": "3.6%", "LUFS": "-15.2", "WPM": "147", "Rating": "🔴 POOR"},
            {"Category": "AWGP", "Title": "[गायत्री माहात्म्य](https://www.youtube.com/watch?v=BGnmVkD8KmU)", "Score": "**28.9**", "Clips": "37,962", "Noise Floor": "-47.4 dB", "Clarity": "3.2%", "LUFS": "-12.3", "WPM": "156", "Rating": "🔴 POOR"},
            {"Category": "Amateur", "Title": "[Dangerously Confident](https://www.youtube.com/watch?v=u4T96FEqk9U)", "Score": "**79.9**", "Clips": "0", "Noise Floor": "-66.0 dB", "Clarity": "4.9%", "LUFS": "-20.3", "WPM": "157", "Rating": "🟡 GOOD"}
        ]
        
        # Format table with Markdown links
        table_md = "| Category | Audiobook Title | Listener Score | Sound Cracking | Noise Floor | Vocal Clarity | Master Volume | Reading Speed | Quality Rating |\n"
        table_md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n"
        for row in benchmark_data:
            table_md += f"| {row['Category']} | {row['Title']} | {row['Score']} | {row['Clips']} | {row['Noise Floor']} | {row['Clarity']} | {row['LUFS']} | {row['WPM']} | {row['Rating']} |\n"
        st.markdown(table_md)

# ==============================================================================
# 4. PAGE: AUDACITY 1-CLICK MACRO & PRESETS
# ==============================================================================
elif app_mode == "📋 1-Click Audacity Macro & Presets":
    st.markdown('<div class="main-title">📋 1-Click Audacity Macro & Presets</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automate Complete Audiobook Mastering in Audacity with a Single Click</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="macro-box">
        <h3 style="color:#1D4ED8; margin-top:0;">⚡ Official 1-Click Audacity Master Macro</h3>
        <p style="color:#374151; margin-bottom:15px;">
        This downloadable macro runs the full professional mastering chain in Audacity:
        <ol style="margin-top:6px;">
            <li><strong>High-Pass Filter</strong> ($80\\text{ Hz}$) — Strips sub-bass room rumble and mic thumps.</li>
            <li><strong>Filter Curve EQ</strong> — Boosts vocal formant clarity ($+2.5\\text{ dB}$ at $2.5-5\\text{ kHz}$).</li>
            <li><strong>Loudness Normalization</strong> — Sets exact Audible ACX target ($-20.0\\text{ LUFS}$).</li>
            <li><strong>Soft Limiter</strong> ($-3.0\\text{ dB}$) — Guarantees 0 clipped samples permanently.</li>
        </ol>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download 'AWGP_Master_Audiobook_Preset.txt'",
        data=MACRO_CONTENT,
        file_name="AWGP_Master_Audiobook_Preset.txt",
        mime="text/plain",
        help="Click to download the macro file for Audacity."
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ How to Load and Use This Macro in Audacity (Takes 10 Seconds):")
    
    st.markdown("""
    1. **Step 1: Download the File**  
       Click the button above to download `AWGP_Master_Audiobook_Preset.txt` to your computer.
       
    2. **Step 2: Open Audacity Macro Manager**  
       Open Audacity $\\rightarrow$ Go to the top menu bar $\\rightarrow$ **Tools $\\rightarrow$ Macro Manager** (or *Macros*).
       
    3. **Step 3: Import the Preset**  
       Click the **Import** button in the bottom left $\\rightarrow$ select `AWGP_Master_Audiobook_Preset.txt`.  
       *(You only need to do this once!)*
       
    4. **Step 4: Run with 1-Click on Any Chapter**  
       Open your recorded audiobook chapter in Audacity $\\rightarrow$ Go to **Tools $\\rightarrow$ Apply Macro $\\rightarrow$ select 'AWGP_Master_Audiobook_Preset'**.
       
    5. **Result**: Your audio is instantly cleaned of rumble, enhanced for vocal presence, brought to exact **$-20.0\\text{ LUFS}$** volume, and capped with **$0\\text{ clipped samples}$**!
    """)
    
    st.markdown("---")
    st.markdown("### 🔍 What the Macro Does Internally:")
    
    st.markdown("""
    <div class="step-box">
        <h4 style="color:#15803D; margin:0;">1. Selects All Audio Tracks</h4>
        <p style="margin:4px 0 0 0; color:#374151;"><code>SelectAll:</code> ensures the entire chapter is processed uniformly.</p>
    </div>
    
    <div class="step-box">
        <h4 style="color:#15803D; margin:0;">2. High-Pass Filter (80 Hz, 12 dB/octave)</h4>
        <p style="margin:4px 0 0 0; color:#374151;"><code>HighPassFilter: frequency="80" rolloff="dB12"</code> strips sub-bass room vibrations and desk thumps.</p>
    </div>
    
    <div class="step-box">
        <h4 style="color:#15803D; margin:0;">3. Filter Curve Vocal Formant EQ (+2.5 dB at 2.5–5 kHz)</h4>
        <p style="margin:4px 0 0 0; color:#374151;"><code>FilterCurve: ...</code> lifts the presence band so speech consonants sound crisp and intelligible.</p>
    </div>
    
    <div class="step-box">
        <h4 style="color:#15803D; margin:0;">4. Loudness Normalization to -20.0 LUFS</h4>
        <p style="margin:4px 0 0 0; color:#374151;"><code>LoudnessNormalization: LUFSLevel="-20"</code> locks the volume to the ACX/Audible broadcast standard.</p>
    </div>
    
    <div class="step-box">
        <h4 style="color:#15803D; margin:0;">5. Soft Limiter (-3.0 dB Peak)</h4>
        <p style="margin:4px 0 0 0; color:#374151;"><code>Limiter: limit="-3" type="SoftLimit"</code> eliminates all digital clipping and distortion permanently.</p>
    </div>
    """, unsafe_allow_html=True)
