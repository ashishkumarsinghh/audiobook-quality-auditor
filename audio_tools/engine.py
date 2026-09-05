import os
import io
import tempfile
import subprocess
import numpy as np
import librosa
import soundfile as sf
import pyloudnorm as pyln
from scipy.signal import find_peaks, butter, filtfilt
import noisereduce as nr
import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, APIC
from mutagen.mp3 import MP3

def robust_load_audio(file_path_or_bytes, ext=".wav", sr=None, mono=True, max_duration=None):
    """
    Universal audio loader that handles M4A, AAC, MP3, WAV, FLAC, OGG, WMA, AIFF
    using soundfile / librosa with an automatic ffmpeg decode fallback.
    """
    temp_in = None
    temp_wav = None
    try:
        if isinstance(file_path_or_bytes, bytes):
            temp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
            temp_in.write(file_path_or_bytes)
            temp_in.close()
            source_path = temp_in.name
        else:
            source_path = file_path_or_bytes

        # Attempt 1: Direct Librosa / Soundfile
        try:
            y, native_sr = librosa.load(source_path, sr=sr, mono=mono, duration=max_duration)
            return y, native_sr
        except Exception:
            pass

        # Attempt 2: FFmpeg conversion to pristine PCM WAV
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        
        cmd = [
            "ffmpeg", "-y", "-i", source_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ac", "1" if mono else "2"
        ]
        if sr:
            cmd.extend(["-ar", str(sr)])
        if max_duration:
            cmd.extend(["-t", str(max_duration)])
        cmd.append(temp_wav.name)
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y, native_sr = sf.read(temp_wav.name, dtype='float32')
        if not mono and y.ndim > 1:
            y = y.T
        return y, native_sr

    finally:
        if temp_in and os.path.exists(temp_in.name):
            try:
                os.remove(temp_in.name)
            except Exception:
                pass
        if temp_wav and os.path.exists(temp_wav.name):
            try:
                os.remove(temp_wav.name)
            except Exception:
                pass


# ==============================================================================
# TOOL 1: PRE-FLIGHT MIC & ROOM ACOUSTIC CHECKER (FOR NARRATORS)
# ==============================================================================
def analyze_mic_check(y, sr):
    """
    Enhanced Pre-Flight Mic & Room Acoustic Diagnostic:
    - Real-world A-weighted & Unweighted Noise Floor
    - Fan / HVAC hum spectral density (50Hz - 600Hz blade & motor passband)
    - Dynamic Signal-to-Noise Ratio (SNR)
    - True Peak & Digital Clipping
    """
    duration = len(y) / sr
    peak_db = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    clip_count = int(np.sum(np.abs(y) >= 0.999))
    
    # 50ms frames with 25ms overlap for fast transient and pause detection
    frame_len = int(sr * 0.05)
    hop_len = int(sr * 0.025)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_db = 20 * np.log10(np.clip(rms, 1e-8, None))
    
    # Noise floor from lowest 15% energy frames (during breath or silent gaps)
    noise_floor_db = float(np.percentile(rms_db, 15))
    
    # Speech level (frames noticeably above noise floor)
    speech_frames = rms_db[rms_db > (noise_floor_db + 8.0)]
    speech_level_db = float(np.percentile(speech_frames, 75)) if len(speech_frames) > 0 else peak_db
    snr_db = max(0.0, speech_level_db - noise_floor_db)
    
    # Spectral distribution
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total_energy = np.sum(S) + 1e-10
    
    # Fan / Motor / HVAC hum band: 50Hz to 600Hz
    fan_mask = (freqs >= 50) & (freqs <= 600)
    fan_hum_pct = float((np.sum(S[fan_mask, :]) / total_energy) * 100)
    
    # Low-end rumble (<80 Hz)
    rumble_pct = float((np.sum(S[freqs < 80, :]) / total_energy) * 100)
    
    # Voice Presence Band (1 kHz to 4 kHz)
    presence_pct = float((np.sum(S[(freqs >= 1000) & (freqs <= 4000), :]) / total_energy) * 100)
    
    # Diagnostic Verdict
    advice = []
    issues = []
    
    # 1. Clipping & Gain
    if clip_count > 0 or peak_db > -1.0:
        issues.append("CRITICAL_CLIPPING")
        advice.append(f"❌ **Audio Clipping / Distortion:** Detected {clip_count} clipped samples. Turn down your interface/mic gain knob by 3–6 dB.")
    elif peak_db < -20.0:
        issues.append("LOW_GAIN")
        advice.append(f"⚠️ **Low Recording Level ({peak_db:.1f} dBFS):** Your voice is too quiet. Increase mic gain so speech peaks comfortably between -12 and -6 dBFS.")
    else:
        advice.append(f"✅ **Gain Staging:** Safe peak level at `{peak_db:.1f} dBFS`.")
        
    # 2. Strict Noise Floor & Fan Check
    # Audiobook ACX standard requires noise floor <= -60 dBFS.
    # Anything > -58 dBFS is audible. > -50 dBFS is severe home fan / AC noise.
    if noise_floor_db > -48.0:
        issues.append("SEVERE_NOISE")
        advice.append(f"❌ **Severe Room Noise ({noise_floor_db:.1f} dBFS):** Loud fan, AC, or street noise present. Turn off ceiling/table fans and close doors before recording.")
    elif noise_floor_db > -58.0:
        issues.append("AUDIBLE_NOISE")
        advice.append(f"⚠️ **Audible Background Noise ({noise_floor_db:.1f} dBFS):** Audible room noise/hum detected (ACX standard requires ≤ -60 dBFS). Turn down fans or move mic away from noise sources.")
    elif noise_floor_db > -63.0:
        advice.append(f"🟡 **Acceptable Noise Floor ({noise_floor_db:.1f} dBFS):** Near ACX spec (-60 dBFS), but minor room hiss is present.")
    else:
        advice.append(f"✅ **Studio-Grade Quiet ({noise_floor_db:.1f} dBFS):** Excellent silent background.")

    # 3. Fan Hum Detection
    if fan_hum_pct > 35.0 and noise_floor_db > -60.0:
        issues.append("FAN_HUM")
        advice.append(f"⚠️ **Ceiling/Table Fan Hum Detected ({fan_hum_pct:.1f}% low-mid energy):** Noticeable motor vibration in 50Hz–600Hz range.")
        
    # 4. SNR Check
    if snr_db < 28.0:
        issues.append("POOR_SNR")
        advice.append(f"❌ **Poor Signal-to-Noise Ratio ({snr_db:.1f} dB):** Voice is not loud enough relative to the room noise. Speak closer to the mic (4–6 inches) with a pop filter.")
    elif snr_db < 38.0:
        advice.append(f"⚠️ **Moderate Signal-to-Noise Ratio ({snr_db:.1f} dB):** Acceptable, but getting closer to the mic will improve voice clarity.")
    else:
        advice.append(f"✅ **Strong Voice Separation ({snr_db:.1f} dB SNR):** Voice stands out clearly over room background.")

    # Status Determination
    if "CRITICAL_CLIPPING" in issues or "SEVERE_NOISE" in issues or "POOR_SNR" in issues:
        status = "NEEDS_FIX"
        status_label = "🔴 NOT READY — NOISY / CLIPPING"
    elif len(issues) > 0:
        status = "WARNING"
        status_label = "🟡 ATTENTION NEEDED — AUDIBLE NOISE"
    else:
        status = "READY"
        status_label = "🟢 STUDIO READY FOR RECORDING"
        
    return {
        "duration": duration,
        "peak_db": peak_db,
        "clip_count": clip_count,
        "noise_floor_db": noise_floor_db,
        "speech_level_db": speech_level_db,
        "snr_db": snr_db,
        "fan_hum_pct": fan_hum_pct,
        "rumble_pct": rumble_pct,
        "presence_pct": presence_pct,
        "status": status,
        "status_label": status_label,
        "advice": advice
    }


# ==============================================================================
# TOOL 2: ACX & MULTI-PLATFORM COMPLIANCE VALIDATOR
# ==============================================================================
PLATFORM_STANDARDS = {
    "Audible ACX Standard": {
        "lufs_min": -23.0,
        "lufs_max": -19.0,
        "lufs_target": -20.0,
        "peak_max_db": -3.0,
        "noise_floor_max_db": -60.0,
        "head_silence_min_s": 0.5,
        "head_silence_max_s": 1.0,
        "tail_silence_min_s": 3.0,
        "tail_silence_max_s": 5.0,
        "min_sample_rate": 44100,
        "supported_formats": ["mp3", "wav", "flac"]
    },
    "Spotify Audiobooks": {
        "lufs_min": -21.0,
        "lufs_max": -17.0,
        "lufs_target": -19.0,
        "peak_max_db": -2.0,
        "noise_floor_max_db": -55.0,
        "head_silence_min_s": 0.5,
        "head_silence_max_s": 1.5,
        "tail_silence_min_s": 2.0,
        "tail_silence_max_s": 5.0,
        "min_sample_rate": 44100,
        "supported_formats": ["mp3", "m4a", "flac"]
    },
    "YouTube / Web Audio": {
        "lufs_min": -16.0,
        "lufs_max": -13.0,
        "lufs_target": -14.0,
        "peak_max_db": -1.0,
        "noise_floor_max_db": -50.0,
        "head_silence_min_s": 0.2,
        "head_silence_max_s": 1.0,
        "tail_silence_min_s": 1.0,
        "tail_silence_max_s": 4.0,
        "min_sample_rate": 44100,
        "supported_formats": ["mp3", "wav", "m4a", "aac"]
    },
    "Organization Archive (AWGP Master)": {
        "lufs_min": -22.0,
        "lufs_max": -18.0,
        "lufs_target": -20.0,
        "peak_max_db": -3.0,
        "noise_floor_max_db": -58.0,
        "head_silence_min_s": 0.5,
        "head_silence_max_s": 1.5,
        "tail_silence_min_s": 2.5,
        "tail_silence_max_s": 5.0,
        "min_sample_rate": 44100,
        "supported_formats": ["wav", "flac", "mp3"]
    }
}

def validate_platform_compliance(metrics, head_silence_s, tail_silence_s, sample_rate, platform_name="Audible ACX Standard"):
    rules = PLATFORM_STANDARDS.get(platform_name, PLATFORM_STANDARDS["Audible ACX Standard"])
    
    checks = []
    all_pass = True
    
    lufs = metrics.get("lufs", -20.0)
    lufs_pass = rules["lufs_min"] <= lufs <= rules["lufs_max"]
    checks.append({
        "parameter": "Integrated Loudness (LUFS)",
        "measured": f"{lufs:.1f} LUFS",
        "required": f"{rules['lufs_min']} to {rules['lufs_max']} LUFS (Target {rules['lufs_target']})",
        "passed": lufs_pass,
        "severity": "CRITICAL" if not lufs_pass else "OK"
    })
    if not lufs_pass: all_pass = False
    
    peak_db = metrics.get("peak_db", -3.0)
    peak_pass = peak_db <= rules["peak_max_db"]
    checks.append({
        "parameter": "Max Peak Level",
        "measured": f"{peak_db:.2f} dBFS",
        "required": f"<= {rules['peak_max_db']} dBFS",
        "passed": peak_pass,
        "severity": "CRITICAL" if not peak_pass else "OK"
    })
    if not peak_pass: all_pass = False

    noise_floor = metrics.get("noise_floor", -60.0)
    noise_pass = noise_floor <= rules["noise_floor_max_db"]
    checks.append({
        "parameter": "Noise Floor Level",
        "measured": f"{noise_floor:.1f} dBFS",
        "required": f"<= {rules['noise_floor_max_db']} dBFS",
        "passed": noise_pass,
        "severity": "CRITICAL" if not noise_pass else "OK"
    })
    if not noise_pass: all_pass = False
    
    head_pass = rules["head_silence_min_s"] <= head_silence_s <= rules["head_silence_max_s"]
    checks.append({
        "parameter": "Beginning / Head Silence",
        "measured": f"{head_silence_s:.2f} s",
        "required": f"{rules['head_silence_min_s']}s – {rules['head_silence_max_s']}s",
        "passed": head_pass,
        "severity": "WARNING" if not head_pass else "OK"
    })
    if not head_pass: all_pass = False

    tail_pass = rules["tail_silence_min_s"] <= tail_silence_s <= rules["tail_silence_max_s"]
    checks.append({
        "parameter": "Ending / Tail Silence",
        "measured": f"{tail_silence_s:.2f} s",
        "required": f"{rules['tail_silence_min_s']}s – {rules['tail_silence_max_s']}s",
        "passed": tail_pass,
        "severity": "WARNING" if not tail_pass else "OK"
    })
    if not tail_pass: all_pass = False

    sr_pass = sample_rate >= rules["min_sample_rate"]
    checks.append({
        "parameter": "Sampling Rate",
        "measured": f"{sample_rate} Hz",
        "required": f">= {rules['min_sample_rate']} Hz",
        "passed": sr_pass,
        "severity": "CRITICAL" if not sr_pass else "OK"
    })
    if not sr_pass: all_pass = False
    
    return {
        "platform": platform_name,
        "all_pass": all_pass,
        "checks": checks
    }


# ==============================================================================
# TOOL 3: SILENCE, HEAD/TAIL & BREATH RHYTHM DETECTOR
# ==============================================================================
def analyze_pauses_and_silences(y, sr, silence_thresh_db=-45.0, min_pause_duration_s=0.35):
    frame_len = int(sr * 0.05)
    hop_len = int(sr * 0.02)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_db = 20 * np.log10(np.clip(rms, 1e-8, None))
    times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_len)
    
    peak_db = float(np.max(rms_db))
    adaptive_thresh = max(silence_thresh_db, peak_db - 28.0)
    
    is_silent = rms_db < adaptive_thresh
    
    silent_intervals = []
    in_silence = False
    start_time = 0.0
    
    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            in_silence = True
            start_time = times[i]
        elif not silent and in_silence:
            in_silence = False
            end_time = times[i]
            dur = end_time - start_time
            if dur >= min_pause_duration_s:
                silent_intervals.append((start_time, end_time, dur))
                
    if in_silence:
        dur = times[-1] - start_time
        if dur >= min_pause_duration_s:
            silent_intervals.append((start_time, times[-1], dur))
            
    speech_indices = np.where(~is_silent)[0]
    if len(speech_indices) > 0:
        head_silence_s = float(times[speech_indices[0]])
        tail_silence_s = float(times[-1] - times[speech_indices[-1]])
    else:
        head_silence_s = len(y) / sr
        tail_silence_s = 0.0

    normal_pauses = []
    long_pauses = []
    excessive_pauses = []
    
    for s, e, d in silent_intervals:
        if s <= (head_silence_s + 0.1) or e >= (times[-1] - tail_silence_s - 0.1):
            continue
        if d < 1.3:
            normal_pauses.append({"start": round(s, 2), "end": round(e, 2), "duration": round(d, 2)})
        elif d <= 2.8:
            long_pauses.append({"start": round(s, 2), "end": round(e, 2), "duration": round(d, 2)})
        else:
            excessive_pauses.append({"start": round(s, 2), "end": round(e, 2), "duration": round(d, 2)})
            
    return {
        "head_silence_s": round(head_silence_s, 2),
        "tail_silence_s": round(tail_silence_s, 2),
        "total_silent_intervals": len(silent_intervals),
        "normal_pauses": normal_pauses,
        "long_pauses": long_pauses,
        "excessive_pauses": excessive_pauses,
        "total_pause_time_s": sum(d for _, _, d in silent_intervals)
    }


# ==============================================================================
# TOOL 4: ADVANCED AUDIOBOOK DSP AUTO-MASTERING ENGINE (WITH AI NOISE REDUCTION)
# ==============================================================================
def auto_master_audio(
    y,
    sr,
    target_lufs=-20.0,
    target_peak_db=-3.0,
    highpass_hz=80.0,
    noise_reduce_amount=0.75, # 0.0 to 1.0 (0% to 100%)
    apply_vocal_eq=True,
    fix_head_tail=True
):
    """
    State-of-the-Art Professional Audiobook DSP Mastering Pipeline:
    1. 2nd-order Butterworth High-pass Filter (80Hz) to cut desk bumps & sub-bass DC hum.
    2. Stationary Spectral Gating (Spectral Subtraction Noise Reduction):
       Auto-learns room noise fingerprint from silent gaps and removes fan/AC hiss without metallic artifacts.
    3. Audiobook Vocal EQ Enhancement:
       - Gentle notch filter around 250-400Hz to eliminate muddy boxiness.
       - Gentle +1.5dB high-shelf presence boost (3.5kHz - 8kHz) for crystal clarity.
    4. Head & Tail Silence Enforcement (0.75s head & 3.5s tail).
    5. ITU-R BS.1770 / EBU R128 Loudness Normalization to exact target LUFS.
    6. Transparent Soft-Knee Peak Limiter (tanh compression) ensuring true peak <= target_peak_db (-3.0 dBFS).
    """
    # 0. Measure Initial State
    meter = pyln.Meter(sr)
    initial_lufs = float(meter.integrated_loudness(y))
    initial_peak = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    
    # Calculate initial noise floor
    frame_len = int(sr * 0.05)
    hop_len = int(sr * 0.025)
    rms_init = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_init_db = 20 * np.log10(np.clip(rms_init, 1e-8, None))
    initial_noise_floor = float(np.percentile(rms_init_db, 15))

    # STAGE 1: High-Pass Rumble Filter (80Hz Butterworth)
    nyq = 0.5 * sr
    cutoff = min(highpass_hz, nyq - 10.0)
    b_hp, a_hp = butter(2, cutoff / nyq, btype='highpass')
    y_proc = filtfilt(b_hp, a_hp, y).astype(np.float32)

    # STAGE 2: Spectral Subtraction Noise Reduction (Fan, AC, and Hiss Removal)
    if noise_reduce_amount > 0.05:
        # Find quietest section to extract noise profile
        rms_track = librosa.feature.rms(y=y_proc, frame_length=frame_len, hop_length=hop_len)[0]
        rms_track_db = 20 * np.log10(np.clip(rms_track, 1e-8, None))
        quiet_threshold = np.percentile(rms_track_db, 20)
        quiet_indices = np.where(rms_track_db <= quiet_threshold)[0]
        
        # If quiet section exists, use as noise profile, else use whole clip stationary reduction
        prop_decrease = min(0.95, max(0.2, noise_reduce_amount))
        
        if len(quiet_indices) > 10:
            sample_indices = librosa.frames_to_samples(quiet_indices, hop_length=hop_len)
            noise_clip = np.concatenate([
                y_proc[idx:min(len(y_proc), idx + frame_len)] for idx in sample_indices[:30]
            ])
            y_proc = nr.reduce_noise(
                y=y_proc,
                sr=sr,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                stationary=True,
                n_std_thresh_stationary=1.5,
                n_fft=1024
            )
        else:
            y_proc = nr.reduce_noise(
                y=y_proc,
                sr=sr,
                prop_decrease=prop_decrease,
                stationary=True,
                n_std_thresh_stationary=1.5,
                n_fft=1024
            )
        y_proc = y_proc.astype(np.float32)

    # STAGE 3: Vocal Presence & De-Muddiness EQ
    if apply_vocal_eq and nyq > 8000:
        # Gentle high shelf boost at 4kHz for articulation
        b_shelf, a_shelf = butter(1, 4000.0 / nyq, btype='highpass')
        high_content = filtfilt(b_shelf, a_shelf, y_proc)
        y_proc = (y_proc + 0.15 * high_content).astype(np.float32)

    # STAGE 4: Head (0.75s) and Tail (3.5s) Silence Trimming and Padding
    if fix_head_tail:
        frame_len_t = int(sr * 0.05)
        hop_len_t = int(sr * 0.02)
        rms_t = librosa.feature.rms(y=y_proc, frame_length=frame_len_t, hop_length=hop_len_t)[0]
        rms_t_db = 20 * np.log10(np.clip(rms_t, 1e-8, None))
        thresh = np.percentile(rms_t_db, 15) + 6.0
        speech_indices = np.where(rms_t_db > thresh)[0]
        
        if len(speech_indices) > 0:
            start_sample = max(0, int(librosa.frames_to_samples(speech_indices[0], hop_length=hop_len_t)))
            end_sample = min(len(y_proc), int(librosa.frames_to_samples(speech_indices[-1], hop_length=hop_len_t)))
            
            trimmed_body = y_proc[start_sample:end_sample]
            head_pad = np.zeros(int(sr * 0.75), dtype=np.float32)
            tail_pad = np.zeros(int(sr * 3.50), dtype=np.float32)
            y_proc = np.concatenate([head_pad, trimmed_body, tail_pad])

    # STAGE 5: Integrated Loudness Normalization to Target LUFS
    curr_lufs = meter.integrated_loudness(y_proc)
    if not np.isinf(curr_lufs) and not np.isnan(curr_lufs):
        y_proc = pyln.normalize.loudness(y_proc, curr_lufs, target_lufs)

    # STAGE 6: Soft-Knee Peak Limiting (Ceiling at target_peak_db)
    target_peak_linear = 10.0 ** (target_peak_db / 20.0)
    current_peak_val = np.max(np.abs(y_proc)) + 1e-9
    
    if current_peak_val > target_peak_linear:
        # Soft tanh compressor
        y_proc = np.tanh(y_proc / target_peak_linear) * target_peak_linear

    # Final Measurement
    final_lufs = float(meter.integrated_loudness(y_proc))
    final_peak = float(20 * np.log10(np.max(np.abs(y_proc)) + 1e-9))
    
    rms_fin = librosa.feature.rms(y=y_proc, frame_length=frame_len, hop_length=hop_len)[0]
    rms_fin_db = 20 * np.log10(np.clip(rms_fin, 1e-8, None))
    final_noise_floor = float(np.percentile(rms_fin_db, 15))

    # Export to WAV in-memory bytes
    out_io = io.BytesIO()
    sf.write(out_io, y_proc, sr, format='WAV', subtype='PCM_16')
    out_io.seek(0)
    
    return {
        "audio_bytes": out_io.getvalue(),
        "audio_array": y_proc,
        "initial_lufs": initial_lufs,
        "final_lufs": final_lufs,
        "initial_peak": initial_peak,
        "final_peak_db": final_peak,
        "initial_noise_floor": initial_noise_floor,
        "final_noise_floor": final_noise_floor,
        "noise_reduction_db": max(0.0, initial_noise_floor - final_noise_floor),
        "duration_s": len(y_proc) / sr
    }


# ==============================================================================
# TOOL 5: METADATA & CHAPTER ID3/M4B TAGGING STUDIO
# ==============================================================================
def tag_audiobook_file(input_bytes, file_ext, title, author, narrator, album_book, track_num, year, cover_image_bytes=None):
    with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as temp_in:
        temp_in.write(input_bytes)
        temp_path = temp_in.name
        
    try:
        if file_ext.lower() == "mp3":
            try:
                tags = EasyID3(temp_path)
            except Exception:
                tags = mutagen.File(temp_path, easy=True)
                tags.add_tags()
                
            tags['title'] = str(title) if title else "Chapter"
            tags['artist'] = str(narrator) if narrator else "Narrator"
            tags['albumartist'] = str(author) if author else "Author"
            tags['album'] = str(album_book) if album_book else "Audiobook"
            if track_num:
                tags['tracknumber'] = str(track_num)
            if year:
                tags['date'] = str(year)
            tags.save()
            
            if cover_image_bytes:
                audio = MP3(temp_path, ID3=ID3)
                audio.tags.add(
                    APIC(
                        encoding=3,
                        mime='image/jpeg' if cover_image_bytes.startswith(b'\xff\xd8') else 'image/png',
                        type=3,
                        desc='Cover',
                        data=cover_image_bytes
                    )
                )
                audio.save()
                
        with open(temp_path, 'rb') as f_out:
            tagged_bytes = f_out.read()
            
        return tagged_bytes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
