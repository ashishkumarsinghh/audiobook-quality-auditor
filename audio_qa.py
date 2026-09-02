#!/usr/bin/env python3
"""
AWGP Audiobook Quality Auditor (CLI Tool for Editors)
Usage:
    python audio_qa.py <audio_file>
    python audio_qa.py --all
"""

import os
import sys
import argparse
import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.signal import find_peaks, butter, filtfilt
from scoring import calculate_weighted_quality_score

def analyze_audio_file(file_path, duration=300):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
        
    print(f"🔍 Analyzing audio quality for: {os.path.basename(file_path)} ...")
    y, sr = librosa.load(file_path, sr=None, duration=duration)
    
    # 1. Digital Clipping
    clip_count = int(np.sum(np.abs(y) >= 0.999))
    peak_db = float(20 * np.log10(np.max(np.abs(y)) + 1e-9))
    
    # 2. Overall Volume (LUFS)
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(y))
    
    # 3. Dynamic Range (LRA)
    frame_len_lra = int(sr * 0.4)
    hop_len_lra = int(sr * 0.1)
    st_rms = librosa.feature.rms(y=y, frame_length=frame_len_lra, hop_length=hop_len_lra)[0]
    st_db = 20 * np.log10(np.clip(st_rms, 1e-8, None))
    speech_st_db = st_db[st_db > (np.max(st_db) - 30)]
    lra = float(np.percentile(speech_st_db, 95) - np.percentile(speech_st_db, 10)) if len(speech_st_db) > 0 else 0.0
    
    # 4. Noise Floor (5th percentile short-term RMS in dBFS)
    frame_len_noise = int(sr * 0.1)
    hop_len_noise = int(sr * 0.05)
    rms_noise = librosa.feature.rms(y=y, frame_length=frame_len_noise, hop_length=hop_len_noise)[0]
    rms_noise_db = 20 * np.log10(np.clip(rms_noise, 1e-8, None))
    noise_floor = float(np.percentile(rms_noise_db, 5))
    
    # 5. Vocal Clarity (1-4 kHz) & Spectral Rolloff
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total_energy = np.sum(S) + 1e-10
    mid_mask = (freqs >= 1000) & (freqs <= 4000)
    vocal_clarity_pct = float((np.sum(S[mid_mask, :]) / total_energy) * 100)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_rolloff = float(np.mean(rolloff))
    
    # 6. Pacing (WPM)
    nyq = 0.5 * sr
    b, a = butter(5, [300 / nyq, 3300 / nyq], btype='band')
    y_filtered = filtfilt(b, a, y)
    frame_len_wpm = int(sr * 0.05)
    hop_len_wpm = int(sr * 0.01)
    rms_wpm = librosa.feature.rms(y=y_filtered, frame_length=frame_len_wpm, hop_len_wpm)[0]
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
    
    print("\n" + "="*68)
    print(f"📊 WEIGHTED LISTENER SCORE: {score:.1f}/100")
    if score >= 85.0:
        print("🏆 VERDICT: 🟢 PRO STUDIO GRADE (ACX Compliant)")
    elif score >= 68.0:
        print("⚠️ VERDICT: 🟡 GOOD (Minor Editing Tweaks Needed)")
    elif score >= 50.0:
        print("🟠 VERDICT: 🟠 FAIR (Needs Noticeable Audio Cleanup)")
    else:
        print("❌ VERDICT: 🔴 POOR (Needs Major Re-Recording / Remastering)")
    print("="*68)
    
    def status_line(name, val_str, subscore, ideal):
        if subscore >= 0.85:
            icon = "🟢 EXCELLENT"
        elif subscore >= 0.50:
            icon = "🟡 ACCEPTABLE"
        else:
            icon = "🔴 FIX NEEDED"
        return f"[{icon:<12}] {name:<28}: {val_str:<14} (Ideal: {ideal})"
        
    print(status_line("Sound Cracking (25%)", f"{clip_count} clips", subscores["clips"], "0 clips"))
    print(status_line("Vocal Formant Clarity (20%)", f"{vocal_clarity_pct:.1f}%", subscores["clarity"], ">= 9.5%"))
    print(status_line("Background Noise (20%)", f"{noise_floor:.1f} dBFS", subscores["noise_floor"], "<= -58 dBFS"))
    print(status_line("Master Volume (15%)", f"{lufs:.1f} LUFS", subscores["lufs"], "-23 to -18 LUFS"))
    print(status_line("Reading Speed (10%)", f"{wpm:.0f} WPM", subscores["wpm"], "100 - 145 WPM"))
    print(status_line("High-Frequency Air (5%)", f"{mean_rolloff:.0f} Hz", subscores["rolloff"], ">= 4700 Hz"))
    print(status_line("Dynamic Expression (5%)", f"{lra:.1f} LU", subscores["lra"], ">= 13.0 LU"))
    
    print("\n" + "-"*68)
    print("🛠️ ACTIONABLE FIXES FOR EDITOR:")
    fixes = []
    if subscores["clips"] < 0.85:
        fixes.append("• [CRACKING] Audio has digital clipping. In Audacity: Apply Limiter at -3.0 dB, or reduce recording mic gain.")
    if subscores["clarity"] < 0.85:
        fixes.append("• [MUFFLED VOICE] Voice lacks crisp presence. Speak 4-6 inches from mic with pop filter; in Audacity, boost +2.5 dB at 3-5 kHz.")
    if subscores["noise_floor"] < 0.85:
        fixes.append("• [HISS/HUM] Background noise is audible. Turn off fans/AC, or apply 6-8 dB Noise Reduction in Audacity.")
    if subscores["lufs"] < 0.85:
        fixes.append(f"• [VOLUME] Volume is {'too loud' if lufs > -18 else 'too quiet'}. In Audacity: Effect -> Loudness Normalization to -20.0 LUFS.")
    if subscores["wpm"] < 0.85:
        fixes.append(f"• [SPEED] Narration is {'too fast/rushed' if wpm > 145 else 'too slow'}. Aim for 110-135 WPM with natural sentence pauses.")
    if subscores["rolloff"] < 0.85 or subscores["lra"] < 0.85:
        fixes.append("• [TIMBRE/DYNAMICS] Export at 44.1kHz WAV, avoid heavy single-band compressors, read with expressive emotion.")
        
    if not fixes:
        print("✨ No fixes needed! This recording meets international pro audiobook standards.")
    else:
        for f in fixes:
            print(f)
    print("-"*68 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWGP Audiobook Quality Auditor")
    parser.add_argument("audio_file", nargs="?", help="Path to audio file (mp3, wav, m4a, flac)")
    parser.add_argument("--all", action="store_true", help="Run benchmark across all dataset audiobooks")
    args = parser.parse_args()
    
    if args.all or not args.audio_file:
        from generate_report_v2 import run_full_report
        run_full_report()
    else:
        analyze_audio_file(args.audio_file)
