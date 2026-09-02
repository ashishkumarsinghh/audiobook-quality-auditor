import os
import sys
import argparse
import librosa
import numpy as np
import pyloudnorm as pyln
from scipy.signal import find_peaks, butter, filtfilt
from scoring import calculate_weighted_quality_score

VIDEOS = {
    # 1. Pro Reference Standards
    "XvIdC56hHVo": ("Pro", "Audible ACX Official Tips"),
    "XNuaseHTX98": ("Pro", "Penguin Audio (Natalie Dormer)"),
    "-x7J-D4dtns": ("Pro", "Neil Gaiman - American Gods"),
    "VVxug7cVLa0": ("Pro", "Elmer Gantry Audiobook"),
    "ENzUY8c98Zw": ("Pro", "Stephen Fry - Odyssey"),
    "Z_utA6j3Oc8": ("Pro", "Christopher Lee - Tell-Tale Heart"),
    
    # 2. Yagyavalkya Newsletter (Top Audiobooks)
    "OC6ah1Z6nXI": ("Yagyavalkya", "Gayatri Mahavigyan Pt-1 (Part 01)"),
    "3x-Sww8Ah5A": ("Yagyavalkya", "Karmakanda Pradeep (Odia)"),
    "IlN0cNEe_6A": ("Yagyavalkya", "सफल जीवन की दिशाधारा (Part 01)"),
    "-3FPoQ0lNlo": ("Yagyavalkya", "Gayatri Mahavigyan Pt-1 (Part 02)"),
    "G6lIgpnblcA": ("Yagyavalkya", "हर सुबह नया जन्म हर रात नई मौत"),

    # 3. AWGP / Shantikunj Audiobooks
    "6Ul43m6mBJI": ("AWGP", "Marriage: A Sacred Union"),
    "de9oNT3UWrw": ("AWGP", "Honesty: Surest Policy for Progress"),
    "8RCRci9nv1E": ("AWGP", "Women's Right to Gayatri"),
    "6gjejRSuvpQ": ("AWGP", "Purpose & Process of Gayatri Sadhana"),
    "LhTOMA9mMdg": ("AWGP", "ईश्वर कौन है?"),
    "-xsOl340NAg": ("AWGP", "गहना कर्मणोगतिः"),
    "4Qp80cEG9Mw": ("AWGP", "Day-1 जीवन जीने की कला"),
    "MJLn0FaXLKI": ("AWGP", "बुढ़ापे से टक्कर लीजिए"),
    "DJJny0NvTxY": ("AWGP", "मैं क्या हूँ?"),
    "IOhueHD3rBE": ("AWGP", "जीवन देवता की साधना आराधना"),
    "BGnmVkD8KmU": ("AWGP", "गायत्री माहात्म्य"),
    "vcXp9Iopx60": ("AWGP", "मस्तिष्क प्रत्यक्ष कल्पवृक्ष"),
    "IAHK2Sqnj74": ("AWGP", "योग के नाम पर मायाचार"),
    
    # 4. Other YouTube Amateur
    "u4T96FEqk9U": ("Amateur", "Dangerously Confident")
}

def analyze_audio(file_path, duration=300):
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
    rms_wpm = librosa.feature.rms(y=y_filtered, frame_length=frame_len_wpm, hop_length=hop_len_wpm)[0]
    rms_wpm = np.clip(rms_wpm, a_min=1e-10, a_max=None)
    intensity_db = 20 * np.log10(rms_wpm)
    intensity_db = intensity_db - np.max(intensity_db)
    peaks, _ = find_peaks(intensity_db, height=-25.0, distance=10, prominence=2.0)
    duration_mins = (len(y) / sr) / 60.0
    wpm = float((len(peaks) / duration_mins) / 1.45) if duration_mins > 0 else 0.0
    
    return {
        "clips": clip_count,
        "peak_db": peak_db,
        "noise_floor": noise_floor,
        "vocal_clarity": vocal_clarity_pct,
        "rolloff": mean_rolloff,
        "lufs": lufs,
        "lra": lra,
        "wpm": wpm
    }

def format_cell(value_str, subscore):
    if subscore >= 0.85:
        icon = "🟢"
    elif subscore >= 0.50:
        icon = "🟡"
    else:
        icon = "🔴"
    return f"{icon} {value_str}"

def run_full_report():
    print("=" * 75)
    print("AWGP & YAGYAVALKYA AUDIOBOOK QUALITY AUDITOR - BENCHMARK REPORT")
    print("=" * 75)
    
    results = []
    for vid, (cat, title) in VIDEOS.items():
        fname = f"audio_{vid}.wav"
        if not os.path.exists(fname):
            print(f"Skipping {vid} (file not found)")
            continue
            
        print(f"Auditing [{cat}] {title}...")
        m = analyze_audio(fname)
        final_score, subscores = calculate_weighted_quality_score(m)
        
        if final_score >= 85.0:
            verdict = "🟢 **PRO GRADE**"
        elif final_score >= 68.0:
            verdict = "🟡 **GOOD**"
        elif final_score >= 50.0:
            verdict = "🟠 **FAIR**"
        else:
            verdict = "🔴 **POOR**"
            
        results.append({
            "cat": cat,
            "title": title,
            "vid": vid,
            "score": final_score,
            "verdict": verdict,
            "m": m,
            "subscores": subscores
        })
        
    # Sort by Category priority (Pro -> Yagyavalkya -> AWGP -> Amateur), then Score descending
    cat_order = {"Pro": 0, "Yagyavalkya": 1, "AWGP": 2, "Amateur": 3}
    results.sort(key=lambda x: (cat_order.get(x["cat"], 9), -x["score"]))
    
    lines = [
        "# 🎧 Master Audiobook Quality Audit Report & Editor Guide",
        "**Multi-Channel Benchmark: AWGP & Yagyavalkya Audiobooks vs. Pro Standards**\n",
        "This report evaluates audiobook recordings across the **AWGP / Shantikunj** and **Yagyavalkya Newsletter** channels against international audio publishing standards (**Audible ACX**, **Penguin Audio**). The quality scoring model is scientifically grounded in psychoacoustic research on **cognitive listening effort**, **auditory fatigue**, and **speech intelligibility** during extended listening sessions.\n",
        "---",
        "## 📊 Executive Quality Scorecard\n",
        "| Category | Audiobook Title | Listener Score | Sound Cracking (25%) | Vocal Clarity (20%) | Noise Floor (20%) | Master Volume (15%) | Reading Speed (10%) | High Air (5%) | Dynamic Life (5%) | Quality Rating |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]
    
    for r in results:
        m = r["m"]
        s = r["subscores"]
        
        str_clips = format_cell(f"{m['clips']} clips", s["clips"])
        str_clarity = format_cell(f"{m['vocal_clarity']:.1f}%", s["clarity"])
        str_noise = format_cell(f"{m['noise_floor']:.1f} dB", s["noise_floor"])
        str_lufs = format_cell(f"{m['lufs']:.1f} LUFS", s["lufs"])
        str_wpm = format_cell(f"{m['wpm']:.0f} WPM", s["wpm"])
        str_rolloff = format_cell(f"{m['rolloff']:.0f} Hz", s["rolloff"])
        str_lra = format_cell(f"{m['lra']:.1f} LU", s["lra"])
        
        link = f"[{r['title']}](https://www.youtube.com/watch?v={r['vid']})"
        lines.append(
            f"| {r['cat']} | {link} | **{r['score']:.1f}/100** | {str_clips} | {str_clarity} | {str_noise} | {str_lufs} | {str_wpm} | {str_rolloff} | {str_lra} | {r['verdict']} |"
        )
        
    lines.extend([
        "\n---",
        "## 🧠 Psychoacoustic Research & Weightage Formulation",
        "",
        "Why are the weights assigned the way they are? In speech perception science (AES, ITU-T, and cognitive psychoacoustic studies), audiobooks represent **long-form solitary listening** (typically 30 minutes to several hours on headphones).",
        "The weights reflect the direct impact of each acoustic factor on **Cognitive Listening Effort** and **Auditory Fatigue**:",
        "",
        "$$\\text{Overall Listener Score} = \\left( \\sum_{i=1}^{7} W_i \\times S_i \\right) \\times 100$$",
        "",
        "### Weight Distribution Table",
        "| Priority Tier | Acoustic Parameter | Weight ($W_i$) | Psychoacoustic Impact on Listener Experience |",
        "| :---: | :--- | :---: | :--- |",
        "| **Tier 1: Fatal Fatigue (45%)** | **⚡ Sound Cracking & Distortion** | **25%** | **Highest Cognitive Fatigue Factor**. Clipped samples create harsh, non-harmonic high-frequency square wave splatters. The auditory cortex is forced to constantly reconstruct missing waveform data, causing headaches and listener dropout within minutes. |",
        "| | **🎙️ Vocal Formant Intelligibility** | **20%** | **Cognitive Decoding Load**. The $1-4\\text{ kHz}$ frequency band carries the $F_2/F_3$ speech formants and consonant bursts (/t/, /k/, /s/, /p/). If muffled ($<6\\%$), the brain works overtime guessing phonemes, resulting in mental exhaustion. |",
        "| **Tier 2: Immersion & Comfort (35%)** | **🤫 Background Noise Floor** | **20%** | **Sensory Gating Load**. Constant background hiss, fan drone, or electrical hum forces the brain's sensory filter to constantly work to separate voice from noise. On headphones, background noise destroys presence and immersion. |",
        "| | **🔊 Master Volume & Loudness** | **15%** | **Acoustic Comfort**. Standardized volume ($-20\\text{ LUFS}$) ensures comfortable listening without requiring the user to constantly reach for the volume knob. Prevents ear strain from quiet audio and pain from hot audio. |",
        "| **Tier 3: Cadence & Artistry (20%)** | **⏱️ Reading Speed & Pacing** | **10%** | **Working Memory Retention**. Fast narration ($>160\\text{ WPM}$) overflows short-term working memory, ruining retention of spiritual/philosophical texts. Natural pauses ($0.5-1.0\\text{s}$) allow cognitive processing. |",
        "| | **🌬️ High-Frequency Air (Rolloff)** | **5%** | **Acoustic Space Definition**. High frequencies ($8-16\\text{ kHz}$) provide natural timbre and room cues, preventing the claustrophobic 'talking in a box' feel. |",
        "| | **🎭 Dynamic Life & Expression (LRA)** | **5%** | **Emotional Engagement**. Natural vocal dynamics keep the listener emotionally engaged, whereas hyper-compressed or totally flat reading induces boredom. |",
        "| **Total** | | **100%** | |",
        "",
        "---",
        "## 📐 Sub-Score Calculation & Tolerance Curves ($S_i$)",
        "",
        "Each parameter is evaluated on a continuous scale ($0.0 - 1.0$) with calibrated psychoacoustic tolerances:",
        "",
        "1. **⚡ Sound Cracking (Clipping Count)**: $0\\text{ clips} = 1.0$ | $1-5\\text{ clips} = 0.85$ | $6-50\\text{ clips} = 0.60$ | $51-500\\text{ clips} = 0.30$ | $>2000\\text{ clips} = 0.0$",
        "2. **🎙️ Vocal Formant Clarity ($1-4\\text{ kHz}$)**: $\\ge 11.0\\% = 1.0$ | $9.5-11.0\\% = 0.85-1.0$ | $6.5-9.5\\% = 0.50-0.85$ | $<3.5\\% \\le 0.20$",
        "3. **🤫 Background Noise Floor**: $\\le -65\\text{ dBFS} = 1.0$ | $-65\\text{ to }-58\\text{ dBFS} = 0.85-1.0$ | $-58\\text{ to }-48\\text{ dBFS} = 0.40-0.85$ | $>-40\\text{ dBFS} = 0.0$",
        "4. **🔊 Master Volume (LUFS)**: $-22\\text{ to }-19\\text{ LUFS} = 1.0$ | $-24\\text{ to }-22\\text{ LUFS} / -19\\text{ to }-17.5\\text{ LUFS} = 0.85-1.0$ | $<-28\\text{ or }>-14\\text{ LUFS} \\le 0.30$",
        "5. **⏱️ Reading Speed (WPM)**: $110-135\\text{ WPM} = 1.0$ | $95-110\\text{ WPM} / 135-150\\text{ WPM} = 0.85-1.0$ | $150-170\\text{ WPM} = 0.40-0.85$ | $>170\\text{ WPM} \\le 0.40$",
        "6. **🌬️ High-Frequency Air (Rolloff)**: $\\ge 5000\\text{ Hz} = 1.0$ | $4700-5000\\text{ Hz} = 0.85-1.0$ | $<3500\\text{ Hz} \\le 0.50$",
        "7. **🎭 Dynamic Life (LRA)**: $\\ge 14.0\\text{ LU} = 1.0$ | $12.0-14.0\\text{ LU} = 0.85-1.0$ | $<9.0\\text{ LU} \\le 0.50$",
        "",
        "---",
        "## 🛡️ Dealbreaker Safety Caps",
        "Regardless of other high scores, catastrophic technical defects cap the maximum final score to protect listener ears:",
        "* **Severe Digital Clipping ($>1,000$ clips)**: Final Score Capped at **$\\le 45.0/100$ (POOR)**.",
        "* **Loud Background Noise ($> -45.0\\text{ dBFS}$)**: Final Score Capped at **$\\le 45.0/100$ (POOR)**.",
        "",
        "---",
        "## 🛠️ Step-by-Step 'Audacity Master Preset' for AWGP Editors\n",
        "To easily bring any AWGP or Yagyavalkya raw recording up to the **🟢 Pro Studio Grade (85+ Score)**, editors can apply this exact 4-step chain in Audacity:\n",
        "1. **Step 1: Clean Low Rumble** $\\rightarrow$ *Effect $\\rightarrow$ High-Pass Filter (80 Hz, 12 dB/octave rolloff)* to remove mic thumps and room rumble.",
        "2. **Step 2: Gentle Background Clean** $\\rightarrow$ *Effect $\\rightarrow$ Noise Reduction (Select pause silence $\\rightarrow$ Get Profile $\\rightarrow$ 8 dB reduction, 6 sensitivity)*.",
        "3. **Step 3: Enhance Vocal Clarity** $\\rightarrow$ *Effect $\\rightarrow$ Filter Curve EQ (Gently boost +2.5 dB between 2.5 kHz and 5.0 kHz)*.",
        "4. **Step 4: Master Volume & Safety** $\\rightarrow$ *Effect $\\rightarrow$ Loudness Normalization to -20.0 LUFS*, followed by *Limiter (Soft Limit to -3.0 dB)*."
    ])
    
    report_content = "\n".join(lines)
    with open("audio_qa_report_v2.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    with open("audio_qa_report_v3.md", "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print("Done! Audit completed and saved to audio_qa_report_v2.md")

if __name__ == "__main__":
    run_full_report()
