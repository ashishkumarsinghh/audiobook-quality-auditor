import numpy as np

def score_clipping(clip_count):
    if clip_count == 0:
        return 1.0
    elif clip_count <= 5:
        return 0.85
    elif clip_count <= 50:
        return 0.60
    elif clip_count <= 500:
        return 0.30
    elif clip_count <= 2000:
        return 0.10
    else:
        return 0.0

def score_noise_floor(nf):
    # Ideal <= -65 dBFS, Acceptable <= -58 dBFS, Distracting >= -48 dBFS, Intolerable >= -40 dBFS
    if nf <= -65.0:
        return 1.0
    elif nf <= -58.0:
        return 0.85 + 0.15 * ((-58.0 - nf) / 7.0)
    elif nf <= -48.0:
        return 0.40 + 0.45 * ((-48.0 - nf) / 10.0)
    elif nf <= -40.0:
        return 0.10 + 0.30 * ((-40.0 - nf) / 8.0)
    else:
        return 0.0

def score_clarity(clarity_pct):
    # Formant intelligibility band 1-4 kHz
    # Ideal >= 11.0%, Good >= 9.5%, Muffled <= 6.5%, Severe <= 3.5%
    if clarity_pct >= 11.0:
        return 1.0
    elif clarity_pct >= 9.5:
        return 0.85 + 0.15 * ((clarity_pct - 9.5) / 1.5)
    elif clarity_pct >= 6.5:
        return 0.50 + 0.35 * ((clarity_pct - 6.5) / 3.0)
    elif clarity_pct >= 3.5:
        return 0.20 + 0.30 * ((clarity_pct - 3.5) / 3.0)
    else:
        return max(0.0, 0.20 * (clarity_pct / 3.5))

def score_lufs(lufs):
    # Ideal: -20.0 LUFS (-23 to -18 is standard zone)
    if -22.0 <= lufs <= -19.0:
        return 1.0
    elif -24.0 <= lufs < -22.0:
        return 0.85 + 0.15 * ((lufs - (-24.0)) / 2.0)
    elif -19.0 < lufs <= -17.5:
        return 0.85 + 0.15 * ((-17.5 - lufs) / 1.5)
    elif -28.0 <= lufs < -24.0:
        return 0.40 + 0.45 * ((lufs - (-28.0)) / 4.0)
    elif -17.5 < lufs <= -14.0:
        return 0.30 + 0.55 * ((-14.0 - lufs) / 3.5)
    elif lufs < -28.0:
        return max(0.0, 0.40 * (1.0 - ((-28.0 - lufs) / 10.0)))
    else: # lufs > -14.0 (brickwalled / painful)
        return max(0.0, 0.30 * (1.0 - ((lufs - (-14.0)) / 5.0)))

def score_pacing(wpm):
    # Ideal: 110 - 135 WPM
    # Acceptable: 95 - 150 WPM
    if 110.0 <= wpm <= 135.0:
        return 1.0
    elif 95.0 <= wpm < 110.0:
        return 0.85 + 0.15 * ((wpm - 95.0) / 15.0)
    elif 135.0 < wpm <= 150.0:
        return 0.85 + 0.15 * ((150.0 - wpm) / 15.0)
    elif 80.0 <= wpm < 95.0:
        return 0.50 + 0.35 * ((wpm - 80.0) / 15.0)
    elif 150.0 < wpm <= 170.0:
        return 0.40 + 0.45 * ((170.0 - wpm) / 20.0)
    elif wpm < 80.0:
        return max(0.0, 0.50 * (wpm / 80.0))
    else: # wpm > 170.0 (rushed)
        return max(0.0, 0.40 * (1.0 - ((wpm - 170.0) / 30.0)))

def score_rolloff(rolloff):
    # Ideal >= 5000 Hz, Good >= 4700 Hz, Poor <= 3500 Hz
    if rolloff >= 5000.0:
        return 1.0
    elif rolloff >= 4700.0:
        return 0.85 + 0.15 * ((rolloff - 4700.0) / 300.0)
    elif rolloff >= 3500.0:
        return 0.50 + 0.35 * ((rolloff - 3500.0) / 1200.0)
    elif rolloff >= 2500.0:
        return 0.20 + 0.30 * ((rolloff - 2500.0) / 1000.0)
    else:
        return max(0.0, 0.20 * (rolloff / 2500.0))

def score_lra(lra):
    # Ideal >= 14.0 LU, Good >= 12.0 LU, Flat <= 9.0 LU
    if lra >= 14.0:
        return 1.0
    elif lra >= 12.0:
        return 0.85 + 0.15 * ((lra - 12.0) / 2.0)
    elif lra >= 9.0:
        return 0.50 + 0.35 * ((lra - 9.0) / 3.0)
    elif lra >= 6.0:
        return 0.20 + 0.30 * ((lra - 6.0) / 3.0)
    else:
        return max(0.0, 0.20 * (lra / 6.0))

def calculate_weighted_quality_score(m):
    """
    Psychoacoustically Grounded Listener Experience Weighting:
    - 25% Digital Distortion / Clipping (Highest cognitive fatigue cause)
    - 20% Vocal Formant Intelligibility (Brain listening effort to decode words)
    - 20% Background Noise Floor (Headphone immersion & sensory gating load)
    - 15% Master Volume / Loudness (Comfortable, non-jarring listening level)
    - 10% Reading Speed & Pacing (Working memory comprehension rate)
    -  5% High-Frequency Air / Rolloff (Natural open-air presence)
    -  5% Dynamic Life / Expression (Narrative engagement vs monotone reading)
    Total = 100%
    """
    weights = {
        "clips": 0.25,       # 25% Audio Cracking & Distortion
        "clarity": 0.20,     # 20% Vocal Formant Intelligibility
        "noise_floor": 0.20, # 20% Background Noise / Hiss
        "lufs": 0.15,        # 15% Master Volume & Loudness
        "wpm": 0.10,         # 10% Reading Speed & Pacing
        "rolloff": 0.05,     #  5% High-Frequency Air
        "lra": 0.05          #  5% Dynamic Life & Expression
    }
    
    subscores = {
        "clips": score_clipping(m["clips"]),
        "clarity": score_clarity(m["vocal_clarity"]),
        "noise_floor": score_noise_floor(m["noise_floor"]),
        "lufs": score_lufs(m["lufs"]),
        "wpm": score_pacing(m["wpm"]),
        "rolloff": score_rolloff(m["rolloff"]),
        "lra": score_lra(m["lra"])
    }
    
    raw_score = sum(weights[k] * subscores[k] for k in weights) * 100.0
    
    # Dealbreaker safety caps for listener experience:
    # Extreme digital distortion (>1000 clips) or loud background hum (>-45 dB)
    # physically ruins the listening experience.
    final_score = raw_score
    if m["clips"] > 1000:
        final_score = min(final_score, 45.0)
    if m["noise_floor"] > -45.0:
        final_score = min(final_score, 45.0)
        
    final_score = round(max(0.0, min(100.0, final_score)), 1)
    
    return final_score, subscores
