# 🎧 AWGP & Shantikunj Audiobook Quality Auditor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

An automated, objective Digital Signal Processing (DSP) and psychoacoustic audio quality testing tool for audiobooks. Designed for **AWGP / Shantikunj** audio editors and narrators to evaluate and master their audio to meet international publishing standards (**Audible ACX**, **Penguin Audio**, and **EBU R128**).

---

## 🌟 Features

* **Multi-Format Audio Upload**: Supports `.mp3`, `.wav`, `.m4a`, `.flac`, and `.ogg`.
* **YouTube URL Auditor**: Direct audio analysis from YouTube channels like [@Shantikunjvideo_Audio_Books](https://www.youtube.com/@Shantikunjvideo_Audio_Books) and [@yagyavalkyanewsletter6311](https://www.youtube.com/@yagyavalkyanewsletter6311).
* **Psychoacoustically Grounded 7-Pillar Scoring ($0 - 100$)**:
  * ⚡ **Sound Cracking & Distortion ($25\%$)**: Digital clipping count ($0\text{ clips}$ required).
  * 🎙️ **Vocal Formant Clarity ($20\%$)**: Intelligibility band ($1-4\text{ kHz}$).
  * 🤫 **Background Noise Floor ($20\%$)**: Silence purity ($\le -58\text{ dBFS}$).
  * 🔊 **Master Volume ($15\%$)**: Integrated loudness ($-20\text{ LUFS}$ target).
  * ⏱️ **Reading Speed & Pacing ($10\%$)**: Comfortable storytelling tempo ($110-135\text{ WPM}$).
  * 🌬️ **High-Frequency Air ($5\%$)**: Spectral rolloff ($\ge 4700\text{ Hz}$).
  * 🎭 **Dynamic Life ($5\%$)**: Theatrical expression ($\ge 13\text{ LU}$).
* **1-Click Audacity Macro**: Automatically cleans rumble, normalizes volume to $-20\text{ LUFS}$, and sets safety soft limiting at $-3.0\text{ dB}$.

---

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/audiobook-quality-auditor.git
cd audiobook-quality-auditor
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud (100% Free)

1. Fork or push this repository to your GitHub account.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with GitHub.
3. Click **"New App"** $\rightarrow$ select this repository $\rightarrow$ select `app.py`.
4. Click **"Deploy"**!
