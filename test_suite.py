import unittest
import numpy as np
import os
import sys
import librosa
from scoring import (
    calculate_weighted_quality_score,
    score_clipping,
    score_noise_floor,
    score_clarity,
    score_lufs,
    score_pacing,
    score_rolloff,
    score_lra
)
from app import analyze_audio_data

class TestAudioQASystem(unittest.TestCase):

    def test_scoring_curves(self):
        """Test individual parameter scoring curves and boundary tolerances."""
        # 1. Clipping
        self.assertEqual(score_clipping(0), 1.0)
        self.assertEqual(score_clipping(3), 0.85)
        self.assertEqual(score_clipping(25), 0.60)
        self.assertEqual(score_clipping(250), 0.30)
        self.assertEqual(score_clipping(5000), 0.0)

        # 2. Noise Floor
        self.assertEqual(score_noise_floor(-70.0), 1.0)
        self.assertAlmostEqual(score_noise_floor(-58.0), 0.85, places=2)
        self.assertAlmostEqual(score_noise_floor(-48.0), 0.40, places=2)
        self.assertEqual(score_noise_floor(-35.0), 0.0)

        # 3. Vocal Clarity
        self.assertEqual(score_clarity(12.0), 1.0)
        self.assertAlmostEqual(score_clarity(9.5), 0.85, places=2)
        self.assertAlmostEqual(score_clarity(6.5), 0.50, places=2)
        self.assertAlmostEqual(score_clarity(3.5), 0.20, places=2)

        # 4. LUFS Loudness
        self.assertEqual(score_lufs(-20.0), 1.0)
        self.assertAlmostEqual(score_lufs(-24.0), 0.85, places=2)
        self.assertAlmostEqual(score_lufs(-28.0), 0.40, places=2)
        self.assertLess(score_lufs(-12.0), 0.35)

        # 5. Reading Speed WPM
        self.assertEqual(score_pacing(120.0), 1.0)
        self.assertAlmostEqual(score_pacing(150.0), 0.85, places=2)
        self.assertAlmostEqual(score_pacing(170.0), 0.40, places=2)

        # 6. Spectral Rolloff
        self.assertEqual(score_rolloff(5200.0), 1.0)
        self.assertAlmostEqual(score_rolloff(4700.0), 0.85, places=2)
        self.assertAlmostEqual(score_rolloff(3500.0), 0.50, places=2)

        # 7. Dynamic Range LRA
        self.assertEqual(score_lra(15.0), 1.0)
        self.assertAlmostEqual(score_lra(12.0), 0.85, places=2)
        self.assertAlmostEqual(score_lra(9.0), 0.50, places=2)

    def test_weighted_scoring_engine(self):
        """Test overall weighted multi-criteria scoring and safety caps."""
        # Perfect Pro recording metrics
        pro_metrics = {
            "clips": 0,
            "peak_db": -3.2,
            "noise_floor": -68.0,
            "vocal_clarity": 12.5,
            "rolloff": 5200.0,
            "lufs": -20.5,
            "lra": 15.2,
            "wpm": 122.0
        }
        score, subscores = calculate_weighted_quality_score(pro_metrics)
        self.assertGreaterEqual(score, 95.0)
        self.assertAlmostEqual(subscores["clips"], 1.0)
        self.assertAlmostEqual(subscores["noise_floor"], 1.0)

        # Severely clipped audio dealbreaker cap
        clipped_metrics = {
            "clips": 5000,
            "peak_db": 0.0,
            "noise_floor": -65.0,
            "vocal_clarity": 11.0,
            "rolloff": 5000.0,
            "lufs": -20.0,
            "lra": 14.0,
            "wpm": 120.0
        }
        score_clip, _ = calculate_weighted_quality_score(clipped_metrics)
        self.assertLessEqual(score_clip, 45.0)

        # Loud background noise dealbreaker cap
        noisy_metrics = {
            "clips": 0,
            "peak_db": -3.0,
            "noise_floor": -38.0,
            "vocal_clarity": 11.0,
            "rolloff": 5000.0,
            "lufs": -20.0,
            "lra": 14.0,
            "wpm": 120.0
        }
        score_noise, _ = calculate_weighted_quality_score(noisy_metrics)
        self.assertLessEqual(score_noise, 45.0)

    def test_audio_analysis_on_real_files(self):
        """Test audio feature extraction pipeline on benchmark WAV files."""
        test_file = "audio_XvIdC56hHVo.wav"
        if os.path.exists(test_file):
            y, sr = librosa.load(test_file, sr=None, duration=60)
            metrics, score, subscores = analyze_audio_data(y, sr)
            self.assertIn("clips", metrics)
            self.assertIn("noise_floor", metrics)
            self.assertIn("vocal_clarity", metrics)
            self.assertIn("rolloff", metrics)
            self.assertIn("lufs", metrics)
            self.assertIn("lra", metrics)
            self.assertIn("wpm", metrics)
            self.assertGreater(score, 80.0)

if __name__ == "__main__":
    unittest.main()
