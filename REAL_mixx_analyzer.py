import os
import numpy as np
import librosa
import librosa.display
from scipy.signal import convolve2d, find_peaks
import time
import matplotlib.pyplot as plt
import pandas as pd

PROFILE = "macro"
HOP = 2048
SR = 22050

W_TIMBRE = 0.6  # 음색 가중치 (60%)
W_HARMONY = 0.4 # 화성 가중치 (40%)
# ==================================================

PROFILES = {
    "macro": dict(
        L_TIMBRE_SEC=8.0,
        L_HARM_SEC=8.0,
        SMOOTH_SEC=0.8,
        ABS_THRESHOLD=0.1,
        MIN_GAP_SEC=1.5,
        TRIM_EDGES_SEC=3.0,
        RMS_GATE_Q=0.15
    ),
}

# --- 핵심 알고리즘 함수 ---
def _foote_novelty(SSM, L_frames=64):
    T = SSM.shape[0]
    if T < (2*L_frames + 1):
        return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)
    
    k = 2*L_frames + 1
    K = np.zeros((k, k), dtype=np.float32)
    K[:L_frames, :L_frames]     =  1
    K[:L_frames, L_frames+1:]   = -1
    K[L_frames+1:, :L_frames]   = -1
    K[L_frames+1:, L_frames+1:] =  1
    
    w = np.hanning(k).astype(np.float32)
    K *= np.outer(w, w)

    N = convolve2d(SSM, K, mode="same", boundary="symm")
    nov = np.diag(N).astype(np.float32)
    
    edge = min(2*L_frames, T//4)
    if edge > 0:
        nov[:edge]  = 0.0
        nov[-edge:] = 0.0
        
    nov_raw = nov.copy()
    np.clip(nov_raw, 0, None, out=nov_raw)
    
    nov_min = nov.min()
    nov_max = nov.max()
    nov_norm = (nov - nov_min) / (nov_max - nov_min + 1e-9)
    
    return nov_raw, nov_norm

def _smooth_1d(x, win_sec, fps):
    w = max(1, int(win_sec * fps))
    if w % 2 == 0:
        w += 1
    h = np.hanning(w).astype(np.float32)
    h /= (h.sum() + 1e-8)
    return np.convolve(x, h, mode="same")

def _z_and_cosine(F):
    F = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + 1e-8)
    X = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-8)
    S = (X.T @ X)
    S = (S + 1.0) / 2.0
    return S.astype(np.float32)

# --- 메인 분석 파이프라인 ---

def analyze(path, *, hop, sr, params):
    filename = os.path.basename(path)
    print(f"[Module] Processing: {filename}...")
    
    # 1. 오디오 로드
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    dur = len(y) / sr
    fps = sr / float(hop)

    # 2. 특징 추출 (음색 + 화성)
    # MFCC (Timbre)
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop, power=2.0)
    Mdb = librosa.power_to_db(M, ref=np.max)
    mfcc = librosa.feature.mfcc(S=Mdb, n_mfcc=20)
    dmf  = librosa.feature.delta(mfcc, order=1)
    F_tim = np.vstack([mfcc, dmf]).astype(np.float32)
    
    # Chroma CENS (Harmony)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    F_harm = chroma.astype(np.float32)

    # 3. 노블티 커브 계산
    L_tim_frames = max(1, int(params["L_TIMBRE_SEC"] * fps))
    L_harm_frames = max(1, int(params["L_HARM_SEC"] * fps))

    S_tim  = _z_and_cosine(F_tim)
    nov_tim_raw, nov_tim_norm = _foote_novelty(S_tim,  L_frames=L_tim_frames)
    
    S_harm = _z_and_cosine(F_harm)
    nov_harm_raw, nov_harm_norm = _foote_novelty(S_harm, L_frames=L_harm_frames)
    
    # 가중치 적용
    nov_norm_un_smoothed = W_TIMBRE * nov_tim_norm + W_HARMONY * nov_harm_norm
    nov_raw_un_smoothed = W_TIMBRE * nov_tim_raw + W_HARMONY * nov_harm_raw

    # 4. 후처리
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
    thr_rms = np.quantile(rms, params["RMS_GATE_Q"])
    edge_f = int(params["TRIM_EDGES_SEC"] * fps)
    
    nov_norm_smoothed = _smooth_1d(nov_norm_un_smoothed, params["SMOOTH_SEC"], fps)
    nov_norm_smoothed[rms < thr_rms] = 0.0
    if edge_f > 0:
        nov_norm_smoothed[:edge_f]  = 0.0
        nov_norm_smoothed[-edge_f:] = 0.0

    # 강도 측정용 Raw Curve
    nov_raw_smoothed = _smooth_1d(nov_raw_un_smoothed, params["SMOOTH_SEC"], fps)
    nov_raw_smoothed[rms < thr_rms] = 0.0
    if edge_f > 0:
        nov_raw_smoothed[:edge_f]  = 0.0
        nov_raw_smoothed[-edge_f:] = 0.0

    # 5. 피크 검출
    std_mult = 1.8
    threshold = np.mean(nov_norm_smoothed) + std_mult * np.std(nov_norm_smoothed)
    threshold = max(threshold, params["ABS_THRESHOLD"])
    
    distance = max(1, int(params["MIN_GAP_SEC"] * fps))
    
    peaks, _ = find_peaks(
        nov_norm_smoothed,
        height=threshold, 
        distance=distance,
        prominence=0.05
    )
    
    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    total = len(peak_times)
    cpm = total / max(1e-6, dur / 60.0)
    
    # 통계량 계산
    if total > 0:
        raw_peak_magnitudes = nov_raw_smoothed[peaks]
        avg_magnitude = np.mean(raw_peak_magnitudes)
        max_magnitude = np.max(raw_peak_magnitudes)
        std_dev_magnitude = np.std(raw_peak_magnitudes)
    else:
        avg_magnitude = 0.0
        max_magnitude = 0.0
        std_dev_magnitude = 0.0
    
    # 통계 지표
    crest_factor = max_magnitude / (avg_magnitude + 1e-9)
    cv_score = std_dev_magnitude / (avg_magnitude + 1e-9)
    mixx_index = cpm * crest_factor

    return {
        "filename": filename,
        "duration_sec": dur,
        "total_changes": total,
        "changes_per_min": cpm,
        "avg_magnitude": avg_magnitude,
        "max_magnitude": max_magnitude,
        "std_dev_magnitude": std_dev_magnitude,
        
        "crest_factor": crest_factor,
        "cv_score": cv_score,
        "mixx_index": mixx_index,
        
        "peak_times_sec_str": str([round(t, 3) for t in peak_times.tolist()]),
        
        "debug_nov_curve": nov_norm_smoothed,
        "debug_peaks": peaks,
        "debug_threshold": threshold
    }


# 이 파일은 '도구'이므로 직접 실행하면 간단한 메시지만 출력하고 끝납니다.
# CSV 저장은 배치 파일(Batch Runner)이 담당합니다.
if __name__ == "__main__":
    print("="*50)
    print(" [MIXXPOP V2 Analyzer Module]")
    print(" 이 파일은 분석기 '부품'입니다.")
    print(" 배치 분석을 하려면 'REAL_mixx_analyzer.py'를 실행하세요.")
    print("="*50)