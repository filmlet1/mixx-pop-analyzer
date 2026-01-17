import os, csv, time, sys
import matplotlib.pyplot as plt
import numpy as np
import librosa

try:
    # W_TIMBRE, W_HARMONY는 기록용으로 가져옵니다
    from REAL_auto_analyzer import analyze, PROFILES, HOP, SR, W_TIMBRE, W_HARMONY
except ImportError:
    print("="*50)
    print("오류: 'mixxpopv2.py' 를 같은 폴더에 두세요.")
    print("="*50)
    sys.exit(1)

# ===== 배치 설정 =====
AUDIO_FOLDER   = r""   # 분석할 파일들 있는 폴더
ANALYSIS_PROFILE = "macro"     # 6:4 가중치 로직은 macro 프로필 기반
RESULTS_FILE   = r"NMIXX_Final_Analysis_Results.csv"
PLOT_FOLDER    = r".\analysis_plots_final"
# =====================

def run_batch():
    if not os.path.exists(AUDIO_FOLDER):
        print(f"오류: '{AUDIO_FOLDER}' 폴더가 없습니다."); return
    
    params = PROFILES.get(ANALYSIS_PROFILE)
    if not params:
        raise ValueError(f"프로필 '{ANALYSIS_PROFILE}'을 찾을 수 없습니다.")

    os.makedirs(PLOT_FOLDER, exist_ok=True)

    exts = ('.mp3', '.flac', '.wav', '.m4a')
    files = [f for f in os.listdir(AUDIO_FOLDER) if f.lower().endswith(exts)]
    if not files:
        print(f"오류: '{AUDIO_FOLDER}' 에 오디오 파일이 없습니다."); return

    all_results = []
    t0 = time.time()
    print(f"=== 총 {len(files)}개 파일 배치 분석 시작 (Weight {W_TIMBRE}:{W_HARMONY}) ===")
    print(f"결과 → {RESULTS_FILE}")
    print(f"그래프 → {PLOT_FOLDER}")

    for i, fn in enumerate(files, 1):
        fp = os.path.join(AUDIO_FOLDER, fn)
        print(f"\n--- [{i}/{len(files)}] {fn} ---")
        try:
            res = analyze(fp, hop=HOP, sr=SR, params=params)
            
            if res is None:
                print(f"Skipping {fn} (Load Error)")
                continue

            # ===== CSV 저장용 데이터 매핑 =====
            row = {
                "filename": res["filename"],
                "mixx_index": res["mixx_index"],       # 종합 점수
                "crest_factor": res["crest_factor"],   # 임팩트 점수
                "cv_score": res["cv_score"],           # 변동성 점수
                "changes_per_min": res["changes_per_min"],
                "total_changes": res["total_changes"],
                "avg_magnitude": res["avg_magnitude"],
                "max_magnitude": res["max_magnitude"],
                "std_dev_magnitude": res["std_dev_magnitude"],
                "duration_sec": res["duration_sec"],
                "peak_times": res["peak_times_sec_str"],
                "profile": ANALYSIS_PROFILE,
                "weights": f"{W_TIMBRE}:{W_HARMONY}"   # 가중치 정보 기록
            }
            all_results.append(row)

            # ===== 그래프 저장 =====
            nov = res["debug_nov_curve"]
            thr = res["debug_threshold"]
            peaks = res["debug_peaks"]
            
            # 시간축 생성 (librosa 사용)
            times = librosa.frames_to_time(np.arange(len(nov)), sr=SR, hop_length=HOP)
            
            # 피크 시간축
            peak_times_viz = librosa.frames_to_time(peaks, sr=SR, hop_length=HOP)

            fig, ax = plt.subplots(figsize=(15,6))
            ax.plot(times, nov, label=f"Novelty (W_Tim={W_TIMBRE}, W_Harm={W_HARMONY})", color='k')
            ax.axhline(y=thr, color='r', linestyle=':', label=f"Threshold={thr:.3f}")
            ax.plot(peak_times_viz, nov[peaks], "x", color="r", markersize=10, label="Detected Peaks")
            
            ax.set_title(f"MIXX POP Analysis — {fn} (Index: {res['mixx_index']:.2f})")
            ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Weighted Novelty Score")
            ax.legend(); ax.grid(True, alpha=0.5)
            
            plot_path = os.path.join(PLOT_FOLDER, f"{os.path.splitext(fn)[0]}_plot.png")
            plt.tight_layout(); plt.savefig(plot_path); plt.close(fig)
            print(f"   -> 그래프 저장: {plot_path}")
            print(f"   -> MIXX Index: {res['mixx_index']:.3f}")

        except Exception as e:
            print(f"!!!! [{i}/{len(files)}] 실패: {fn} — {e}")
            all_results.append({"filename": fn, "error": str(e)})

    print("\n" + "="*50)
    print(f"=== 완료. 총 소요 {time.time()-t0:.2f} sec ===")
    print("="*50)

    # ===== CSV 저장 =====
    if not all_results:
        print("분석 결과가 없습니다."); return

    header = set()
    for r in all_results: header.update(r.keys())
    
    priority_cols = ["filename", "mixx_index", "crest_factor", "cv_score", "changes_per_min"]
    remaining_cols = sorted([h for h in header if h not in priority_cols])
    ordered = priority_cols + remaining_cols

    # 데이터에 없는 컬럼이 priority에 있을 경우 제거
    ordered = [c for c in ordered if c in header]

    try:
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            w.writerows(all_results)
        print(f"\n성공! → {os.path.abspath(RESULTS_FILE)}")
    except Exception as e:
        print(f"\n!!!! CSV 저장 실패: {e} (엑셀 열려있는지 확인)")

if __name__ == "__main__":

    run_batch()
