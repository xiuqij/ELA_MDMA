#%%
import os
import pandas as pd
from joblib import Parallel, delayed
import utils_chase as utils

PARQUET_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\parquet"
OUTPUT_FOLDER = r"L:\Lopez Laboratory - NEURO\Xiuqi\ELA_MDMA\April_2026\chase"

EXPS = ["female_P42","male_P35"]
TPS = ["baseline","MDMA","MDMA_acute"]

MODEL_PATH = r"L:\Lopez Laboratory - NEURO\Xiuqi\Xmas training 2025\models\16012026_AUG_F1_0.685_on_test_tresh0.9.joblib"

WINDOW_SIZE = 20
WINDOW_STEP = 5
PROB_THR = 0.9
MAX_GAP_FRAMES = 3
MARGIN = 0.1

N_JOBS = 6   # adjust to your CPU (e.g., 6–12)

model = utils.load_model(MODEL_PATH)

# ---- worker ----
def process_file(file, input_dir, output_dir):
    pq_path = os.path.join(input_dir, file)
    video = os.path.splitext(file)[0]

    try:
        df_all = utils.process_one_file_batch(
            model=model,
            pq_path=pq_path,
            video=video,
            WINDOW_SIZE=WINDOW_SIZE,
            WINDOW_STEP=WINDOW_STEP,
            PROB_THR=PROB_THR,
            MAX_GAP_FRAMES=MAX_GAP_FRAMES,
            MARGIN=MARGIN
        )

        if df_all is not None and not df_all.empty:
            out_file = os.path.join(output_dir, f"{video}.csv")
            df_all.to_csv(out_file, index=False)

    except Exception as e:
        print(f"[ERROR] {video}: {e}")


# ---- main ----
for exp in EXPS:
    print(f"\nProcessing {exp}")
    for s in TPS:
        print(f"Processing {s}")
        input_dir = os.path.join(PARQUET_FOLDER, exp, s)
        output_dir = os.path.join(OUTPUT_FOLDER, exp, s)
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(os.listdir(input_dir))

        Parallel(n_jobs=N_JOBS)(
            delayed(process_file)(f, input_dir, output_dir)
            for f in files
        )
        print(f"Finished {s}")

    print(f"Finished {exp}")