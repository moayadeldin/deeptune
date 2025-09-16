import options
from pytorch_tabular import TabularModel
import pandas as pd
import json
from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from utils import get_model_cls,RunType,set_seed,save_process_times
import time


def main():

    args = DeepTuneVisionOptions(RunType.EVAL)

    TEST_PATH = args.eval_df
    MODEL_WEIGHTS = args.model_weights
    OUT = args.out
    MODEL_STR = 'GANDALF'
    
    TEST_OUTPUT_DIR = (OUT / f"test_output_{MODEL_STR}_{UNIQUE_ID}") if OUT else Path(f"deeptune_results/test_output_{MODEL_STR}_{UNIQUE_ID}")

    test = pd.read_parquet(TEST_PATH)
    
    start_time = time.time()

    loaded_model = TabularModel.load_model(MODEL_WEIGHTS)
    result = loaded_model.evaluate(test)
    print(result)
    print(f'Test Accuracy iS ', result[0]['test_accuracy'])
    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times=1, total_duration=total_time, outdir=TEST_OUTPUT_DIR, process="evaluation")
    args.save_args(TEST_OUTPUT_DIR)
    with open(TEST_OUTPUT_DIR / "full_metrics.json", 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    
    main()