import options
from pytorch_tabular import TabularModel
import pandas as pd
import json
from cli import DeepTuneVisionOptions
from pathlib import Path
from options import UNIQUE_ID
from utils import get_model_cls,RunType,set_seed,save_process_times
import time
import os

from adaptive_error.run import run_adaptive_error

def evaluate(
    eval_df: Path,
    model_weights: Path,
    out:Path,
    args: DeepTuneVisionOptions,
    model_str='GANDALF',
):
    
    TEST_OUTPUT_DIR = out

    model_weights = os.path.join(model_weights, 'GANDALF_model')

    eval_df = pd.read_parquet(eval_df)

    start_time = time.time()

    loaded_model = TabularModel.load_model(model_weights)
    result_dic = loaded_model.evaluate(eval_df)
    mapping = {
    "test_loss": "loss",
    "test_loss_0": "loss",
    "test_accuracy": "accuracy",
}
    result_dic = result_dic[0]  # extract dictionary from list
    result_dic = {mapping.get(k, k): v for k, v in result_dic.items() if k not in ["test_loss_0"]}

    # print(result_dic)
    # print(f'Test Accuracy iS ', result_dic[0]['test_accuracy'])
    end_time = time.time()
    total_time = end_time - start_time
    args.save_args(TEST_OUTPUT_DIR)
    with open(TEST_OUTPUT_DIR / "full_metrics.json", 'w') as f:
        json.dump(result_dic, f, indent=4)
    save_process_times(epoch_times=1, total_duration=total_time, outdir=TEST_OUTPUT_DIR, process="evaluation")

    return result_dic


def main():

    args = DeepTuneVisionOptions(RunType.GANDALF)

    TEST_PATH = args.eval_df
    VAL_PATH = args.val_df
    TARGET = args.tabular_target_column
    MODEL_WEIGHTS = args.model_weights
    MODEL_STR = 'GANDALF'
    OUT = (args.out / f"test_output_{MODEL_STR}_{UNIQUE_ID}")

    _ = evaluate(
        eval_df=TEST_PATH,
        model_weights=MODEL_WEIGHTS,
        out=OUT,
        args=args,
        model_str=MODEL_STR
    )
    
    if args.adaptive_error:
        try:
            print("Conducting adaptive error rate post-processing...")
            run_adaptive_error(
                    args=args,
                    modality='tabular',
                    model_version='gandalf',
                    ckpt_directory=MODEL_WEIGHTS,
                    val_data_path=VAL_PATH,
                    test_data_path=TEST_PATH,
                    out_dir=OUT,
                    target_column=TARGET,
            )
        except Exception as exc:
            import traceback
            tb_str = traceback.format_exc()
            print(f"Warning: adaptive error rate post-processing failed: {exc}")
            print(f"Error traceback:\n{tb_str}")
    
    

if __name__ == "__main__":
    
    main()