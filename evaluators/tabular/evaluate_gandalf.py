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


def evaluate(
    eval_df: Path,
    model_weights: Path,
    out:Path,
    args: DeepTuneVisionOptions,
    model_str='GANDALF',
):
    
    TEST_OUTPUT_DIR = (out / f"test_output_{model_str}_{UNIQUE_ID}")

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

    args = DeepTuneVisionOptions(RunType.EVAL)

    TEST_PATH = args.eval_df
    MODEL_WEIGHTS = args.model_weights
    OUT = args.out
    MODEL_STR = 'GANDALF'


    evaluate(
        eval_df=TEST_PATH,
        model_weights=MODEL_WEIGHTS,
        out=OUT,
        args=args,
        model_str=MODEL_STR
    )

if __name__ == "__main__":
    
    main()