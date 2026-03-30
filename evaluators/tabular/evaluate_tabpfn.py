from tabpfn.model_loading import load_fitted_tabpfn_model
from pathlib import Path
from options import DEVICE
from options import UNIQUE_ID
from cli import DeepTuneVisionOptions
from utils import save_process_times
from utils import RunType
from joblib import load
import json
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from adaptive_error.run import run_adaptive_error

def main():

    args = DeepTuneVisionOptions(RunType.TabPFN)
    TARGET = args.target_column
    MODEL_STR = 'TABPFN'
    MODE = args.mode
    OUT = (args.out / f"eval_output_{MODEL_STR}_{MODE}_{UNIQUE_ID}")
    FINETUNING_MODE = args.finetuning_mode
    EVAL_PATH = args.eval_df
    VAL_PATH = args.val_df
    MODEL_WEIGHTS = args.model_weights

    X_eval = pd.read_parquet(EVAL_PATH).drop(columns=[TARGET])
    y_eval = pd.read_parquet(EVAL_PATH)[TARGET]


    if MODE == 'cls':
        evaluate_tabpfn(
            X_eval=X_eval,
            y_eval=y_eval,
            out=OUT,
            model_path=MODEL_WEIGHTS,
            mode=MODE,
            args=args,
            finetuning_mode=FINETUNING_MODE,
            model_str='TABPFN',
        )

    elif MODE == 'reg':
        evaluate_tabpfn(
            X_eval=X_eval,
            y_eval=y_eval,
            out=OUT,
            model_path=MODEL_WEIGHTS,
            mode=MODE,
            args=args,
            finetuning_mode=FINETUNING_MODE,
            model_str='TABPFN',
        )

    else:
        raise ValueError(f"Unsupported evaluation mode: {MODE}. Supported modes are 'cls' and 'reg'.")

    if args.adaptive_error:
        try:
            print("Conducting adaptive error rate post-processing...")
            run_adaptive_error(
                    args=args,
                    modality='tabular',
                    model_version='tabpfn',
                    ckpt_directory=MODEL_WEIGHTS,
                    val_data_path=VAL_PATH,
                    test_data_path=EVAL_PATH,
                    out_dir=OUT,
                    target_column=TARGET,
            )
        except Exception as exc:
            import traceback
            tb_str = traceback.format_exc()
            print(f"Warning: adaptive error rate post-processing failed: {exc}")
            print(f"Error traceback:\n{tb_str}")
    
def evaluate_tabpfn(
        X_eval,
        y_eval,
        out,
        args,
        model_path,
        mode,
        finetuning_mode,
        model_str='TABPFN',
):
    
        if mode == 'cls':
            if finetuning_mode:

                clf = load(Path(model_path))
                eval_results = clf.predict(X_eval)
                accuracy = accuracy_score(y_eval, eval_results)
                print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

                result_dic ={
                        "accuracy": accuracy
                    }

            else:
                clf = load_fitted_tabpfn_model(model_path, device=DEVICE)
                eval_results = clf.predict(X_eval)
                accuracy = accuracy_score(y_eval, eval_results)
                print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")

                result_dic = {
                        "accuracy": accuracy,
                    }

        elif mode == 'reg':
            if finetuning_mode:
                reg = load(Path(model_path))
                eval_results = reg.predict(X_eval)
                mse = mean_squared_error(y_eval, eval_results)
                mae = mean_absolute_error(y_eval, eval_results)
                print(f"Evaluation MSE: {mse:.4f}")
                print(f"Evaluation MAE: {mae:.4f}")

                result_dic = {
                    "Mean Squared Error": mse,
                    "Mean Absolute Error": mae,
                }
            else:
                reg = load_fitted_tabpfn_model(model_path, device=DEVICE)
                eval_results = reg.predict(X_eval)
                mse = mean_squared_error(y_eval, eval_results)
                mae = mean_absolute_error(y_eval, eval_results)
                print(f"Evaluation MSE: {mse:.4f}")
                print(f"Evaluation MAE: {mae:.4f}")

                result_dic ={
                        "Mean Squared Error": mse,
                        "Mean Absolute Error": mae,
                    }
        else:
            raise ValueError(f"Unsupported evaluation mode: {mode}. Supported modes are 'cls' and 'reg'.")
        
        EVAL_OUTPUT_DIR = out
        args.save_args(EVAL_OUTPUT_DIR)
        with open(EVAL_OUTPUT_DIR / "full_metrics.json", 'w') as f:
            json.dump(result_dic, f, indent=4)
        save_process_times(epoch_times="For TabPFN we only track total time", total_duration="N/A", outdir=EVAL_OUTPUT_DIR, process="evaluation")
        print(f"Evaluation results saved to {EVAL_OUTPUT_DIR / 'full_metrics.json'}")

        return result_dic




if __name__ == "__main__":
    main()




