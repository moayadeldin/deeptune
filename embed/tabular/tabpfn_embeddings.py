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
from tabpfn_extensions.embedding import TabPFNEmbedding
import time

def main():

    args = DeepTuneVisionOptions(RunType.TabPFNEMBD)
    TARGET = args.target_column
    OUT = args.out
    FINETUNING_MODE = args.finetuning_mode
    TRAIN_INPUT_PATH = args.train_df
    EVAL_INPUT_PATH = args.eval_df
    MODEL_WEIGHTS = args.model_weights
    MODE = args.mode
    GROUPER = args.grouper

    train_df = pd.read_parquet(TRAIN_INPUT_PATH)
    eval_df = pd.read_parquet(EVAL_INPUT_PATH)
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_eval = eval_df.drop(columns=[TARGET])
    y_eval = eval_df[TARGET]

    if MODE == 'cls':

        get_tabpfn_embeddings(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            out=OUT,
            model_path=MODEL_WEIGHTS,
            mode=MODE,
            args = args,
            finetuning_mode=FINETUNING_MODE,
            grouper = GROUPER,
            model_str='TABPFN',
        )

    elif MODE == 'reg':

        get_tabpfn_embeddings(
            X_train=X_train,
            y_train=y_train,
            X_eval=X_eval,
            y_eval=y_eval,
            out=OUT,
            model_path=MODEL_WEIGHTS,
            mode=MODE,
            args = args,
            finetuning_mode=FINETUNING_MODE,
            grouper = GROUPER,
            model_str='TABPFN',
        )

    else:
        raise ValueError(f"Invalid mode: {MODE}. Supported modes are 'cls' and 'reg'.")


def get_tabpfn_embeddings(
    X_train,
    y_train,
    X_eval,
    y_eval,
    out,
    args,
    model_path,
    mode,
    finetuning_mode,
    model_str='TABPFN',
    grouper=None,):
    """
    Generate embeddings for the given tabular data using the provided TabPFN model.

    Args:
        model: Pre-trained TabPFN model used to generate embeddings.
        data (pd.DataFrame): Input tabular data for which embeddings are to be generated.
        batch_size (int): Number of samples to process in each batch.
    """

    if mode == 'cls':


        if finetuning_mode:
            EMBED_OUTPUT = out / f"embed_output_FINETUNED_{model_str}_embeddings_{UNIQUE_ID}"
            clf = load(Path(model_path))
        else:
            EMBED_OUTPUT = out / f"embed_output_TRAINED_{model_str}_embeddings_{UNIQUE_ID}"
            clf = load_fitted_tabpfn_model(Path(model_path),
                device=DEVICE,
            )
        EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)
        EMBED_FILE = EMBED_OUTPUT / f"{model_str}_cls_embeddings.parquet"
        start_time = time.time()
        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_eval,
            data_source="test",
        )

        embed_df = pd.DataFrame(test_embeddings[0])

        labels_df = pd.DataFrame(y_eval.reset_index(drop=True), columns=["label"])
        combined_df = pd.concat([embed_df, labels_df], axis=1)

        if grouper is not None and grouper in X_eval.columns:
            combined_df[grouper] = X_eval[grouper]

        combined_df.to_parquet(EMBED_FILE, index=False)

        end_time = time.time()
        total_time = end_time - start_time
        args.save_args(EMBED_OUTPUT)

        save_process_times(epoch_times='We only track total time for embedding', total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")
        print(f'The embeddings file is saved in {EMBED_OUTPUT}')

    if mode == 'reg':

        if finetuning_mode:
            EMBED_OUTPUT = out / f"embed_output_FINETUNED_{model_str}_embeddings_{UNIQUE_ID}"
            reg = load(Path(model_path))
        else:  
            EMBED_OUTPUT = out / f"embed_output_TRAINED_{model_str}_embeddings_{UNIQUE_ID}"
            reg = load_fitted_tabpfn_model(Path(model_path),device=DEVICE)
            

        EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)
        EMBED_FILE = EMBED_OUTPUT / f"{model_str}_reg_embeddings.parquet"
        start_time = time.time()
        embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=0)

        test_embeddings = embedding_extractor.get_embeddings(
            X_train,
            y_train,
            X_eval,
            data_source="test",
        )

        embed_df = pd.DataFrame(test_embeddings[0])

        labels_df = pd.DataFrame(y_eval.reset_index(drop=True), columns=["label"])
        combined_df = pd.concat([embed_df, labels_df], axis=1)

        if grouper is not None and grouper in X_eval.columns:
            combined_df[grouper] = X_eval[grouper].values

        combined_df.to_parquet(EMBED_FILE, index=False)

        end_time = time.time()
        total_time = end_time - start_time
        args.save_args(EMBED_OUTPUT)
        save_process_times(epoch_times='We only track total time for embedding', total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")
        print(f'The embeddings file is saved in {EMBED_OUTPUT}')

    return out, combined_df.shape
if __name__ == "__main__":
    main()



