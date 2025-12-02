from pytorch_forecasting import TimeSeriesDataSet
import pytorch_forecasting

import pandas as pd

from src.timeseries.deepAR import deepAR
from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID

import numpy as np
import torch
from pytorch_forecasting.models.deepar import DeepAR
import time
from utils import save_process_times
from pathlib import Path

def embed(
        eval_df: Path,
        out: Path,
        batch_size: int,
        timeindex_column: str,
        target_column: list,
        model_weights: Path,
        max_encoder_length: int = 30,
        max_prediction_length: int = 1,
        group_ids = None,
        time_varying_known_categoricals: list = [],
        time_varying_unknown_categoricals: list = [],
        static_categoricals: list = [],
        time_varying_known_reals: list = [],
        time_varying_unknown_reals: list = [],
        static_reals: list = [],
        model_str='deepAR',
        args: DeepTuneVisionOptions = None,
):
    
    df = pd.read_parquet(eval_df)
    total_rows = len(df)

    df['group'] = '0'
    
    df[timeindex_column] = pd.to_datetime(df[timeindex_column])
    df = df.sort_values(timeindex_column)
    
    time_col = df[timeindex_column]
    df["time_idx"] = ((time_col - time_col.min()).dt.total_seconds() // 3600).astype(int)
    
    df[target_column] = df[target_column].astype(np.float64)

    ckpt_path = next(Path(model_weights).glob("*.ckpt"))

    start_time = time.time()
    model = DeepAR.load_from_checkpoint(ckpt_path)
    model.eval()

    GROUP_IDS = ["group"] if group_ids is None else group_ids

    """
    Given that PyTorch Forecasting does not provide a direct way to extract one sample per row embeddings from DeepAR. Instead, it creates sequences of a certain length (max_encoder_length + max_prediction_length) and uses those sequences for training and inference.

    This means that the rows that we will have embeddings for, will correspond to the end of each decoder step, with the corresponding target value from the original dataframe, and of course (len(embeddings) < total_rows).

    Then to provide a convenient way to work around this, we will consider extracting the embeddings at each decoder time step, and for the rest of the rows that is embedded internally within the slicing mechanism of PyTorch Forecasting, we will fill those embeddings with a padding value. 
    """

    cache = {"feats":None}
    handle = model.rnn.register_forward_hook(
        lambda m, inp, out: cache.__setitem__(
            "feats", (out[0] if isinstance(out, tuple) else out).detach().cpu()
        )
    )

    embedding_dict = {}

    print(f"\n{'='*40}")
    print(f"PROCESSING EMBEDDINGS WITH 1:1 ROW MAPPING")
    print(f"{'='*40}")
    print(f"Total rows: {total_rows}")
    print(f"Max encoder length: {max_encoder_length}")
    print(f"Max prediction length: {max_prediction_length}")
    print(f"{'='*40}\n")

    df_with_history = df[df['time_idx'] >= max_encoder_length].copy()
    dataset = TimeSeriesDataSet(
            df_with_history,
            time_idx="time_idx",
            target=target_column,
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            static_categoricals=static_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=(time_varying_unknown_reals) + [target_column],
            static_reals=static_reals,
            group_ids=GROUP_IDS,
            allow_missing_timesteps=True,
            target_normalizer=pytorch_forecasting.data.encoders.TorchNormalizer(),
            predict_mode=False,
        )
    
    dataloader = dataset.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0,
    )

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            _ = model(x)

            H = cache["feats"]
            decoder_cont = x["decoder_cont"]              
            decoder_target = x["decoder_target"]          
            time_idx = x["decoder_time_idx"]              
            encoder_cont = x["encoder_cont"]              
            encoder_target = x["encoder_target"]          
            target_scale = x["target_scale"]

            B, T_dec, H_dim = H.shape

            for b in range(B):
                for t in range(T_dec):
                    if not torch.isfinite(decoder_target[b, t]):
                        continue

                    time_idx_val = int(time_idx[b, t].item())
                    emb_vec = H[b, t].numpy()
                    y_norm = decoder_target[b, t].item()
                    embedding_dict[time_idx_val] = {
                            "embedding": emb_vec,
                            "y_t": y_norm,
                            "time_idx": time_idx_val,
                            "decoder_cont": decoder_cont[b, t].numpy().tolist(),
                            "encoder_last_target": float(encoder_target[b, -1].item()),
                            "target_scale": target_scale[b].numpy().tolist(),
                            "is_padded": False
                        }

    print(f"Collected {len(embedding_dict)} embeddings from rows with history\n")

    ### know the size of the current embeddings obtained from RNN ###
    
    if embedding_dict:
        H_dim = list(embedding_dict.values())[0]['embedding'].shape[0]
    else:
        # Fallback: infer H_dim from model if no embeddings collected yet
        print("Warning: No embeddings collected yet, inferring dimension from model...")
        H_dim = model.rnn.hidden_size
    
    print(f"STEP 2: Embedding dimension determined: {H_dim}\n")

    ### embed all rows by filling missing ones with padding ###

    all_time_indices = df['time_idx'].unique()
    early_time_indices = all_time_indices[all_time_indices < max_encoder_length]

    num_padded = 0
    for time_idx_val in early_time_indices:
        if time_idx_val not in embedding_dict:
            row = df[df['time_idx'] == time_idx_val].iloc[0]
            
            # create zero-padding embedding
            padded_embedding = np.zeros(H_dim, dtype=np.float32)
            y_val = row[target_column]
            
            embedding_dict[time_idx_val] = {
                "embedding": padded_embedding,
                "y_t": y_val,
                "time_idx": time_idx_val,
                "decoder_cont": [],
                "encoder_last_target": 0.0,
                "target_scale": [0.0, 1.0],
                "is_padded": True
            }
            num_padded += 1

    print(f"Created {num_padded} padded embeddings\n")

    rows = []
    missing_count = 0
    
    # Iterate through ORIGINAL dataframe to maintain 1:1 mapping
    for idx, row in df.iterrows():
        time_idx_val = int(row['time_idx'])
        
        if time_idx_val in embedding_dict:
            # Use existing embedding
            entry = embedding_dict[time_idx_val]
            rows.append({
                "embedding": entry["embedding"],
                "y_t": entry["y_t"],
                "time_idx": time_idx_val,
                "is_padded": entry["is_padded"],
                "original_idx": idx  # Track original row index
            })
        else:
            # This shouldn't happen, but create padded embedding as fallback
            print(f"Warning: Missing embedding for time_idx={time_idx_val}, creating padded embedding")
            padded_embedding = np.zeros(H_dim, dtype=np.float32)
            rows.append({
                "embedding": padded_embedding,
                "y_t": row[target_column],
                "time_idx": time_idx_val,
                "is_padded": True,
                "original_idx": idx
            })
            missing_count += 1
    
    if missing_count > 0:
        print(f"WARNING: Created {missing_count} additional padded embeddings for missing time indices\n")


    embeddings = np.stack([row["embedding"] for row in rows], axis=0)

    y_t = np.array([row["y_t"] for row in rows], dtype=np.float32)
    
    time_idx_array = np.array([row["time_idx"] for row in rows], dtype=np.int32)
    
    is_padded = np.array([row["is_padded"] for row in rows], dtype=bool)
    
    original_idx = np.array([row["original_idx"] for row in rows], dtype=np.int32)

    # convert embeddings into columns
    emb_cols = [f"emb_{i}" for i in range(H_dim)]
    df_out = pd.DataFrame(embeddings, columns=emb_cols)

    df_out[target_column] = y_t
    # df_out["time_idx"] = time_idx_array
    df_out["is_padded"] = is_padded
    # df_out["original_idx"] = original_idx

    handle.remove()

    EMBED_OUTPUT = (out / f"embed_output_{model_str}_{UNIQUE_ID}")
    EMBED_OUTPUT.mkdir(parents=True, exist_ok=True)

    EMBED_FILE = EMBED_OUTPUT / f"{model_str}_embeddings.parquet"

    df_out.to_parquet(EMBED_FILE, index=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    args.save_args(EMBED_OUTPUT)
    
    if len(df_out) != total_rows:
        print(f"WARNING: Row count mismatch! Expected {total_rows}, got {len(df_out)}")
    else:
        pass
    
    print(f"\n{'='*40}")
    print(f"Original dataframe rows:{total_rows}")
    print(f"Output embeddings rows:{len(df_out)}")
    print(f"Padded rows:{df_out['is_padded'].sum()}")
    print(f"Non-padded rows:{(~df_out['is_padded']).sum()}")
    print(f"Embedding dimension:{H_dim}")
    print(f"{'='*40}")
    print(f"Saved to:{EMBED_FILE}")
    print(f"{'='*40}")
    
    save_process_times(epoch_times=1, total_duration=total_time, outdir=EMBED_OUTPUT, process="embedding")

    return out, df_out.shape


def main():

    args = DeepTuneVisionOptions(RunType.TIMESERIES)

    DF_PATH = args.df
    OUT = args.out
    BATCH_SIZE = args.batch_size
    TIMEINDEX_COLUMN = args.time_idx_column
    TARGET_COLUMN = args.target_column
    MAX_ENCODER_LENGTH = args.max_encoder_length
    MAX_PREDICTION_LENGTH = args.max_prediction_length
    TIME_VARYING_KNOWN_CATEGORICALS = args.time_varying_known_categoricals
    TIME_VARYING_UNKNOWN_CATEGORICALS = args.time_varying_unknown_categoricals
    STATIC_CATEGORICALS = args.static_categoricals
    TIME_VARYING_KNOWN_REALS = args.time_varying_known_reals
    TIME_VARYING_UNKNOWN_REALS = args.time_varying_unknown_reals
    STATIC_REALS = args.static_reals
    MODEL_WEIGHTS = args.model_weights
    MODEL_STR = 'deepAR'

    embed(
        eval_df=DF_PATH,
        out=OUT,
        batch_size=BATCH_SIZE,
        timeindex_column=TIMEINDEX_COLUMN,
        target_column=TARGET_COLUMN,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        time_varying_unknown_categoricals=TIME_VARYING_UNKNOWN_CATEGORICALS,
        static_categoricals=STATIC_CATEGORICALS,
        time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        static_reals=STATIC_REALS,
        model_weights=MODEL_WEIGHTS,
        model_str=MODEL_STR,
        args=args
    )

if __name__ == "__main__":
    main()