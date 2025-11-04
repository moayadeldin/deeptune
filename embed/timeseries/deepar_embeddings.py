from pytorch_forecasting import TimeSeriesDataSet
import pytorch_forecasting
import pandas as pd
from src.timeseries.deepar import deepAR
from cli import DeepTuneVisionOptions
from utils import RunType
from options import UNIQUE_ID
import numpy as np
import torch
from pytorch_forecasting.models.deepar import DeepAR
import time 
from utils import save_process_times

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
    
    df = pd.read_parquet(DF_PATH)
    
    print(len(df))
    
    df['group'] = '0'
    
    df[TIMEINDEX_COLUMN] = pd.to_datetime(df[TIMEINDEX_COLUMN])
    df = df.sort_values(TIMEINDEX_COLUMN)
    
    time_col = df[TIMEINDEX_COLUMN]
    df["time_idx"] = ((time_col - time_col.min()).dt.total_seconds() // 3600).astype(int)
    
    GROUP_IDS = ['group'] if args.group_ids is None else args.group_ids
    
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(np.float64)
    
    dataset = TimeSeriesDataSet(
    df,
    time_idx = "time_idx",
    target=TARGET_COLUMN,
    max_prediction_length = MAX_PREDICTION_LENGTH,
    max_encoder_length = MAX_ENCODER_LENGTH,
    time_varying_known_categoricals = TIME_VARYING_KNOWN_CATEGORICALS,
    time_varying_unknown_categoricals = TIME_VARYING_UNKNOWN_CATEGORICALS,
    static_categoricals = STATIC_CATEGORICALS,
    time_varying_known_reals = TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals=(TIME_VARYING_UNKNOWN_REALS) + [TARGET_COLUMN],
    static_reals = STATIC_REALS,
    group_ids = GROUP_IDS,
    allow_missing_timesteps=True,
    target_normalizer=pytorch_forecasting.data.encoders.TorchNormalizer(),
    predict_mode=False
    )
    
    dataloader = dataset.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    
    model = DeepAR.load_from_checkpoint(MODEL_WEIGHTS)
    
    extracted_embeddings=[]
    
    cache = {"feats":None}

    handle = model.rnn.register_forward_hook(
        # LSTM returns (output, (h_n, c_n)); we want output -> [B, T_total, hidden]
        lambda m, inp, out: cache.__setitem__("feats", (out[0] if isinstance(out, tuple) else out).detach().cpu())
    )
    
    with torch.no_grad():
        for x, _ in dataloader: 
            _ = model(x)
            H =  cache['feats']
            decoder_cont = x['decoder_cont']
            decoder_target = x['decoder_target']
            time_idx = x['decoder_time_idx']
            target = x["target_scale"]
            encoder_cont = x['encoder_cont']
            encoder_target = x['encoder_target']
            
            B, T_dec, _ = H.shape
            for b in range(B):
                for t in range(T_dec):
                    row = {
                        "embedding": H[b, t].numpy(),                              
                        "time_idx": int(time_idx[b, t].item()),
                        "decoder_cont": decoder_cont[b, t].numpy().tolist(),          
                        "decoder_target": float(decoder_target[b, t].item()),
                        "encoder_cont": encoder_cont[b,t].numpy().tolist(),
                        "encoder_target":encoder_target[b,t].numpy().tolist(),
                        "target_scale": target[b].numpy().tolist(),
                    }
                        
                extracted_embeddings.append(row)
    
    handle.remove()   
    
    extracted_embeddings = np.vstack(extracted_embeddings)
    
    print(extracted_embeddings)
    print(len(extracted_embeddings))
    
    
if __name__ == "__main__":
    
    main()    
    
                
    
