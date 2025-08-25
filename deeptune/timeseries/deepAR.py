import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss
import options
from lightning.pytorch.tuner import Tuner

from deeptune.utilities import get_args, add_datetime_column_to_predictions
from deeptune.options import DEEPTUNE_RESULTS

parser = options.parser
args = get_args()

INPUT_DIR = args.input_dir
TARGET_COLUMN = args.target_column
TIMEINDEX_COLUMN = args.time_idx_column
MAX_ENCODER_LENGTH = args.max_encoder_length
MAX_PREDICTION_LENGTH = args.max_prediction_length
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs

"""Some Assumptions:

1. The input data must contain a time index column, which is used to create the TimeSeriesDataSet. Formated in YYYY-MM-DD.

2. We assume that the data is only working on a single time series. If you have multiple time series, you need to filter the data accordingly. We are going to add a dummy column that represents the group of this single time series.

3. The target column must be numeric and should not contain any NaN values.

4. We are currently working with a single target column. If you have multiple target columns, you need to filter the data accordingly.

5. As nearly all the datasets available for the timeseries on internet are in csv format, we are working with csv files not parquet files.

6. We assume for now that all the columns are numeric. If you have categorical columns, you need to filter the data accordingly.

7. We assume for the time varying known reals is that all columns except the target column are time varying known reals.

8. The model finds its own optimum learning rate. We are not using any learning rate input for now.

"""


df = pd.read_csv(INPUT_DIR)

df[TIMEINDEX_COLUMN] = pd.to_datetime(df[TIMEINDEX_COLUMN])

df = df.sort_values(TIMEINDEX_COLUMN) 

df[TIMEINDEX_COLUMN] = df.index.astype(int)

df["group"] = "0"  # add a dummy column to represent the group of this single time series

training_cutoff = df[TIMEINDEX_COLUMN].max() - MAX_PREDICTION_LENGTH

EXCLUDED_COLS = ["group", TARGET_COLUMN] 

# Get all real columns excluding target and group
time_varying_known_reals = [
    col for col in df.columns
    if col not in EXCLUDED_COLS and col != TIMEINDEX_COLUMN and df[col].dtype in ["float64", "float32", "int64", "int32"]
]

time_varying_known_reals.append(TIMEINDEX_COLUMN)

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx.astype(int) <= training_cutoff],
    time_idx=TIMEINDEX_COLUMN,
    target=TARGET_COLUMN,  # or another target you want to forecast
    group_ids=["group"],
    categorical_encoders={"group": NaNLabelEncoder().fit(df.group)},
    static_categoricals=["group"],
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_reals=[TARGET_COLUMN],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
)

validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)


train_dataloader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=0, batch_sampler="synchronized"
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=0, batch_sampler="synchronized"
)

trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)

net_lr = DeepAR.from_dataset(
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2, # the loss here defaults to normal distribution loss
    loss=NormalDistributionLoss(),
    optimizer="Adam",
)


res = Tuner(trainer).lr_find(
    net_lr,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100,
)

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)


net = DeepAR.from_dataset(
    training,
    learning_rate=res.suggestion(),
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=NormalDistributionLoss(),
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

best_model_path = trainer.checkpoint_callback.best_model_path
print(best_model_path)
best_model = DeepAR.load_from_checkpoint(best_model_path)

pred = best_model.predict(
    val_dataloader,
    return_index=True,
    return_decoder_lengths=True,
    return_x=True,
    mode="prediction",
    trainer_kwargs=dict(accelerator="cpu"),
)

# Extract the tensors from the prediction variable
output = pred.output.detach().cpu().numpy().flatten()
decoder_time_idx = pred.x['decoder_time_idx'].detach().cpu().numpy().flatten()
decoder_target = pred.x['decoder_target'].detach().cpu().numpy().flatten()
group = pred.x['groups'].detach().cpu().numpy().flatten()[0]
target_scale = pred.x['target_scale'].detach().cpu().numpy().flatten()

pred_df = pd.DataFrame({
    "time_idx": decoder_time_idx,
    "group": [group] * len(output),
    "target": decoder_target,
    "prediction": output,
    "target_scale_loc": [target_scale[0]] * len(output),
    "target_scale_scale": [target_scale[1]] * len(output),
})

pred_df = add_datetime_column_to_predictions(
    pred_df=pred_df,
    input_csv_path=INPUT_DIR,
    time_idx_col="time_idx",
    datetime_col="date"
)

pred_df.to_csv(DEEPTUNE_RESULTS / "predictions.csv", index=True)

raw_predictions = net.predict(
    val_dataloader, mode="raw", return_x=True, n_samples=100, trainer_kwargs=dict(accelerator="cpu")
)

series = validation.x_to_index(raw_predictions.x)["group"]
n_samples = raw_predictions.output["prediction"].shape[0]

for idx in range(n_samples):  # safe range based on data
    best_model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
    plt.suptitle(f"Group: {series.iloc[idx]}")
    plt.show()