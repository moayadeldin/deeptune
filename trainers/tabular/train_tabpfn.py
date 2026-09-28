import json
import pandas as pd
import os
import time
from itertools import chain
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tabpfn import TabPFNClassifier
from torch.optim import Adam
from utils import RunType
from joblib import dump
from cli import DeepTuneVisionOptions
from tabpfn.finetuning.data_util import meta_dataset_collator
from tabpfn.finetuning.train_util import clone_model_for_evaluation
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tabpfn.model_loading import save_fitted_tabpfn_model
from sklearn.metrics import accuracy_score,log_loss
import numpy as np
from options import DEVICE
from options import UNIQUE_ID
from utils import save_process_times
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks, meta_dataset_collator
from tabpfn.architectures.interface import PerformanceOptions


os.environ["HF_TOKEN"] = "PUT_YOUR_TOKEN"

def main():

    args = DeepTuneVisionOptions(RunType.TabPFNTRAIN)
    TARGET = args.target_column
    MODE = args.mode
    OUT = args.out
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    FINETUNING_MODE: bool = args.finetuning_mode
    BATCH_SIZE: int = args.batch_size
    NUM_EPOCHS: int = args.num_epochs


    X_train = pd.read_parquet(TRAIN_PATH)
    y_train = X_train[TARGET]
    X_train = X_train.drop(columns=[TARGET])
    X_val = pd.read_parquet(VAL_PATH)
    y_val = X_val[TARGET]
    X_val = X_val.drop(columns=[TARGET])

    

    # we want to implement the logic of training and fine-tuning both here

    if FINETUNING_MODE:
        finetune_tabpfn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            mode = MODE,
            out=OUT,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            args=args,
            model_str="TABPFN",
        )
        
    else:
        train_tabpfn_from_scratch(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            mode = MODE,
            args = args,
            out=OUT,
            model_str="TABPFN"
        )



def finetune_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    out: Path,
    mode: Path,
    args : DeepTuneVisionOptions,
    batch_size: int,
    num_epochs: int,
    model_str: str = "TABPFN",
):
    start_time = time.time()
    TRAIN_OUTPUT_DIR = (out / f"trainval_output_{model_str}_finetuning_{UNIQUE_ID}")
    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if mode == 'cls':
        
        clf = TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_estimators=2,
            ignore_pretraining_limits=True,
            fit_mode="batched",
            differentiable_input=False,
        )

        training_datasets=get_preprocessed_dataset_chunks(
            calling_instance=clf,
            X_raw=X_train,
            y_raw=y_train,
            split_fn=train_test_split,
            max_data_size=10000,
            model_type="classifier",
            equal_split_size=True,
            data_shuffle_seed=42,
            preprocessing_random_state=42,
        )

        val_datasets=get_preprocessed_dataset_chunks(
            calling_instance=clf,
            X_raw=X_val,
            y_raw=y_val,
            split_fn=train_test_split,
            max_data_size=2000,
            model_type="classifier",
            equal_split_size=True,
            data_shuffle_seed=42,
            preprocessing_random_state=42,
        )


        train_loader = DataLoader(
        training_datasets, batch_size=batch_size, shuffle=True, collate_fn=meta_dataset_collator)
        
        val_loader = DataLoader(
            val_datasets, batch_size=batch_size, shuffle=False, collate_fn=meta_dataset_collator)

        optimizer = Adam(clf.model_.parameters(), lr=1e-5)

        loss_fn = torch.nn.CrossEntropyLoss()

        performance_options = PerformanceOptions()

        metrics = []  

        for epoch in range(num_epochs):
                
                train_losses = []
                train_correct = 0
                train_total = 0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]"):

                    optimizer.zero_grad()

                    clf.fit_from_preprocessed(
                        batch.X_context,
                        batch.y_context,
                        batch.cat_indices,
                        batch.configs,
                        performance_options=performance_options
                    )

                    preds = clf.forward(batch.X_query, return_logits=True)

                    targets = batch.y_query.to(DEVICE).long()

                    loss = loss_fn(preds, targets)
                    train_losses.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    predicted_classes = preds.argmax(dim=1)
                    train_correct += (predicted_classes.cpu() == batch.y_query.cpu()).sum().item()
                    train_total += batch.y_query.size(0)

                    mean_train_loss = np.mean(train_losses)
                    train_accuracy = train_correct / train_total

                    print(f"Epoch {epoch + 1} Training Loss: {mean_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

                val_losses = []
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [val]"):

                        clf.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs,performance_options=performance_options)

                        preds = clf.forward(batch.X_query, return_logits=True)
                        loss = loss_fn(preds, batch.y_query.to(clf.device))
                        val_losses.append(loss.item())

                        predicted_classes = preds.argmax(dim=1)
                        val_correct += (predicted_classes.cpu() == batch.y_query).sum().item()
                        val_total += batch.y_query.size(0)

                mean_val_loss = np.mean(val_losses)
                val_accuracy = val_correct / val_total
                print(f"Epoch {epoch + 1} Validation Loss: {mean_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

                metrics.append({
                    "epoch": epoch,
                    "train_loss": float(mean_train_loss),
                    "val_loss": float(mean_val_loss),
                    "train_accuracy": float(train_accuracy),
                    "val_accuracy": float(val_accuracy),
                })

        eval_clf = clone_model_for_evaluation(clf, {}, TabPFNClassifier)
        eval_clf.fit(X_train, y_train)
        args.save_args(TRAIN_OUTPUT_DIR)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(TRAIN_OUTPUT_DIR / "training_log.csv", index=False)
        model_path = Path(f"{TRAIN_OUTPUT_DIR}/finetuned_{model_str}_{mode}.joblib")
        dump(eval_clf, model_path)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Model saved to {model_path}")

    elif mode == 'reg':

        reg = TabPFNRegressor(
            n_estimators=2,
            ignore_pretraining_limits=True,
            fit_mode="batched",
            differentiable_input=False,
        )
            
        training_datasets=get_preprocessed_dataset_chunks(
            calling_instance=reg,
            X_raw=X_train,
            y_raw=y_train,
            split_fn=train_test_split,
            max_data_size=10000,
            model_type="regressor",
            equal_split_size=True,
            data_shuffle_seed=42,
            preprocessing_random_state=42,
        )
        val_datasets=get_preprocessed_dataset_chunks(
            calling_instance=reg,
            X_raw=X_val,
            y_raw=y_val,
            split_fn=train_test_split,
            max_data_size=2000,
            model_type="regressor",
            equal_split_size=True,
            data_shuffle_seed=42,
            preprocessing_random_state=42,
        )


        
        train_loader = DataLoader(
            training_datasets,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=meta_dataset_collator,
        )
        val_loader = DataLoader(
            val_datasets,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=meta_dataset_collator,
        )

        performance_options = PerformanceOptions()

        optimizer = Adam(reg.model_.parameters(), lr=1e-5)

        loss_fn = torch.nn.MSELoss()

        metrics = [] 
        for epoch in range(num_epochs):

            reg.model_.train()
            train_losses = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]"):
                optimizer.zero_grad()

                reg.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs,performance_options=performance_options)

                logits,_,_ = reg.forward(batch.X_query)
                targets = batch.y_query.to(DEVICE).float()

                bardist = batch.znorm_space_bardist # work on the distribution of the predictions

                preds = bardist.mean(logits) # compute the mean of the distribution as the prediction

                """"NOTES : We are working on the distribution of the predictions. We compute the mean of the distribution as the prediction and then calculate the loss based on that. This is because the TabPFN model outputs a distribution over predictions rather than a single point estimate. The mean of this distribution is used as the final prediction for the MSE loss calculation."""
                
                loss = loss_fn(preds, targets).mean()

                train_losses.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(reg.model_.parameters(), 1.0)
                optimizer.step()

            mean_train_loss = sum(train_losses)/len(train_losses)
            print(f"Epoch {epoch + 1} Training Loss: {mean_train_loss:.6f}")

            reg.model_.eval()
            with torch.no_grad():
                val_losses = []
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [val]"):

                    reg.fit_from_preprocessed(batch.X_context, batch.y_context, batch.cat_indices, batch.configs,performance_options=performance_options)

                    logits,_,_ = reg.forward(batch.X_query)
                    targets = batch.y_query.to(DEVICE).float()

                    """"NOTES : We are working on the distribution of the predictions. We compute the mean of the distribution as the prediction and then calculate the loss based on that. This is because the TabPFN model outputs a distribution over predictions rather than a single point estimate. The mean of this distribution is used as the final prediction for the MSE loss calculation."""

                    bardist = batch.znorm_space_bardist # work on the distribution of the predictions

                    preds = bardist.mean(logits) # compute the mean of the distribution as the prediction
                    val_loss = loss_fn(preds, targets).mean()
                    
                    val_losses.append(val_loss.item())

            mean_val_loss = sum(val_losses)/len(val_losses)
            print(f"Epoch {epoch + 1} Validation Loss: {mean_val_loss:.6f}")

            metrics.append({
                "epoch": epoch,
                "train_loss": float(mean_train_loss),
                "val_loss": float(mean_val_loss),
                "train_accuracy": 0,
                "val_accuracy": 0,
            })

        metrics_df = pd.DataFrame(metrics)

        reg_clf = clone_model_for_evaluation(reg, {}, TabPFNRegressor)
        reg_clf.fit(X_train, y_train)
            
        args.save_args(TRAIN_OUTPUT_DIR)
        model_path = Path(f"{TRAIN_OUTPUT_DIR}/finetuned_{model_str}_{mode}.joblib")
        dump(reg_clf, model_path)
        metrics_df.to_csv(TRAIN_OUTPUT_DIR / "training_log.csv", index=False)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Model saved to {model_path}")

    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}. Supported modes are 'cls' and 'reg'.")
    save_process_times(epoch_times="For TabPFN we only track total time", total_duration=total_time, outdir=TRAIN_OUTPUT_DIR,process='finetuning')
    
    output_dir = Path(f"{TRAIN_OUTPUT_DIR}/finetuned_{model_str}_{mode}.joblib")

    return output_dir

def train_tabpfn_from_scratch(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        out: Path,
        mode:Path,
        args : DeepTuneVisionOptions,
        model_str: str = "TABPFN"
):
    
    start_time = time.time()

    TRAIN_OUTPUT_DIR = (out / f"trainval_output_{model_str}_train_{UNIQUE_ID}")

    if mode == 'cls':

        clf = TabPFNClassifier(
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_estimators=2,
            ignore_pretraining_limits=True,
            fit_mode="batched",
            differentiable_input=False,
        )
        clf.fit(X_train, y_train)

        labels = clf.classes_

        train_proba = clf.predict_proba(X_train)
        val_proba = clf.predict_proba(X_val)

        preds_train = clf.predict(X_train)
        preds_val = clf.predict(X_val)
        
        train_acc = accuracy_score(y_train, preds_train)
        val_acc = accuracy_score(y_val, preds_val)

        train_loss = log_loss(y_train, train_proba,labels=labels)
        val_loss = log_loss(y_val, val_proba, labels=labels)
        
        print(f"Training Accuracy: {train_acc * 100.:.2f}%")
        print(f"Validation Accuracy: {val_acc * 100.:.2f}%")


        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        result_dic = [
            {
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
            }
        ]

        args.save_args(TRAIN_OUTPUT_DIR)
        with open(TRAIN_OUTPUT_DIR / "training_metrics.json", 'w') as f:
            json.dump(result_dic, f, indent=4)

        save_fitted_tabpfn_model(clf, Path(f"{TRAIN_OUTPUT_DIR}/trained_{mode}.tabpfn_fit"))

    elif mode == 'reg':

        reg = TabPFNRegressor()

        reg.fit(X_train, y_train)

        preds_train = reg.predict(X_train)
        preds_val = reg.predict(X_val)

        train_mse = mean_squared_error(y_train, preds_train)
        val_mse = mean_squared_error(y_val, preds_val)
        train_mae = mean_absolute_error(y_train, preds_train)
        val_mae = mean_absolute_error(y_val, preds_val)


        print(f"Training MSE: {train_mse:.4f}")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")

        result_dic = [
            {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
            }
        ]

        args.save_args(TRAIN_OUTPUT_DIR)
        with open(TRAIN_OUTPUT_DIR / "training_metrics.json", 'w') as f:
            json.dump(result_dic, f, indent=4)

        save_fitted_tabpfn_model(reg, Path(f"{TRAIN_OUTPUT_DIR}/trained_{mode}.tabpfn_fit"))

    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}. Supported modes are 'cls' and 'reg'.")

    end_time = time.time()
    total_time = end_time - start_time
    save_process_times(epoch_times="For TabPFN we only track total time", total_duration=total_time, outdir=TRAIN_OUTPUT_DIR, process="training")
    print(f"Model saved to {TRAIN_OUTPUT_DIR}/trained_{mode}.tabpfn_fit")

    output_dir = Path(f"{TRAIN_OUTPUT_DIR}/trained_{mode}.tabpfn_fit")

    return output_dir

if __name__ == "__main__":


    main()
    




