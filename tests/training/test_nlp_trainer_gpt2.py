import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import torch
import torch.nn as nn
from trainers.nlp.train_gpt2 import train, GPTrainer


@patch('trainers.nlp.train_gpt2.GPTrainer')
@patch('trainers.nlp.train_gpt2.TextDataset')
@patch('trainers.nlp.train_gpt2.AdjustedGPT2Model')
@patch('trainers.nlp.train_gpt2.load_gpt2_model_offline')
@patch('trainers.nlp.train_gpt2.set_seed')

class TestGPT2Trainer:
    
    def test_gpt2train_with_valid_parameters(
        self,
        mock_set_seed,
        mock_load_model,
        mock_adjusted_model_cls,
        mock_dataset,
        mock_trainer_cls
    ):
        
        mock_gpt_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load_model.return_value = (mock_gpt_model, mock_tokenizer)
        
        mock_adjusted_model_instance = MagicMock()
        mock_adjusted_model_cls.return_value = mock_adjusted_model_instance
        
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__.return_value = 10
        mock_dataset.return_value = mock_dataset_instance
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance
        mock_trainer_instance.save_tunedgpt2model.return_value = Path("/fake/output")
        
        # Create mock args
        mock_args = MagicMock()
        
        # Call function
        result = train(
            train_df=Path("/fake/train.parquet"),
            val_df=Path("/fake/val.parquet"),
            out=Path("/fake/output"),
            freeze_backbone=False,
            use_peft=False,
            fixed_seed=42,
            batch_size=16,
            num_epochs=3,
            learning_rate=1e-4,
            model_str="GPT2",
            args=mock_args
        )
        
        # Assertions
        assert result == Path("/fake/output")
        mock_set_seed.assert_called_once_with(42)
        mock_load_model.assert_called_once()
        mock_adjusted_model_cls.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_tunedgpt2model.assert_called_once()
