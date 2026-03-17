import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch,call
from trainers.vision.train import train

@patch('trainers.vision.train.ParquetImageDataset')
@patch('trainers.vision.train.get_model_cls')
@patch('trainers.vision.train.DataLoader')
@patch('trainers.vision.train.Trainer')
@patch('trainers.vision.train.set_seed')

class TestVisionTrainer:

    
    
    def test_vision_train_with_normal_parameters(
        self,mock_set_seed,mock_trainer_class,mock_get_cls,mock_dataloader,mock_dataset
    ):
        
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        mock_model = MagicMock()
        mock_get_cls.return_value = mock_model
        
        mock_dataset.from_parquet.return_value = MagicMock()
        
        # create mock args
        
        mock_args = MagicMock()
        mock_args.model_architecture = "resnet"
        
        train_path = Path("/fake/train.parquet")
        val_path = Path("/fake/val.parquet")
        out_path = Path("/fake/output")
        
        # call function 
        
        result = train(
            train_df = train_path,
            val_df = val_path,
            out = out_path,
            freeze_backbone = False,
            use_peft = False,
            fixed_seed=0,
            mode="classification",
            model_version="resnet50",
            batch_size=32,
            num_epochs=10,
            learning_rate=0.001,
            added_layers=2,
            num_classes=10,
            embed_size=512,
            model_str="resnet50",
            args=mock_args
        )
        
        assert result is not None
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.validate.assert_called_once()
        mock_args.save_args.assert_called_once()
        
        
    def test_vision_trainer_with_peft_enabled(
            self,
            mock_set_seed,
            mock_trainer_class,
            mock_dataloader,
            mock_get_cls,
            mock_dataset,
        ):
        
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        mock_model = MagicMock()
        mock_get_cls.return_value = mock_model
        
        mock_dataset.from_parquet.return_value = MagicMock()
        
        # create mock args
        
        mock_args = MagicMock()
        mock_args.model_architecture = "resnet"
        
        train_path = Path("/fake/train.parquet")
        val_path = Path("/fake/val.parquet")
        out_path = Path("/fake/output")
        
        # call function 
        
        result = train(
            train_df = train_path,
            val_df = val_path,
            out = out_path,
            freeze_backbone = False,
            use_peft = True,
            fixed_seed=0,
            mode="classification",
            model_version="resnet50",
            batch_size=32,
            num_epochs=10,
            learning_rate=0.001,
            added_layers=2,
            num_classes=10,
            embed_size=512,
            model_str="resnet50",
            args=mock_args
        )
        
        assert result is not None
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.validate.assert_called_once()
        mock_args.save_args.assert_called_once()
        
    def test_vision_trainer_raises_error_when_added_layers_is_zero(
            self,
            mock_set_seed,
            mock_trainer_class,
            mock_dataloader,
            mock_get_cls,
            mock_dataset,
        ):
        
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        mock_model = MagicMock()
        mock_get_cls.return_value = mock_model
        
        mock_dataset.from_parquet.return_value = MagicMock()
        
        # create mock args
        
        mock_args = MagicMock()
        mock_args.model_architecture = "resnet"
        
        train_path = Path("/fake/train.parquet")
        val_path = Path("/fake/val.parquet")
        out_path = Path("/fake/output")
        
        with pytest.raises(ValueError, match=r"please choose 1 or 2 as your preferred number of added_layers"):
            train(
                train_df = train_path,
                val_df = val_path,
                out = out_path,
                freeze_backbone = False,
                use_peft = True,
                fixed_seed=0,
                mode="classification",
                model_version="resnet50",
                batch_size=32,
                num_epochs=10,
                learning_rate=0.001,
                added_layers=0,  # invalid value
                num_classes=10,
                embed_size=512,
                model_str="resnet50",
                args=mock_args
            )
            
    @patch('trainers.vision.train.train_siglip')
    def test_vision_trainer_train_siglip_model(self, mock_train_siglip, mock_set_seed, mock_trainer_class, mock_dataloader, mock_get_cls, mock_dataset):
        
        mock_trainer_instance = MagicMock()
        mock_train_siglip.return_value = mock_trainer_instance
        
        # create mock args
        
        mock_args = MagicMock()
        mock_args.model_architecture = "siglip"
        
        train_path = Path("/fake/train.parquet")
        val_path = Path("/fake/val.parquet")
        out_path = Path("/fake/output")
        
        result = train(
            train_df = train_path,
            val_df = val_path,
            out = out_path,
            freeze_backbone = False,
            use_peft = False,
            fixed_seed=0,
            mode="classification",
            model_version="siglip",
            batch_size=32,
            num_epochs=10,
            learning_rate=0.001,
            added_layers=2,
            num_classes=10,
            embed_size=512,
            model_str="siglip",
            args=mock_args
        )
        
        assert result is not None
        mock_train_siglip.assert_called_once()
        mock_args.save_args.assert_called_once()

        
        
        
        
        
        