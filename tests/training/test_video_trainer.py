import pytest
from unittest.mock import MagicMock, patch
from trainers.video.train import train


@patch('trainers.video.train.ParquetVideoDataset')
@patch('trainers.video.train.build_video_model')
@patch('trainers.video.train.DataLoader')
@patch('trainers.video.train.Trainer')
@patch('trainers.video.train.set_seed')
class TestVideoTrainer:

    def test_video_train_with_normal_parameters(
        self, mock_set_seed, mock_trainer_class, mock_dataloader, mock_build_model, mock_dataset, tmp_path
    ):

        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        mock_dataset.from_parquet.return_value = MagicMock()

        mock_args = MagicMock()

        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        out_path = tmp_path / "output"

        result = train(
            train_df=train_path,
            val_df=val_path,
            out=out_path,
            freeze_backbone=False,
            use_peft=False,
            fixed_seed=0,
            mode="cls",
            model_version="resnet50",
            batch_size=32,
            num_epochs=10,
            learning_rate=0.001,
            added_layers=2,
            num_classes=10,
            embed_size=512,
            model_str="resnet50_video",
            args=mock_args,
            num_frames=8,
            pooling="mean",
        )

        assert result is not None
        mock_build_model.assert_called_once_with(
            model_version="resnet50",
            num_classes=10,
            added_layers=2,
            embed_size=512,
            freeze_backbone=False,
            mode="cls",
            use_peft=False,
            pooling="mean",
        )
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.validate.assert_called_once()
        mock_args.save_args.assert_called_once()

    def test_video_trainer_with_peft_and_native_architecture(
        self, mock_set_seed, mock_trainer_class, mock_dataloader, mock_build_model, mock_dataset, tmp_path
    ):

        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance

        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        mock_dataset.from_parquet.return_value = MagicMock()

        mock_args = MagicMock()

        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        out_path = tmp_path / "output"

        result = train(
            train_df=train_path,
            val_df=val_path,
            out=out_path,
            freeze_backbone=True,
            use_peft=True,
            fixed_seed=0,
            mode="cls",
            model_version="r3d_18",
            batch_size=4,
            num_epochs=5,
            learning_rate=0.001,
            added_layers=1,
            num_classes=3,
            embed_size=256,
            model_str="r3d18_video",
            args=mock_args,
            num_frames=16,
            pooling="attention",
        )

        assert result is not None
        mock_build_model.assert_called_once_with(
            model_version="r3d_18",
            num_classes=3,
            added_layers=1,
            embed_size=256,
            freeze_backbone=True,
            mode="cls",
            use_peft=True,
            pooling="attention",
        )
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.validate.assert_called_once()

    def test_video_trainer_raises_error_when_added_layers_is_zero(
        self, mock_set_seed, mock_trainer_class, mock_dataloader, mock_build_model, mock_dataset, tmp_path
    ):

        mock_args = MagicMock()

        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        out_path = tmp_path / "output"

        with pytest.raises(ValueError, match=r"please choose 1 or 2 as your preferred number of added_layers"):
            train(
                train_df=train_path,
                val_df=val_path,
                out=out_path,
                freeze_backbone=False,
                use_peft=False,
                fixed_seed=0,
                mode="cls",
                model_version="resnet50",
                batch_size=32,
                num_epochs=10,
                learning_rate=0.001,
                added_layers=0,  # invalid value
                num_classes=10,
                embed_size=512,
                model_str="resnet50_video",
                args=mock_args,
            )
