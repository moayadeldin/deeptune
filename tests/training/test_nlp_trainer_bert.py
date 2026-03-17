from pathlib import Path
from unittest.mock import MagicMock, patch
from trainers.nlp.train_multilingualbert import get_model, train

@patch("trainers.nlp.train_multilingualbert.BERTrainer")
@patch("trainers.nlp.train_multilingualbert.TextDataset")
@patch("trainers.nlp.train_multilingualbert.torch.utils.data.DataLoader")
@patch("trainers.nlp.train_multilingualbert.load_nlp_bert_ml_model_offline")
@patch("trainers.nlp.train_multilingualbert.get_model")
@patch("trainers.nlp.train_multilingualbert.set_seed")


class TestBERTTrainer:
	def test_berttrain_with_valid_parameters(
		self,
		mock_set_seed,
		mock_get_model,
		mock_load_bert,
		mock_dataloader,
		mock_dataset,
		mock_trainer_cls,
        tmp_path
	):
		model_factory = MagicMock()
		model_instance = MagicMock()
		model_factory.return_value = model_instance
		mock_get_model.return_value = model_factory

		tokenizer = MagicMock()
		mock_load_bert.return_value = (MagicMock(), tokenizer)

		mock_dataset.return_value = MagicMock()
		mock_dataloader.return_value = MagicMock()

		trainer_instance = MagicMock()
		expected_output = tmp_path / "output"
		trainer_instance.save_tunedbertmodel.return_value = expected_output
		mock_trainer_cls.return_value = trainer_instance

		mock_args = MagicMock()

		result = train(
			train_df=tmp_path / "train.parquet",
			val_df=tmp_path / "val.parquet",
			out=tmp_path / "output",
			freeze_backbone=False,
			use_peft=False,
			fixed_seed=123,
			batch_size=16,
			num_epochs=2,
			learning_rate=1e-4,
			added_layers=2,
			num_classes=3,
			embed_size=128,
			model_str="BERT",
			args=mock_args,
		)

		assert result is not None
		trainer_instance.train.assert_called_once()
		trainer_instance.save_tunedbertmodel.assert_called_once()
		mock_args.save_args.assert_called_once_with(expected_output)
