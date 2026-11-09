
Time-Series Workflow Components
===============================

DeepTune provides dedicated components for time-series forecasting
using DeepAR, organized into model training, evaluation, and embedding
extraction workflows. These components leverage PyTorch Forecasting
for probabilistic forecasting and temporal feature representation.


Model Training
--------------

**Entry point:** ``train_deepar.train``

The training function prepares time-series datasets, configures
the DeepAR architecture, and delegates model optimization to
the PyTorch Lightning ``Trainer``.

.. py:currentmodule:: src.timeseries.deepAR

.. py:class:: deepAR(dataset, **kwargs)

   Wraps the PyTorch Forecasting DeepAR architecture.

   :param dataset: Configured ``TimeSeriesDataSet``.
   :param kwargs: Additional DeepAR model configuration parameters.

   The wrapper initializes the model using
   ``DeepAR.from_dataset`` with the specified configuration.

   .. py:method:: load_from_checkpoint(path)
      :classmethod:

      Loads a trained DeepAR model from a checkpoint.


The training workflow uses an RNN-based DeepAR architecture
with two recurrent layers, a hidden size of 30, and
``NormalDistributionLoss`` for probabilistic forecasting.

.. py:currentmodule:: train_deepar

.. py:function:: train(train_df, val_df, num_epochs, out, batch_size, timeindex_column, target_column, args, max_encoder_length=8, max_prediction_length=1, time_varying_known_categoricals=[], time_varying_unknown_categoricals=[], static_categoricals=[], time_varying_known_reals=[], group_ids=None, time_varying_unknown_reals=[], static_reals=None, model_str="DeepAR", learning_rate=1e-3)

   Prepares training and validation DataLoaders using
   ``TimeSeriesDataSet`` and optimizes the model with Adam.

   The validation dataset includes historical observations
   from the training set to provide the required encoder context.

   Training and validation losses are recorded in
   ``training_log.csv``.

   The best-performing checkpoint is selected according to
   validation loss and saved in ``.ckpt`` format.


Model Evaluation
----------------

**Entry point:** ``evaluate_deepar.evaluate``

The evaluation function reconstructs the time-series dataset
using the trained model's dataset parameters and performs
forecasting on held-out observations.

.. py:currentmodule:: evaluate_deepar

.. py:function:: evaluate(train_df_path, val_df_path, eval_df_path, out, batch_size, timeindex_column, target_column, model_weights, max_encoder_length=30, max_prediction_length=1, group_ids=None, time_varying_known_categoricals=[], time_varying_unknown_categoricals=[], static_categoricals=[], time_varying_known_reals=[], time_varying_unknown_reals=[], static_reals=[], args=None)

   Loads the trained DeepAR checkpoint and generates forecasts
   using historical observations from the training and validation
   datasets as encoder context.

   Computes the following evaluation metrics:

   * **MAE:** Mean Absolute Error.
   * **RMSE:** Root Mean Squared Error.
   * **Loss:** Mean Squared Error (MSE).
   * **Predictions:** Number of evaluated prediction values.

   Evaluation metrics are saved to ``full_metrics.json``,
   alongside the forecasting outputs.


Embedding Extraction
--------------------

**Entry point:** ``deepAR_embeddings.embed``

The embedding workflow loads a trained DeepAR model and
extracts temporal feature representations from its encoder.

.. py:currentmodule:: deepAR_embeddings

.. py:function:: embed(eval_df, out, batch_size, timeindex_column, target_column, model_weights, max_encoder_length=30, max_prediction_length=1, group_ids=None, time_varying_known_categoricals=[], time_varying_unknown_categoricals=[], static_categoricals=[], time_varying_known_reals=[], time_varying_unknown_reals=[], static_reals=[], model_str="deepAR", args=None, history_df_paths=None)

   Extracts numerical representations from time-series data
   using the trained DeepAR encoder.

   :param eval_df: Path to the evaluation dataset.
   :param out: Directory for embedding outputs.
   :param batch_size: Batch size for embedding extraction.
   :param timeindex_column: Column identifying temporal order.
   :param target_column: Target variable for forecasting.
   :param model_weights: Path to the trained model checkpoint.
   :param history_df_paths: Optional historical datasets used
       to provide encoder context.

   The workflow extracts the encoder's final hidden-state
   representations and associates them with evaluation samples.

   Samples without sufficient historical context receive
   zero-filled embeddings and are identified through the
   ``is_padded`` column.

   The resulting DataFrame contains extracted embeddings,
   target values, optional group identifiers, and padding flags.

   Embeddings are saved to ``deepAR_embeddings.parquet``.
