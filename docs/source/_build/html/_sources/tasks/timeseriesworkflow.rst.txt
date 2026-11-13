
Time-Series Workflow Components
===============================

DeepTune provides dedicated components for time-series forecasting
using DeepAR, organized into model training, evaluation, and
embedding extraction workflows, leveraging PyTorch Forecasting for probabilistic
time-series modeling and PyTorch Lightning for model optimization.
DeepAR uses a recurrent neural network (RNN) to learn temporal
dependencies from historical observations and generate forecasts
for subsequent time steps.

The workflow supports configurable encoder and prediction lengths,
time-varying and static features, and optional time-series grouping.
This page provides a high-level abstract overview of the DeepAR workflow
components and their implementation.


Model Training
--------------

**Entry point:** ``train_deepar.train``

The training workflow prepares the time-series datasets, constructs
the DeepAR architecture, and delegates model optimization to the
PyTorch Lightning ``Trainer``.

The implementation consists of a model wrapper, time-series dataset
construction, probabilistic training, validation, and checkpoint
management.


DeepAR Architecture
~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: src.timeseries.deepAR

.. py:class:: deepAR(dataset, **kwargs)

   Provides a wrapper around the PyTorch Forecasting
   ``DeepAR`` architecture.

   :param dataset: Configured PyTorch Forecasting
       ``TimeSeriesDataSet``.
   :param kwargs: Additional parameters controlling the
       DeepAR architecture and optimization.

   The wrapper initializes a native DeepAR model through
   ``DeepAR.from_dataset()``, inheriting the dataset configuration
   and the provided model hyperparameters.

   The wrapper returns the underlying PyTorch Forecasting model
   rather than implementing a separate neural network architecture.

   .. py:method:: load_from_checkpoint(path)
      :classmethod:

      Reconstructs a previously trained DeepAR model from a
      PyTorch Lightning checkpoint using
      ``DeepAR.load_from_checkpoint()``.

The current training implementation configures DeepAR with
the following parameters:

* **Hidden size:** 30 recurrent hidden units.
* **Recurrent layers:** Two RNN layers.
* **Optimizer:** Adam.
* **Learning rate:** ``1e-3`` by default.
* **Loss function:** ``NormalDistributionLoss``.
* **Training accelerator:** CPU.
* **Gradient clipping:** ``0.1``.

The model uses a probabilistic forecasting objective rather than
directly minimizing point-prediction errors such as MSE.

``NormalDistributionLoss`` models the target through a normal
distribution and optimizes the likelihood of observed target values
under the predicted distribution.


Time-Series Training Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: train_deepar

.. py:function:: train(train_df, val_df, num_epochs, out, batch_size, timeindex_column, target_column, args, max_encoder_length, max_prediction_length, time_varying_known_categoricals, time_varying_unknown_categoricals, static_categoricals, time_varying_known_reals, group_ids, time_varying_unknown_reals, static_reals, model_str, learning_rate)

   Coordinates time-series preprocessing, model initialization,
   training, validation, and checkpoint management.

   :param train_df: Path to the training Parquet dataset.
   :param val_df: Path to the validation Parquet dataset.
   :param num_epochs: Maximum number of training epochs.
   :param out: Directory for training outputs.
   :param batch_size: Number of time-series samples per batch.
   :param timeindex_column: Column representing temporal order.
   :param target_column: Target variable to forecast.
   :param args: Parsed execution configuration.
   :param max_encoder_length: Maximum number of historical
       time steps used as encoder context.
   :param max_prediction_length: Maximum number of future
       time steps predicted by the model.
   :param time_varying_known_categoricals: Categorical variables
       known across the relevant forecasting horizon.
   :param time_varying_unknown_categoricals: Categorical variables
       whose future values are not assumed to be known.
   :param static_categoricals: Categorical attributes that remain
       constant within a time series.
   :param time_varying_known_reals: Numerical variables whose
       values are available across the forecasting horizon.
   :param time_varying_unknown_reals: Numerical variables whose
       future values are not assumed to be known.
   :param static_reals: Numerical attributes that remain constant
       within a time series.
   :param group_ids: Optional identifiers distinguishing
       independent time-series groups.
   :param model_str: Identifier used for the output directory.
   :param learning_rate: Learning rate supplied to DeepAR.

   The function reads the training and validation datasets,
   prepares the temporal observations, constructs the appropriate
   DataLoaders, and initializes the DeepAR model. Model optimization is performed using the PyTorch Lightning
   ``Trainer``, with validation and checkpointing integrated
   into the training procedure, and returns the directory containing the training outputs.


Time-Series Dataset Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training function loads the training and validation datasets
from Parquet files. Each dataset is assigned a temporary split identifier before
the datasets are concatenated and processed through
``prepare_timeseries()``. This helper prepares the time-index and grouping information
required by the downstream time-series workflow. Following preprocessing, the training and validation subsets
are recovered using their split identifiers.

The training dataset is constructed using PyTorch Forecasting's
``TimeSeriesDataSet`` with the following configuration:

* ``time_idx`` identifies the temporal index.
* ``target`` identifies the forecasting target.
* ``max_encoder_length`` controls the available historical context.
* ``max_prediction_length`` controls the prediction horizon.
* ``group_ids`` identifies the independent time series.
* ``allow_missing_timesteps=True`` enables handling of missing
  time-index steps through the dataset implementation.
* ``TorchNormalizer`` provides target normalization.

The target variable is additionally included among the
time-varying unknown real-valued features, allowing the model
to use historical target observations. The resulting training dataset is converted into a DataLoader
using the configured batch size and synchronized batch sampling.


Validation Context Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepAR requires historical observations to initialize its encoder
before generating forecasts. Consequently, validation cannot always be performed using
the validation observations in isolation.

The training implementation retrieves the final
``max_encoder_length`` observations from the training dataset
for each time-series group. These observations are concatenated with the validation data
and sorted according to the group identifiers and time index.

The validation dataset is then constructed through
``TimeSeriesDataSet.from_dataset()`` using the training dataset
as its reference, preserving the training dataset configuration
and preprocessing parameters while providing historical context
for validation predictions.

The validation dataset is configured with:

* ``predict=True`` to construct prediction-oriented windows.
* ``stop_randomization=True`` to disable random window selection.

Historical training observations supply encoder context,
while validation observations are used for forecasting
and model selection.


Model Optimization and Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The training workflow initializes the PyTorch Lightning
``Trainer`` with the configured number of epochs and
CPU execution. The trainer uses gradient clipping with a maximum value of
``0.1`` and a checkpoint callback monitoring validation loss, and the model optimization is executed through
``Trainer.fit()``, using the training and validation DataLoaders.

A ``ModelCheckpoint`` callback monitors ``val_loss``
and retains the checkpoint corresponding to the lowest
observed validation loss.

The checkpoint filename follows the pattern::

   deepar-<epoch>-<val_loss>.ckpt

Training progress is additionally monitored through a
custom callback.

.. py:currentmodule:: train_deepar

.. py:class:: _LossHistory(destination)

   Implements a PyTorch Lightning callback for recording
   epoch-level training and validation losses.

   :param destination: Directory for saving training metrics.

   .. py:method:: on_train_epoch_end(trainer, pl_module)

      Retrieves available training and validation metrics
      from the Lightning trainer.

      Records the current epoch, training loss, and
      validation loss when available.

      The accumulated metrics are saved to
      ``training_log.csv``.

After training, the best-performing checkpoint is loaded
using ``deepAR.load_from_checkpoint()``. The selected model generates validation predictions through
``predict()``. The implementation also invokes the MAE metric on validation
predictions and targets, although this value is not separately
written to a dedicated metrics file.

The validation prediction outputs are saved through
``save_timeseries_prediction_to_json()``. Training logs are additionally recorded through
``TensorBoardLogger``, while execution configuration and
processing-time information are saved to the output directory.


Model Evaluation
----------------

**Entry point:** ``evaluate_deepar.evaluate``

The evaluation workflow loads the selected DeepAR checkpoint
and performs forecasting on a held-out time-series dataset.

Unlike the training workflow, evaluation does not update
model parameters. The implementation reconstructs the evaluation dataset using
the preprocessing parameters stored in the trained model
and incorporates historical observations to supply the
required encoder context.


Evaluation Function
~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: evaluate_deepar

.. py:function:: evaluate(train_df_path, val_df_path, eval_df_path, out, batch_size, timeindex_column, target_column, model_weights, max_encoder_length, max_prediction_length, group_ids, time_varying_known_categoricals, time_varying_unknown_categoricals, static_categoricals, time_varying_known_reals, time_varying_unknown_reals, static_reals, args)

   Evaluates a trained DeepAR model using held-out
   time-series observations.

   :param train_df_path: Path to the training Parquet dataset.
   :param val_df_path: Path to the validation Parquet dataset.
   :param eval_df_path: Path to the evaluation Parquet dataset.
   :param out: Directory for evaluation outputs.
   :param batch_size: Batch size used during inference.
   :param timeindex_column: Column representing temporal order.
   :param target_column: Target variable used for forecasting.
   :param model_weights: Directory containing the trained
       DeepAR checkpoint.
   :param max_encoder_length: Number of preceding observations
       retained as historical context per group.
   :param max_prediction_length: Configured prediction horizon.
   :param group_ids: Optional time-series grouping identifiers.
   :param time_varying_known_categoricals: Configuration of
       known time-varying categorical features.
   :param time_varying_unknown_categoricals: Configuration of
       unknown time-varying categorical features.
   :param static_categoricals: Static categorical features.
   :param time_varying_known_reals: Known time-varying
       numerical features.
   :param time_varying_unknown_reals: Unknown time-varying
       numerical features.
   :param static_reals: Static numerical features.
   :param args: Parsed execution configuration.

   The function loads the specified checkpoint, constructs
   the evaluation dataset, generates forecasts, and calculates
   prediction-error metrics.

   Returns the evaluation output directory and a NumPy array
   containing the generated predictions.


Evaluation Dataset Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function loads the training, validation, and evaluation
datasets and assigns identifiers to distinguish their
original splits. The datasets are concatenated and processed through
``prepare_timeseries()`` before recovering the separate
training, validation, and evaluation subsets.

The evaluation workflow uses historical observations from
both the training and validation datasets. For each time-series group, the function selects the final
``max_encoder_length`` historical observations and combines
them with the evaluation dataset.

The resulting dataset is sorted by group identifiers and
time index to preserve temporal ordering.

The evaluation ``TimeSeriesDataSet`` is reconstructed using
``TimeSeriesDataSet.from_parameters()`` and the
``dataset_parameters`` stored in the trained checkpoint. This preserves the trained model's preprocessing configuration,
including its encoders and normalization parameters.

The evaluation dataset uses:

* ``predict=False`` for constructing evaluation windows.
* ``stop_randomization=True`` for deterministic window selection.
* ``min_prediction_idx`` set to the earliest evaluation time
  index to restrict the prediction starting point.

The resulting dataset is converted into an evaluation
DataLoader without training-time shuffling.


Forecast Generation and Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The evaluation workflow generates predictions using
``DeepAR.predict()`` with ``mode="prediction"``.

The prediction call also requests:

* Input tensors through ``return_x=True``.
* Ground-truth targets through ``return_y=True``.
* Time-series indices through ``return_index=True``.
* Decoder sequence lengths through
  ``return_decoder_lengths=True``.

The returned prediction values are compared with the
corresponding ground-truth targets. The implementation calculates the following metrics:

* **MAE:** Mean Absolute Error between predictions and targets.
* **RMSE:** Root Mean Squared Error.
* **Loss:** Mean Squared Error (MSE).
* **Predictions:** Total number of evaluated prediction values.

The metrics are calculated over the returned prediction
elements and are not separately aggregated by time-series group.

The evaluation ``loss`` field represents MSE, rather than
the probabilistic ``NormalDistributionLoss`` used during training. The final metrics are saved to ``full_metrics.json``. Forecast outputs are additionally serialized using
``save_timeseries_prediction_to_json()``.

The workflow saves the execution configuration and
processing-time information alongside the evaluation outputs.


Embedding Extraction
--------------------

**Entry point:** ``deepAR_embeddings.embed``

The embedding workflow extracts numerical representations
of temporal information from a trained DeepAR model. Unlike the evaluation workflow, which focuses on forecast
generation, embedding extraction retrieves intermediate
representations from the recurrent encoder.

These representations summarize historical time-series
information and can be used as features for subsequent
machine learning tasks.


Embedding Extraction Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: deepAR_embeddings

.. py:function:: embed(eval_df, out, batch_size, timeindex_column, target_column, model_weights, max_encoder_length, max_prediction_length, group_ids, time_varying_known_categoricals, time_varying_unknown_categoricals, static_categoricals, time_varying_known_reals, time_varying_unknown_reals, static_reals, model_str="deepAR", args, history_df_paths)

   Extracts temporal feature representations from the
   recurrent encoder of a trained DeepAR model.

   :param eval_df: Path to the input Parquet dataset.
   :param out: Directory for embedding outputs.
   :param batch_size: Number of samples processed per batch.
   :param timeindex_column: Column representing temporal order.
   :param target_column: Target column or columns retained
       in the embedding output.
   :param model_weights: Path to a DeepAR checkpoint file
       or a directory containing checkpoint files.
   :param max_encoder_length: Configured encoder length.
   :param max_prediction_length: Configured prediction horizon.
   :param group_ids: Optional time-series grouping identifiers.
   :param time_varying_known_categoricals: Known time-varying
       categorical feature configuration.
   :param time_varying_unknown_categoricals: Unknown time-varying
       categorical feature configuration.
   :param static_categoricals: Static categorical features.
   :param time_varying_known_reals: Known numerical
       time-varying feature configuration.
   :param time_varying_unknown_reals: Unknown numerical
       time-varying feature configuration.
   :param static_reals: Static numerical feature configuration.
   :param model_str: Model identifier used for output naming.
   :param args: Optional execution configuration.
   :param history_df_paths: Optional list of historical Parquet
       datasets providing additional encoder context.

   The function loads the trained DeepAR model, reconstructs
   the time-series dataset, and extracts encoder representations
   without gradient computation, and returns the embedding output directory and the dimensions
   of the generated embedding DataFrame.


Historical Context Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The embedding workflow first loads the evaluation dataset
and verifies that it contains observations. Each evaluation row is assigned an internal identifier
through the ``__embedding_row`` column.

When ``history_df_paths`` is provided, the function loads
the specified historical datasets and concatenates them
with the evaluation observations. Historical observations receive an identifier of ``-1``
to distinguish them from the rows requiring embeddings.

The combined dataset is processed using
``prepare_timeseries()``. After preprocessing, the workflow selects the evaluation
rows and restores their original ordering using
``__embedding_row``.

The historical observations are used as encoder context
but are not included as separate rows in the final
embedding DataFrame.

The time-series dataset is reconstructed through
``TimeSeriesDataSet.from_parameters()`` using the
preprocessing configuration stored in the trained checkpoint. This ensures that the embedding workflow reuses the
training-time encoders, feature transformations, and
normalization parameters instead of fitting new ones
on evaluation data.

The reconstructed dataset uses ``predict=False``,
``stop_randomization=True``, and a minimum encoder
length of two observations. The minimum prediction index is set to the earliest
time index among the evaluation rows.


Encoder Representation Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The extraction procedure runs the trained model in
evaluation mode with gradient computation disabled.

For each batch, the workflow invokes:

.. code-block:: python

   hidden = model.encode(x)

The returned encoder representation is processed to
obtain the final hidden-state tensor used by the
implementation.

The selected hidden-state representation is converted
to a NumPy array and associated with the corresponding
time-series group and decoder time indices. These group-time associations are recovered using
``TimeSeriesDataSet.x_to_index()`` and the decoder
time-index information in each batch.

The implementation assigns the selected encoder vector
to the decoder time steps associated with its window,
rather than independently computing an embedding
from each individual target observation.

The embedding dimension is determined by the
trained model's recurrent hidden size. For the current training configuration, the hidden
size is ``30``, producing 30-dimensional embedding
vectors.


Embedding Alignment and Missing Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After extracting the available encoder representations,
the workflow aligns the embeddings with the original
evaluation rows. Each evaluation observation is matched using its
time-series group identifiers and time index.

If an embedding is available, the corresponding
encoder representation is assigned to the row. If no matching embedding is available, the workflow
creates a zero-filled vector with the same dimension
as the model's recurrent hidden size.

An ``is_padded`` column records whether an embedding
was unavailable and replaced with zeros.

This mechanism preserves the number and original
ordering of evaluation rows. A padded embedding indicates that no matching encoder
representation was produced for that observation,
which may occur when the available temporal context
is insufficient for constructing a valid window.

The final embedding DataFrame contains:

* ``emb_0``, ``emb_1``, ..., ``emb_<N-1>``:
  Extracted temporal feature representations.
* Target column or columns corresponding to the
  original observations.
* ``is_padded``: Boolean indicator of missing embeddings.
* Optional time-series group identifiers.

The embedding DataFrame is saved to
``deepAR_embeddings.parquet``.

The workflow also records the total extraction time
and saves the execution configuration when provided.


Implementation Notes
--------------------

Several implementation details are important when configuring
and interpreting the DeepAR workflows.

**Training and evaluation objectives**

Training uses ``NormalDistributionLoss`` to optimize a
probabilistic forecasting model, whereas the held-out
evaluation function calculates point-prediction errors
using MAE, RMSE, and MSE.

**Temporal context**

Training validation and held-out evaluation explicitly
include earlier observations to provide encoder context.
Embedding extraction can optionally receive additional
historical datasets through ``history_df_paths``.

**Checkpoint selection**

Only the best checkpoint according to validation loss
is retained by the configured ``ModelCheckpoint`` callback.
Evaluation and embedding extraction load saved checkpoints
rather than retraining the model.

**Saved preprocessing parameters**

Evaluation and embedding extraction reuse the dataset
parameters stored in the checkpoint. The feature-category
parameters in their function signatures are not used to
rebuild independent feature encoders.

**Configuration differences**

The training function defaults to
``max_encoder_length=8``, while evaluation and embedding
extraction default to ``max_encoder_length=30``.

During evaluation, the latter controls how many historical
rows are explicitly selected, while the reconstructed
dataset also inherits the trained model's parameters.

In the current embedding implementation,
``max_encoder_length`` and ``max_prediction_length`` are
accepted as arguments but are not directly supplied to
``TimeSeriesDataSet.from_parameters()``.

**Embedding availability**

The embedding workflow uses a group-and-time-index lookup
to align extracted representations with evaluation rows.
When multiple windows produce a representation for the
same group and time index, the later assignment replaces
the earlier one.

The ``is_padded`` column should therefore be considered
when using the resulting embeddings in downstream models.
