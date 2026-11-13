
Tabular Workflow Components
===========================

DeepTune provides dedicated components for tabular classification
and regression using GANDALF and TabPFN. These components support
model training, evaluation, and embedding extraction through
separate workflows.


GANDALF uses the PyTorch Tabular framework for
model training, while TabPFN supports both fine-tuning
and in-context learning using pretrained models.

This page provides a high-level abstract overview of the tabular
workflow components and their implementation.


Model Training
--------------

The training workflow is implemented separately for GANDALF
and TabPFN. Both workflows accept tabular datasets stored in
Parquet format and support classification and regression.


GANDALF
~~~~~~~

**Entry point:** ``train_gandalf.train``

The GANDALF training workflow prepares the training and validation
datasets, configures the model architecture, and delegates optimization
to the PyTorch Tabular ``TabularModel`` framework.

.. py:currentmodule:: src.tabular.gandalf

.. py:class:: GANDALF(data_config, optimizer_config, trainer_config, task, gflu_stages, gflu_feature_init_sparsity, gflu_dropout, learning_rate)

   Configures the GANDALF architecture by extending the PyTorch
   Tabular ``GANDALFConfig`` class.

   :param data_config: Configuration of input features and targets.
   :param optimizer_config: Optimizer configuration.
   :param trainer_config: Training configuration.
   :param task: Classification or regression task.
   :param gflu_stages: Number of GFLU stages.
   :param gflu_feature_init_sparsity: Initial feature sparsity.
   :param gflu_dropout: Dropout rate.
   :param learning_rate: Model learning rate.

   The configuration specifies the Gated Feature Learning Unit
   (GFLU) architecture and its training hyperparameters.


.. py:currentmodule:: train_gandalf

.. py:function:: train(train_df, val_df, out, type, batch_size, num_epochs, learning_rate, gflu_stages, target, continuous_cols, categorical_cols, model_str, args)

   Trains a GANDALF model using the PyTorch Tabular framework.

   :param train_df: Path to the training Parquet dataset.
   :param val_df: Path to the validation Parquet dataset.
   :param out: Directory for training outputs.
   :param type: Model task type, either ``classification`` or ``regression``.
   :param batch_size: Number of samples per training batch.
   :param num_epochs: Maximum number of training epochs.
   :param learning_rate: Model learning rate.
   :param gflu_stages: Number of GFLU stages.
   :param target: Name of the target column.
   :param continuous_cols: Names of continuous input features.
   :param categorical_cols: Names of categorical input features.
   :param model_str: Model identifier.
   :param args: Parsed configuration arguments.

   The function constructs ``DataConfig``, ``OptimizerConfig``,
   and ``TrainerConfig`` objects before initializing the
   ``TabularModel`` with the GANDALF configuration.

   Model training is performed through ``TabularModel.fit()``,
   using the training and validation datasets.

   A ``PerformanceLoggerCallback`` records training progress,
   and the fitted model is saved under ``GANDALF_model``.

   Training metrics are saved to ``training_log.csv``.


TabPFN
~~~~~~

**Entry points:** ``train_tabpfn.finetune_tabpfn`` and
``train_tabpfn.train_tabpfn_in_context_learning``

The TabPFN training workflow supports two distinct approaches:
gradient-based fine-tuning and in-context learning.

The ``train_tabpfn.main`` function handles argument parsing and
dispatches execution to the appropriate function according to
the ``finetuning_mode`` configuration.

Both approaches use pretrained TabPFN estimators and support
classification (``cls``) and regression (``reg``).


Fine-Tuning
^^^^^^^^^^^^

.. py:currentmodule:: train_tabpfn

.. py:function:: finetune_tabpfn(X_train, y_train, X_val, y_val, out, mode, args, batch_size, num_epochs, model_str)

   Fine-tunes pretrained TabPFN models by updating their neural
   network parameters using gradient-based optimization.

   :param X_train: Training feature DataFrame.
   :param y_train: Training target values.
   :param X_val: Validation feature DataFrame.
   :param y_val: Validation target values.
   :param out: Directory for training outputs.
   :param mode: Task mode, either ``cls`` or ``reg``.
   :param args: Parsed configuration arguments.
   :param batch_size: Number of samples per training batch.
   :param num_epochs: Number of fine-tuning epochs.
   :param model_str: Model identifier.

   The function initializes a pretrained ``TabPFNClassifier``
   or ``TabPFNRegressor`` according to the selected task.

   Training and validation data are converted into preprocessed
   context-query batches using ``get_preprocessed_dataset_chunks``
   and ``meta_dataset_collator``.

   During each training iteration, the model uses the context
   samples to generate predictions for the query samples.

   The prediction loss is backpropagated through the pretrained
   network, and its parameters are updated using the Adam
   optimizer.

   The optimization objective depends on the selected task:

   * **Classification:** Cross-Entropy Loss, with classification
     accuracy tracked during training and validation.
   * **Regression:** Mean Squared Error (MSE), computed using
     the mean of the predicted probability distribution.

   Regression fine-tuning additionally applies gradient clipping
   with a maximum norm of 1.0 to stabilize training.

   After fine-tuning, the model is prepared for inference using
   ``clone_model_for_evaluation`` and fitted on the training data.

   Training and validation metrics are saved to
   ``training_log.csv``, and the fine-tuned estimator is
   serialized as ``finetuned_TABPFN_<mode>.joblib``.


In-Context Learning
^^^^^^^^^^^^^^^^^^^

.. py:function:: train_tabpfn_in_context_learning(X_train, y_train, X_val, y_val, out, mode, args, model_str)

   Fits a pretrained TabPFN estimator using in-context learning,
   without performing gradient-based optimization.

   :param X_train: Training feature DataFrame.
   :param y_train: Training target values.
   :param X_val: Validation feature DataFrame.
   :param y_val: Validation target values.
   :param out: Directory for training outputs.
   :param mode: Task mode, either ``cls`` or ``reg``.
   :param args: Parsed configuration arguments.
   :param model_str: Model identifier.

   The function initializes a pretrained ``TabPFNClassifier``
   or ``TabPFNRegressor`` and calls ``fit(X_train, y_train)``.

   Unlike fine-tuning, this operation prepares the training
   observations as context for subsequent predictions without
   updating the pretrained neural network weights.

   The fitted estimator performs inference on both the
   training and validation datasets.

   The evaluation metrics depend on the selected task:

   * **Classification:** Accuracy and logarithmic loss.
   * **Regression:** Mean Squared Error (MSE) and
     Mean Absolute Error (MAE).

   Classification accuracy and regression metrics are saved
   to ``training_metrics.json``.

   The fitted estimator is saved in ``.tabpfn_fit`` format
   using ``save_fitted_tabpfn_model``.

   This approach does not require an explicit epoch-based
   training loop or backpropagation.


Model Evaluation
----------------

The evaluation workflows load previously fitted models and
perform inference on held-out tabular datasets.

Evaluation results are stored in ``full_metrics.json``.


GANDALF Evaluation
~~~~~~~~~~~~~~~~~~

**Entry point:** ``evaluate_gandalf.evaluate``

.. py:currentmodule:: evaluate_gandalf

.. py:function:: evaluate(eval_df, model_weights, out, args, model_str)

   Evaluates a trained GANDALF model on a held-out dataset.

   :param eval_df: Path to the evaluation Parquet dataset.
   :param model_weights: Directory containing the saved model.
   :param out: Directory for evaluation outputs.
   :param args: Parsed configuration arguments.
   :param model_str: Model identifier.

   The function reconstructs the trained model using
   ``TabularModel.load_model()`` and performs evaluation
   through ``TabularModel.evaluate()``.

   The resulting task-specific metrics are collected,
   standardized where applicable, and saved to
   ``full_metrics.json``.

   The function also records the evaluation execution time.


TabPFN Evaluation
~~~~~~~~~~~~~~~~~

**Entry point:** ``evaluate_tabpfn.evaluate_tabpfn``

.. py:currentmodule:: evaluate_tabpfn

.. py:function:: evaluate_tabpfn(X_eval, y_eval, out, args, model_path, mode, finetuning_mode, model_str)

   Evaluates a fitted or fine-tuned TabPFN model on a
   held-out dataset.

   :param X_eval: Evaluation feature DataFrame.
   :param y_eval: Evaluation target values.
   :param out: Directory for evaluation outputs.
   :param args: Parsed configuration arguments.
   :param model_path: Path to the saved TabPFN model.
   :param mode: Task mode, either ``cls`` or ``reg``.
   :param finetuning_mode: Whether the model was fine-tuned.
   :param model_str: Model identifier.

   Model loading depends on the training approach:

   * **Fine-tuning:** Loads the serialized Joblib estimator.
   * **In-context learning:** Loads the fitted TabPFN estimator
     using ``load_fitted_tabpfn_model``.

   The loaded model performs inference on the evaluation
   dataset, and the following metrics are computed:

   * **Classification:** Prediction accuracy.
   * **Regression:** Mean Squared Error (MSE) and
     Mean Absolute Error (MAE).

   Results are saved to ``full_metrics.json``.


Embedding Extraction
--------------------

The embedding workflows extract numerical feature
representations from tabular datasets using previously
fitted GANDALF or TabPFN models.

The resulting embeddings are stored in Parquet format,
together with target labels and optional grouping information.


GANDALF Embeddings
~~~~~~~~~~~~~~~~~~

**Entry point:** ``gandalf_embeddings.embed``

.. py:currentmodule:: gandalf_embeddings

.. py:function:: embed(eval_df, out, model_weights, cont_cols, cat_cols, batch_size, target, args, grouper, model_str, device)

   Extracts intermediate feature representations from a
   trained GANDALF model.

   :param eval_df: Path to the input Parquet dataset.
   :param out: Directory for embedding outputs.
   :param model_weights: Directory containing the trained model.
   :param cont_cols: Continuous feature columns.
   :param cat_cols: Categorical feature columns.
   :param batch_size: Configured batch size.
   :param target: Name of the target column.
   :param args: Parsed configuration arguments.
   :param grouper: Optional grouping column.
   :param model_str: Model identifier.
   :param device: Device used for inference.

   The function loads the trained model using
   ``TabularModel.load_model()`` and registers a forward hook
   on the model's backbone.

   Embeddings are captured from the backbone outputs during
   ``TabularModel.predict()``.

   Using the fitted model's prediction interface preserves
   the categorical encodings and continuous feature
   transformations established during training.

   The extracted feature representations are combined with
   their corresponding target labels and optional grouping
   information.

   The resulting DataFrame is saved to
   ``GANDALF_embeddings.parquet``.


TabPFN Embeddings
~~~~~~~~~~~~~~~~~

**Entry point:** ``tabpfn_embeddings.get_tabpfn_embeddings``

.. py:currentmodule:: tabpfn_embeddings

.. py:function:: get_tabpfn_embeddings(X_train, y_train, X_eval, y_eval, out, args, model_path, mode, finetuning_mode, model_str, grouper)

   Extracts numerical feature representations using a
   fitted or fine-tuned TabPFN classifier or regressor.

   :param X_train: Training features used as context.
   :param y_train: Training target values.
   :param X_eval: Features for embedding extraction.
   :param y_eval: Corresponding evaluation target values.
   :param out: Directory for embedding outputs.
   :param args: Parsed configuration arguments.
   :param model_path: Path to the saved TabPFN model.
   :param mode: Task mode, either ``cls`` or ``reg``.
   :param finetuning_mode: Whether the model was fine-tuned.
   :param model_str: Model identifier.
   :param grouper: Optional grouping column.

   The function loads the appropriate TabPFN estimator
   according to the fine-tuning configuration.

   A ``TabPFNEmbedding`` instance is initialized using the
   loaded classifier or regressor, with ``n_fold=0``.

   Embeddings are generated through ``get_embeddings()``,
   which receives the training observations as context
   and extracts representations for the evaluation samples.

   The extracted embeddings are combined with target labels
   and optional grouping information to be saved as a Parquet DataFrame.
