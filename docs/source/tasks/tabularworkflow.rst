
Tabular Workflow Components
===========================

DeepTune provides dedicated components for tabular classification
and regression using GANDALF and TabPFN. These components support
model training, evaluation, and embedding extraction. This page
provides a high-level overview of the tabular workflow components.


Model Training
--------------

**Entry points:** ``train_gandalf.train`` and ``train_tabpfn.main``

The training workflows prepare tabular datasets, configure the
selected model, and perform training or fine-tuning according
to the specified task.

GANDALF
~~~~~~~

.. py:currentmodule:: src.tabular.gandalf

.. py:class:: GANDALF(data_config, optimizer_config, trainer_config, task, gflu_stages, gflu_feature_init_sparsity, gflu_dropout, learning_rate)

   Configures the GANDALF architecture using PyTorch Tabular.

   :param data_config: Configuration of input features and targets.
   :param optimizer_config: Optimizer configuration.
   :param trainer_config: Training configuration.
   :param task: Classification or regression task.
   :param gflu_stages: Number of GFLU stages.
   :param gflu_feature_init_sparsity: Initial feature sparsity.
   :param gflu_dropout: Dropout rate.
   :param learning_rate: Model learning rate.

   The training workflow initializes a ``TabularModel``, fits it
   on training and validation datasets, and saves the model
   alongside training metrics and configuration parameters.


TabPFN
~~~~~~

**Entry point:** ``train_tabpfn.main``

The TabPFN workflow supports classification and regression
through two execution modes.

.. py:currentmodule:: train_tabpfn

.. py:function:: finetune_tabpfn(X_train, y_train, X_val, y_val, out, mode, args, batch_size, num_epochs, model_str="TABPFN")

   Fine-tunes a pretrained TabPFN model using preprocessed
   training batches and the Adam optimizer.

   * **Classification:** Cross-Entropy Loss and accuracy.
   * **Regression:** Mean Squared Error (MSE) loss.

   Training and validation metrics are recorded in
   ``training_log.csv``, and the fine-tuned model is saved
   in Joblib format.

.. py:function:: train_tabpfn_in_context_learning(X_train, y_train, X_val, y_val, out, mode, args, model_str="TABPFN")

   Fits a pretrained TabPFN estimator to the training data
   without gradient-based fine-tuning.

   Computes classification accuracy and log loss, or regression
   MSE and MAE, on the training and validation datasets.

   Results are saved to ``training_metrics.json``, and the
   fitted model is saved in ``.tabpfn_fit`` format.


Model Evaluation
----------------

**Entry points:** ``evaluate_gandalf.evaluate`` and
``evaluate_tabpfn.evaluate_tabpfn``

The evaluation workflows load the fitted models and evaluate
their predictions on held-out tabular datasets.

.. py:currentmodule:: evaluate_gandalf

.. py:function:: evaluate(eval_df, model_weights, out, args, model_str)

   Loads the saved GANDALF model using PyTorch Tabular and
   evaluates it on the provided dataset.

   Returns the evaluation metrics and saves them to
   ``full_metrics.json``.


.. py:currentmodule:: evaluate_tabpfn

.. py:function:: evaluate_tabpfn(X_eval, y_eval, out, args, model_path, mode, finetuning_mode, model_str)

   Loads the fitted or fine-tuned TabPFN model and performs
   inference on the evaluation dataset.

   * **Classification:** Accuracy.
   * **Regression:** MSE and MAE.

   Results are saved to ``full_metrics.json``.


Embedding Extraction
--------------------

**Entry points:** ``gandalf_embeddings.embed`` and
``tabpfn_embeddings.get_tabpfn_embeddings``

The embedding workflows extract numerical feature
representations from tabular datasets using trained models.


GANDALF Embeddings
~~~~~~~~~~~~~~~~~~

.. py:currentmodule:: gandalf_embeddings

.. py:function:: embed(eval_df, out, model_weights, cont_cols, cat_cols, batch_size, target, args, grouper, model_str, device)

   Loads a trained GANDALF model and extracts intermediate
   feature representations through a forward hook on the
   model backbone.

   The embeddings are combined with their corresponding labels
   and optional grouping information.

   Results are saved to ``GANDALF_embeddings.parquet``.


TabPFN Embeddings
~~~~~~~~~~~~~~~~~

.. py:currentmodule:: tabpfn_embeddings

.. py:function:: get_tabpfn_embeddings(X_train, y_train, X_eval, y_eval, out, args, model_path, mode, finetuning_mode, model_str, grouper)

   Extracts tabular feature representations using the
   ``TabPFNEmbedding`` utility from a fitted or fine-tuned
   TabPFN classifier or regressor.

   The extraction process uses training data as context
   and generates embeddings for evaluation samples.

   The resulting DataFrame includes extracted embeddings,
   target labels, and optional grouping information.

   Embeddings are saved in Parquet format as
   ``TABPFN_cls_embeddings.parquet`` or
   ``TABPFN_reg_embeddings.parquet``.
