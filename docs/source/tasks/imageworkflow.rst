.. _imageworkflow:

Image-based Workflow Components
===============================

DeepTune provides dedicated components for image classification
and regression, organized into model training, evaluation, and
embedding extraction workflows. These components are coordinated
through the main execution module. In this page, we provide a class-level abstract
overview of the image base workflow components.


Model Training
--------------

**Entry point:** ``trainers.vision.train.train``

The training function initializes the selected vision architecture,
prepares the training and validation data loaders, and delegates
model optimization to the ``Trainer`` class.

.. py:currentmodule:: trainers.vision.trainer

.. py:class:: Trainer(model, train_loader, val_loader, learning_rate, mode, num_epochs, output_dir)

   Manages training and validation of PyTorch vision models.

   :param model: PyTorch model to be trained.
   :param train_loader: DataLoader containing the training samples.
   :param val_loader: DataLoader containing the validation samples.
   :param learning_rate: Learning rate for the Adam optimizer.
   :param mode: Task type, either ``cls`` or ``reg``.
   :param num_epochs: Number of training epochs.
   :param output_dir: Directory for training logs and model checkpoints.

   Classification uses Cross-Entropy Loss, while regression uses
   Mean Squared Error (MSE).

   .. py:method:: train()

      Executes the training loop, including validation,
      metric logging, and execution-time tracking.

   .. py:method:: validate()

      Evaluates the model on the validation dataset and
      returns task-specific validation results.

   .. py:method:: saveModel()

      Saves the trained model parameters.



Model Evaluation
----------------

**Entry point:** ``evaluators.vision.evaluate.evaluate``

The evaluation function reconstructs the selected architecture
and delegates evaluation to the ``TestTrainer`` class.

.. py:currentmodule:: evaluators.vision.evaluator

.. py:class:: TestTrainer(model, batch_size, test_loader, mode, output_dir, device)

   Evaluates trained vision models on a held-out Evaluation dataset.

   :param model: PyTorch model to evaluate.
   :param batch_size: Configured evaluation batch size.
   :param test_loader: DataLoader containing the Evaluation samples.
   :param mode: Task type, either ``cls`` or ``reg``.
   :param output_dir: Directory for evaluation outputs.
   :param device: Device used for inference.

   .. py:method:: test()

      Loads the specified checkpoint, performs inference, and
      computes task-specific evaluation metrics.

      * **Classification:** Classification report and AUROC,
        when applicable.
      * **Regression:** MAE, RMSE, and R-squared.

      Results are saved to ``full_metrics.json``.


Embedding Extraction
--------------------

**Entry point:** ``embed.vision.embed.embed``

The embedding workflow loads the selected model and prepares its
feature extraction pathway according to the architecture and
model configuration.

.. py:currentmodule:: embed.vision.embed

.. py:class:: EmbeddingModel(model, model_name, model_version, added_layers, embed_size, task_type, device)

   Extracts numerical feature representations from image datasets.

   :param model: Model configured for embedding extraction.
   :param model_name: Model architecture identifier.
   :param model_version: Specific model variant.
   :param added_layers: Number of additional model layers.
   :param embed_size: Configured intermediate embedding dimension.
   :param task_type: Task identifier.
   :param device: Device used for embedding extraction.

   An ``EmbeddingModel`` instance accepts a dataset DataFrame
   and an optional batch size through its callable interface.
   It returns a DataFrame containing extracted feature
   representations and associated labels.

The embedding workflow saves the resulting DataFrame in
Parquet format.


Specialized SiGLiP Implementation
---------------------------------

SiGLiP uses dedicated components for image classification,
with specialized preprocessing, model training, evaluation,
and embedding extraction.


SiGLiP Training
~~~~~~~~~~~~~~~

**Entry point:** ``trainers.vision.custom_siglip_train.train_siglip``

.. py:currentmodule:: trainers.vision.custom_siglip_train

.. py:class:: SiglipTrainer(model, train_loader, val_loader, num_epochs, learning_rate, outdir, device, use_peft)

   Manages SiGLiP classification training using Adam,
   Cross-Entropy Loss, and automatic mixed precision.

   :param model: SiGLiP classification model.
   :param train_loader: Training data loader.
   :param val_loader: Validation data loader.
   :param num_epochs: Number of training epochs.
   :param learning_rate: Optimizer learning rate.
   :param outdir: Directory for training outputs.
   :param device: Training device.
   :param use_peft: Whether PEFT is enabled.

   .. py:method:: train()

      Executes training and saves the model checkpoint,
      configuration, and training logs.

   .. py:method:: validate()

      Computes validation loss and classification accuracy.


SiGLiP Evaluation
~~~~~~~~~~~~~~~~~

**Entry point:** ``evaluators.vision.custom_siglip_evaluate.evaluate_siglip``

.. py:currentmodule:: evaluators.vision.custom_siglip_evaluate

.. py:class:: TestTrainer(model, data_loader, output_dir, device, use_peft)

   Evaluates a trained SiGLiP classifier.

   :param model: Trained SiGLiP model.
   :param data_loader: Evaluation data loader.
   :param output_dir: Evaluation output directory.
   :param device: Inference device.
   :param use_peft: Whether the PEFT configuration is used.

   .. py:method:: test()

      Computes classification metrics, confusion matrix,
      and AUROC where applicable, saving the results
      to ``full_metrics.json``.


SiGLiP Embedding Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Entry point:** ``embed.vision.custom_siglip_embed.embed_siglip``

The ``embed_siglip`` function extracts image representations
using the SiGLiP feature extraction interface.

It supports pretrained, fine-tuned, and PEFT configurations
and accepts the dataset path, model checkpoint, model
configuration, execution device, and output paths.

Extracted representations are saved in Parquet format.


