.. _videoworkflow:

Video-based Workflow Components
===============================

DeepTune provides clip-level video classification through
model training, evaluation, and embedding extraction workflows. Video
models use either native spatio-temporal architectures or shared 2D
image backbones with temporal frame pooling. The workflows reuse
DeepTune's existing vision training and evaluation components.


Model Training
--------------

**Entry point:** ``trainers.video.train.train``

The training function constructs the selected video model using
``utils.build_video_model``, prepares video clips from Parquet datasets,
and delegates optimization to the vision ``Trainer`` class. Each clip
has shape ``[B, T, C, H, W]``, where ``T`` is the number of sampled frames.

.. py:currentmodule:: src.video.video3d

.. py:class:: adjustedVideo3D(num_classes, video_version, added_layers, embedding_layer_size, freeze_backbone, task_type, output_dim)

   Wraps a Kinetics-400-pretrained spatio-temporal backbone, using 3D
   convolutions or video-transformer attention across frames.

   :param num_classes: Number of output classes for classification.
   :param video_version: Architecture identifier (``r3d_18``, ``mc3_18``,
      ``r2plus1d_18``, ``mvit_v2_s``, ``swin3d_t``, ``swin3d_s``, or ``swin3d_b``).
   :param added_layers: Number of added prediction layers (one or two).
   :param embedding_layer_size: Intermediate feature dimension when using two layers.
   :param freeze_backbone: Whether to freeze the pretrained backbone.
   :param task_type: ``cls`` for classification.
   :param output_dim: Number of outputs; defaults to one.

   Native models accept clips in ``[B, T, C, H, W]`` order and internally
   convert them to ``[B, C, T, H, W]`` for the backbone.

.. py:currentmodule:: src.video.video3d_peft

.. py:class:: adjustedPeftVideo3D(num_classes, video_version, added_layers, lora_attention_dimension, freeze_backbone, task_type, output_dim)

   Provides a LoRA-based parameter-efficient variant of the native video
   models. LoRA targets supported convolutional and linear layers of the
   spatio-temporal backbone.

   :param lora_attention_dimension: LoRA rank and, when two prediction
      layers are used, the intermediate feature dimension.

   The remaining parameters follow ``adjustedVideo3D``.

.. py:currentmodule:: src.video.frame_pool

.. py:class:: VideoFramePoolClassifier(frame_model_cls, frame_model_version, num_classes, added_layers, embedding_layer_size, freeze_backbone, task_type, pooling)

   Applies a single shared 2D vision backbone independently to every
   sampled frame, then combines per-frame predictions into a clip-level
   output. Supports standard and PEFT-enabled image backbones.

   :param frame_model_cls: Class implementing the shared 2D backbone.
   :param frame_model_version: Selected 2D architecture variant.
   :param num_classes: Number of output classes for classification.
   :param added_layers: Number of added layers on the frame backbone.
   :param embedding_layer_size: Intermediate feature dimension.
   :param freeze_backbone: Whether to freeze the frame backbone.
   :param task_type: ``cls`` for classification.
   :param pooling: ``mean`` or learned ``attention`` pooling of
      per-frame predictions.

   .. py:method:: forward(x, extract_embed)

      Produces clip-level predictions from a video batch. When
      ``extract_embed=True``, returns mean-pooled frame embeddings,
      regardless of the prediction pooling setting.

.. py:currentmodule:: trainers.vision.trainer

.. py:class:: Trainer(model, train_loader, val_loader, learning_rate, mode, num_epochs, output_dir)

   Trains and validates either video model family using the shared
   vision training loop. Classification uses Cross-Entropy Loss;
   regression uses Mean Squared Error (MSE).

   .. py:method:: train()

      Runs optimization and training-time logging.

   .. py:method:: validate()

      Computes task-specific validation results.

   .. py:method:: saveModel(path)

      Saves model parameters to a checkpoint.


Model Evaluation
----------------

**Entry point:** ``evaluators.video.evaluate.evaluate``

The evaluation function reconstructs the selected video model, loads
its checkpoint through the shared vision ``TestTrainer``, and evaluates
clips from a held-out Parquet dataset.

.. py:currentmodule:: evaluators.vision.evaluator

.. py:class:: TestTrainer(model, batch_size, test_loader, mode, output_dir, device)

   Evaluates video models using the existing vision evaluation workflow.

   .. py:method:: test(best_model_weights_path)

      Runs inference and computes task-specific evaluation metrics:

      * **Classification:** Classification report and AUROC, when applicable.

      Evaluation results are saved through the vision evaluator.


Embedding Extraction
--------------------

**Entry point:** ``embed.video.embed.embed``

The embedding function constructs the selected video model, loads a
checkpoint for fine-tuned or PEFT models, and extracts clip-level
features using the model's ``forward(x, extract_embed=True)`` interface.

.. py:currentmodule:: embed.video.embed

.. py:class:: VideoEmbeddingModel(model, model_version, added_layers, embed_size, num_frames, device)

   Extracts representations from clips sampled from a video dataset.

   :param model: Native spatio-temporal or frame-pooling video model.
   :param model_version: Selected architecture variant.
   :param added_layers: Number of added prediction layers.
   :param embed_size: Configured intermediate embedding dimension.
   :param num_frames: Number of frames sampled per clip.
   :param device: Device used for embedding extraction.

   .. py:method:: __call__(df, batch_size)

      Returns a DataFrame with extracted clip embeddings, labels when
      available, and additional dataset columns. Frame-pooling models
      always average per-frame embeddings over time.

The embedding workflow saves the resulting DataFrame in Parquet format
as ``<model_str>_<mode>_embeddings.parquet``.
