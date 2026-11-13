
Autoencoder Workflow Components
===============================

DeepTune provides dedicated components for image reconstruction
using convolutional autoencoders. The workflow supports image
preprocessing, model training, evaluation, and reconstructed image
generation. This page provides a class-level overview of the
autoencoder workflow components.


Model Architecture
------------------

**Entry point:** ``autoencoders.components.architecture.AutoEncoder``

The autoencoder architecture consists of a convolutional encoder,
a bottleneck representation, and a decoder that reconstructs
the input image.

.. py:currentmodule:: autoencoders.components.architecture

.. py:class:: ConvBlock(in_ch, out_ch, norm, groups)

   Implements a convolutional processing block consisting of
   a 2D convolution, optional Group Normalization, and
   SiLU activation.

   :param in_ch: Number of input channels.
   :param out_ch: Number of output channels.
   :param norm: Normalization type.
   :param groups: Maximum number of normalization groups.


.. py:class:: AutoEncoder(use_deeper, in_ch, base, blur_k, blur_simga)

   Implements an encoder-decoder architecture for image
   reconstruction.

   :param use_deeper: Whether to use the deeper architecture
       with additional encoding and decoding blocks.
   :param in_ch: Number of input image channels.
   :param base: Base number of convolutional feature channels.
   :param blur_k: Kernel size of the Gaussian blur operation.
   :param blur_simga: Standard deviation of the Gaussian blur.

   The encoder progressively downsamples the input image
   through convolutional blocks, while the decoder reconstructs
   its spatial dimensions using transposed convolutions.

   The default architecture uses three downsampling stages,
   while the deeper configuration uses five stages.

   .. py:method:: forward(x)

      Applies Gaussian blur to the input image, extracts
      intermediate representations through the encoder,
      and reconstructs the image using the decoder.

      Returns the reconstructed image and the original
      unblurred input tensor.

      The reconstructed image is passed through a sigmoid
      activation to constrain its values to ``[0, 1]``.


Image Preprocessing
-------------------

**Entry point:** ``autoencoders.components.transforms.apply_transform``

.. py:currentmodule:: autoencoders.components.transforms

.. py:function:: apply_transform(image_size, if_grayscale)

   Constructs the image preprocessing pipeline.

   :param image_size: Target height and width of the image.
   :param if_grayscale: Whether to convert images to grayscale.

   The pipeline resizes images to square dimensions, optionally
   converts them to grayscale, and transforms them into
   PyTorch tensors.


Model Training
--------------

**Entry point:** ``autoencoders.autoencoder.run_autoencoder``

The training workflow initializes the selected autoencoder
architecture, prepares image DataLoaders, and optimizes
reconstruction using Adam and Smooth L1 Loss.

.. py:currentmodule:: autoencoders.autoencoder

.. py:function:: run_autoencoder(model, train_df_path, transform, test_df_path, num_epochs, learning_rate, out, model_weights, device)

   Coordinates autoencoder training and evaluation.

   :param model: Autoencoder model to train or evaluate.
   :param train_df_path: Path to the training Parquet dataset.
   :param transform: Image preprocessing transformations.
   :param test_df_path: Path to the testing Parquet dataset.
   :param num_epochs: Number of training epochs.
   :param learning_rate: Learning rate for the Adam optimizer.
   :param out: Directory for model and reconstruction outputs.
   :param model_weights: Optional path to existing model weights.
   :param device: Device used for model execution.

   When ``model_weights`` is not provided, the function
   trains the autoencoder by minimizing the Smooth L1 Loss
   between reconstructed images and original inputs.

   When ``model_weights`` is provided, the function loads
   the existing parameters and performs evaluation without
   additional training.


Model Evaluation
----------------

**Entry point:** ``autoencoders.autoencoder.test_autoencoder``

The evaluation workflow reconstructs images from the testing
dataset using the trained autoencoder.

.. py:currentmodule:: autoencoders.autoencoder

.. py:function:: test_autoencoder(model, test_loader, device)

   Evaluates the model on the testing dataset without
   gradient computation.

   Returns three lists containing the original target
   tensors, reconstructed images, and input images.

   The current implementation performs reconstruction
   without computing quantitative evaluation metrics.


Reconstruction Output
---------------------

**Entry point:** ``autoencoders.autoencoder.save_results``

The reconstruction workflow generates and organizes
reconstructed images according to their corresponding
dataset class labels.

.. py:currentmodule:: autoencoders.autoencoder

.. py:function:: save_results(model, out_dir, model_weights, test_loader)

   Performs inference on the testing dataset and saves
   the reconstructed images in class-specific directories.

   When training a new model, its parameters are saved
   as ``autoencoder_model.pth``.


.. py:function:: rescale_image(x, eps)

   Applies channel-wise spatial min-max normalization to
   rescale reconstructed image tensors to ``[0, 1]``
   before saving them.

The model checkpoint is generated only when training
a new autoencoder. Existing checkpoints are not saved
again during evaluation-only execution.
