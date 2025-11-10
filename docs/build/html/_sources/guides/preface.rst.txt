Preface
=========

Now as you installed the required packages needed for *DeepTune*, and just before running your first *DeepTune* program, you may read this preface to get more engaged on what exactly to expect from the program.

*DeepTune* currently supports finetuning 6 different state-of-the-art image classification models to fine-tune your dataset with: ResNet, DenseNet, Swin, EfficientNet, VGGNet, and ViT for images, with BERT, and GPT-2 for images. More details of the supported variants is found in the documentation's supported models table.

*DeepTune* gives you also a wide flexible set of options to choose what you think would suit your case the best. The options are as follows:

- **Transfer Learning Mode**: *DeepTune* currently supports applying partial fine-tuning or full fine-tuning for image and text pre-trained models. Only the added layers are updated during partial fine-tuning, while the rest of the model remains frozen. For full fine-tuning, the weights update is applied across the whole architecture.

- **Supporting Parameter Efficient Fine-tuning (PeFT)**: PeFT techniques are known as capable of providing performance improvements while being resource-efficient. In *DeepTune*, PeFT with Low-Rank Adaptation (LoRA) is supported for full fine-tuning.

.. note::

   *DeepTune* currently doesn't support PEFT for GPT-2.

- **Adjustable Additional Layer Choices**: Fine-tuning is commonly applied in Deep Learning by adding one or more layer(s) on the top of the fine-tuned model. *DeepTune* gives you the choice of adding one, or two layers on the top of the model. Moreover, for the last layer size (also referred to as Embedding Layer) this is specified by the user choice as a CLI argument.

- **Task Type**: *DeepTune* provides initial support for converting classification-based models to work for regression.

- **Embeddings Extraction**: *DeepTune* provides a wide support for extracting embeddings for your dataset for all of the models mentioned above. This application is extremely useful if you want to get a meaningful representation of your own dataset to utilize further (e.g, projecting in 2D and see how they correlate, provide them to classical ML approach, etc.)

.. note::
    Kindly note that *DeepTune* for images and texts only accepts Parquet files as an input. The parquet file expected is actually containing two columns (among potentially many), If we work with images, then the two columns are [`images`, `labels`] pair. **Images must be in Bytes Format for efficient representation, and labels must be numerically encoded**. If we work with text, then the two columns are [`text`, `labels`] pair. For text, **label column must be numerically encoded also.**
