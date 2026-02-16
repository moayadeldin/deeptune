Using Autoencoders in **DeepTune**
==================================

Autoencoders are neural networks that learn how to compress the input images into a compact representation and then reconstruct the images from the compressed form. They consist of two main parts: the encoder, which compresses the input data, and the decoder, which reconstructs the original data from the compressed representation. In DeepTune, we provide the option of using autoencoders for image reconstruction tasks.

.. note::
    The autoencoders functionality is currently in an experimental stage, with **DeepTune** providing support for basic image reconstruction without providing further details (e.g., loss values, evaluation metrics) for the time being. We recommend using it with caution and providing feedback to help us improve it. AI can produce mistakes. Please verify the results and report any issues you encounter.

We provide an oversight of the autoencoder architecture:

.. important::
    The autoencoder architecture follows an encoder-decoder structure designed to extract and reconstruct hierarchical features. The process begins with a Z-score normalization and a Gaussan blur to the input. The encoder consists of three successive stages, each stage utilizes a 3 by 3 convolutions → Group Normalization → SiLU functon, with a stride of 2 for doubling the feature channels and halving the spatial dimensions. The decoder mirrors the encoder and the model is projecting the feature maps back to the original image space dimensions.

    For more information on the Group normalization and SiLU function, please refer to the following resources: `Group Normalization <https://wandb.ai/wandb_fc/GroupNorm/reports/Group-Normalization-in-Pytorch-With-Examples---VmlldzoxMzU0MzMy>`_ and `SiLU Function <https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html>`_ 

To use autoencoders in **DeepTune**, you can follow the following command:

.. code-block:: console

    $ python -m trainers.vision.train \
    --train_df <str> \
    --test_df <str> \
    -- num_epochs <int> \
    --learning_rate <float> \
    --if-grayscale <bool> \

.. note::
    The ``--if-grayscale`` flag is optional and can be set to `True` if the input images are grayscale, or `False` if they are RGB. By default, it is set to `False`.


