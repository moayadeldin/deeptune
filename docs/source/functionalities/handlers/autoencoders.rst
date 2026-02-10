Using Autoencoders in **DeepTune**
==================================

Autoencoders are neural networks that learn how to compress the input images into a compact representation and then reconstruct the images from the compressed form. They consist of two main parts: the encoder, which compresses the input data, and the decoder, which reconstructs the original data from the compressed representation. In DeepTune, we provide the option of using autoencoders for image reconstruction tasks.

.. note::
    The autoencoders functionality is currently in an experimental stage, with **DeepTune** providing support for basic image reconstruction without providing further details (e.g., loss values, evaluation metrics) for the time being. We recommend using it with caution and providing feedback to help us improve it. AI can produce mistakes. Please verify the results and report any issues you encounter.

We provide an oversight of the autoencoder architecture:

.. important::
    The autoencoder architecture

To use autoencoders in **DeepTune**, you can follow the following command:

.. code-block:: console

    $ python -m trainers.vision.train \
    --train_df <str> \
    --test_df <str> \


