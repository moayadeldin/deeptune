.. _autoencoder_config.rst:

Autoencoder Configuration Parameters Reference
----------------------------------------------

The following are the configuration parameters for the Autoencoder Image Reconstruction task in **DeepTune**:

.. note::
    The autoencoders functionality is currently in an experimental stage, with **DeepTune** providing support for basic image reconstruction without providing further details (e.g., loss values, evaluation metrics) for the time being. We recommend using it with caution and providing feedback to help us improve it. AI can produce mistakes. Please verify the results and report any issues you encounter.


Model and Training Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--num_epochs``
     - Specifies the number of training epochs.
   * - ``--batch_size``
     - Number of samples processed in each training batch.
   * - ``--learning_rate``
     - The learning rate for the optimizer.
   * - ``--image_size``
     - The size of the output images.

Optional Execution Flags
~~~~~~~~~~~~~~~~~~~~~~~~

The following flags enable additional execution and
fine-tuning capabilities.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``--if-grayscale``
     - Enables processing of grayscale images.
   * - ``--use-deeper``
     - Enables processing of the deeper 5 x 5 autoencoder architecture.

Unified CLI Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After running the unified CLI command, **DeepTune** will generate an output directory containing the results of your experiment. The structure of the output directory will be as follows:


.. code-block:: text

   output_directory/
   └── reconstructed_images_<yymmdd>/
       ├── autoencoder_model.pth
       ├── Class 0/
       │   ├── reconstructed_0.png
       │   └── reconstructed_1.png
       ├── Class 1/
       │   ├── reconstructed_0.png
       │   └── reconstructed_1.png
       ├── Class 2/
       │   ├── reconstructed_0.png
       │   └── reconstructed_1.png
       ├── Class 3/
       │   ├── reconstructed_0.png
       │   └── reconstructed_1.png
       └── ...

    

