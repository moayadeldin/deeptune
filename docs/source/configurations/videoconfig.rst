.. _videoconfig.rst:

Video Classification Parameters Reference
------------------------------------------------------

Dataset and Task Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--modality``
     - Specifies the input data modality (e.g., ``video``).
   * - ``--mode``
     - Defines the machine learning task (i.e., ``cls`` for classification).
   * - ``--target``
     - Name of the target column containing the prediction labels.
   * - ``--num_classes``
     - Number of target classes for classification.

Model and Training Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--model``
     - Specifies the model version.
   * - ``--batch_size``
     - Number of samples processed in each training batch.
   * - ``--num_epochs``
     - Number of training epochs.
   * - ``--learning_rate``
     - Learning rate used by the optimizer.
   * - ``--num_frames``
     - Number of frames sampled from each video clip for model input.
   * - ``--pooling``
     - Pooling method used to aggregate frame-level features into a clip-level representation. Supported methods are 'mean' and 'attention'.

Optional Execution Flags
~~~~~~~~~~~~~~~~~~~~~~~~

The following flags enable additional execution and fine-tuning capabilities.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``--fixed-seed``
     - Enables fixed random seeds to improve experiment reproducibility.
   * - ``--use-peft``
     - Enables parameter-efficient fine-tuning (PEFT) for the model.
   * - ``--freeze-backbone``
     - Freezes the backbone of the model during training, allowing only the added layers to be trained.
    * - ``--raw-data``
     - Enables processing of raw (non-parquet) input data.

Predefined Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~
To simplify the initial experience with **DeepTune**, a set of predefined hyperparameters is used. These hyperparameters have been chosen based on common practices and are intended to provide a good starting point for most experiments. When you run the unified CLI command without specifying certain hyperparameters, **DeepTune** will automatically apply these predefined values. The predefined hyperparameters are as follows:

- **Learning Rate**: 0.0001
- **Number of Epochs**: 10
- **Number of Added Layers**: 2
- **Embedding Size**: 1000 (applicable when using two added layers)
- **Train/Validation/Test Split Ratios**: 70% / 10% / 20%
- **Num Frames**: 8 (i.e., 8 frames are sampled from each video clip for model input)
- **Pooling Method**: 'mean' (i.e., mean pooling is applied by default to aggregate frame-level features into a clip-level representation)
- **Disable Numerical Encoding**: False (i.e., numerical encoding is applied by default as part of preprocessing).

Raw Data Handling
~~~~~~~~~~~~~~~~~~

**DeepTune** was designed initially to only support data in format of Parquet files. However, to accomodate users who have their datasets in raw video formats (e.g., MP4, etc.), the library now supports raw data processing. When using the ``--raw-data`` flag, **DeepTune** will automatically convert the raw video files into Parquet format for efficient processing. The raw video data must be organized in a directory structure where each subdirectory represents a class label containing the respective video clips.

For example, the directory structure for video data should look like this:

.. code-block:: text

    data/
    ├── class_1/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── class_2/
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   └── ...
    ├── class_3/
    │   ├── video1.mp4
    │   └── ...
    └── class_4/
        ├── video1.mp4
        └── ...

Unified CLI Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After running the unified CLI command, **DeepTune** will generate an output directory containing the results of your experiment. The structure of the output directory will be as follows:

.. code-block:: text

   output_directory
   ├── deeptune-<yyyymmdd>-exp<int>
       ├── <modality>_dataset_<yyyymmdd_hhmm>.parquet
       ├── data_splits_<yyyymmdd_hhmm>
       │   ├── train_split.parquet
       │   ├── val_split.parquet
       │   ├── test_split.parquet
       │   └── test_indices.csv
       │   └── split_summary.json
       │   └── label_mapping.json
       ├── trainval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── cli_arguments.json
       │   ├── model_weights.pth
       │   ├── training_details.json
       │   ├── training_log.csv
       ├── eval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── full_metrics.json
       │   ├── cli_arguments.json
       │   └── evaluation_details.json
       ├── embed_output_<model_version>_<yyyymmdd_hhmm>
           ├── <model_version>_cls_embeddings.parquet
           ├── cli_arguments.json 
           ├── embeddings_details.json
