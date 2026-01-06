Your First **DeepTune** Run
===========================
Welcome to your first **DeepTune** project! This tutorial will guide you through the essential steps to set up and run a **DeepTune** experiment using a sample dataset. By the end of this tutorial, you will have a basic understanding of how to use **DeepTune** for your machine learning tasks. You should have already prepared the virtual environment and/or installed the dependencies as described in the :doc:`guides/install` section, and got an idea about **DeepTune** functionalities from the :doc:`guides/preface` section.


Unified Single CLI Call Structure
---------------------------------

The easiest way to get started with **DeepTune** is to use the unified pipeline that handles all the steps needed to run a complete experiment, from raw data handling to training, evaluation, and embeddings extraction.

.. code-block:: console

    $ python -m deeptune \
        --df <str> \
        --modality <images_or_text_or_timeseries_or_tabular> \
        --target <str> \
        --num_classes <int> \
        --batch_size <int> \
        --model_version <model_name> \
        --out <str> \
        [--fixed-seed] \
        [--use-peft] \
        [--raw-data] \
        [--freeze_backbone] \
        [--time_idx_column] <str> \

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``--df <str>``
     - Path to dataset to split (must be a parquet file if not using the ``--raw-data`` flag).
   * - ``--modality <images_or_text_or_timeseries_or_tabular>``
     - The data modality you are working with.
   * - ``--target <str>``
     - The name of the target column in your dataset. DeepTune will automatically rename this column to 'labels' for consistency across modalities.
    * - ``--num_classes <int>``
        - Number of classes in your dataset for classification tasks.
    * - ``--model_version <model_name>``
        - The model to use along with its respective architecture version.  
          You may refer to the *Supported Models* table for available options.
    * - ``--batch_size <int>``
        - Number of samples per batch.
    * - ``--out <str>``
        - Path to the directory where you want to save the results.
    * - ``--use_peft``
        - *(Flag)* Enables the use of Parameter Efficient Fine-Tuning (PeFT) techniques during training.
    * - ``--raw-data``
        - *(Flag)* Indicates that the input data is in raw format (e.g., CSV, image files) and needs to be converted to Parquet format.
   * - ``--fixed-seed``
     - *(Flag)* Ensures that a fixed random seed is set for reproducibility.
    * - ``--freeze_backbone``
        - *(Flag)* Indicates whether to freeze the backbone of the model during training.
    * - ``--time_idx_column <str>``
        - Name of the time index column in your time-series dataset. *Required only for time-series modality.*
.. note::

   You only need to specify the ``--num_classes`` argument for image classification tasks and with Multilingual BERT. It is not needed for other time-series, tabular models, and GPT-2 as they handle this internally.

For example, suppose that we want to run an experiment on an image classification task using a dataset stored in directory named ``data`` with fine-tuned ResNet18, and we decide that the target column would be named ``label`` for the parquet file saved during preprocessing. The dataset has 3 classes, we choose a batch size of 8, and we want to save the results in the ``output`` directory. We also want to use PeFT during training, and ensure that the random seed is fixed for reproducibility. The command would look like this:

.. code-block:: console

    $ python -m deeptune \
        --df data \
        --modality images \
        --target label \
        --num_classes 3 \
        --batch_size 8 \
        --out output \
        --model_version resnet18 \
        --use-peft \
        --fixed-seed \
        --raw-data

In case we want to run an experiment on a text classification task using a dataset stored in ``data/text_data.parquet`` with fine-tuned Multilingual BERT, with the target column named ``sentiment``, 2 classes, batch size of 16, saving results in ``text_output``, and without using PeFT, the command would be:

.. code-block:: console

    $ python -m deeptune \
        --df data/text_data.parquet \
        --model_version bert \
        --modality text \
        --target sentiment \
        --num_classes 2 \
        --batch_size 16 \
        --out text_output \
        --fixed-seed \

Raw Data Handling
------------------

**DeepTune** was designed initially to only support data in format of Parquet files. However, to make it easier for users to get started, **DeepTune** provides support for converting multiple raw data formats (CSV/XLSX for tabular, timeseries, or text data, and image files for vision tasks) into the required Parquet format using the ``--raw-data`` flag.

For the text, tabular, and time-series modalities, the raw data file must be in CSV or XLSX format. For the image modality, the raw data must be organized in a directory structure where each subdirectory represents the train, validation and test split. Inside each split subdirectory, it would contain the class labels containing the respective images.

For example, the directory structure for image data should look like this:

.. code-block:: text

    data/
    ├── train/
    │   ├── class_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class_2/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── class_3/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    ├── val/
    │   ├── class_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class_2/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── class_3/
    │       ├── image1.jpg
    │       ├── image2.jpg
    │       └── ...
    └── test/
        ├── class_1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── class_2/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        └── class_3/
            ├── image1.jpg
            ├── image2.jpg
            └── ...


When using the ``--raw-data`` flag, **DeepTune** will automatically handle the conversion process before proceeding with the experiment. Make sure to specify the correct path to your raw data using the ``--df`` argument. At the end, the converted Parquet files will be saved in the output directory specified by the ``--out`` argument, along with the experiment results.

Predefined Hyperparameters
---------------------------
To simplify the initial experience with **DeepTune**, a set of predefined hyperparameters is used. These hyperparameters have been chosen based on common practices and are intended to provide a good starting point for most experiments. When you run the unified CLI command without specifying certain hyperparameters, **DeepTune** will automatically apply these predefined values. The predefined hyperparameters are as follows:

- **Learning Rate**: 0.0001
- **Number of Epochs**: 10
- **Number of Added Layers**: 2
- **Embedding Size**: 1000 (applicable when using two added layers)
- **Train/Validation/Test Split Ratios**: 70% / 10% / 20%
- **Task Mode**: Classification (`cls`) by default
- **Disable Numerical Encoding**: False (i.e., numerical encoding is applied by default as part of preprocessing).
- **GFUL Stages**: 6 (applicable for GANDALF only)


Unified CLI Output Structure
---------------------------
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
       │   └── label_mapping.json
       ├── trainval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── cli_arguments.json
       │   ├── model_weights.pth
       │   ├── training_details.json
       │   ├── training_log.csv
       ├── eval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── full_metrics.json
       │   ├── cli_arguments.json
       ├── embed_output_<model_version>_<yyyymmdd_hhmm>
           ├── <model_version>_cls_embeddings.parquet
           ├── cli_arguments.json 
           ├── embeddings_details.json 

In order to avoid overwriting previous experiment results, each run of the unified CLI command will create a new subdirectory within the specified output directory. The subdirectory will be named using the format ``deeptune-<yyyymmdd>-exp<int>``, where `<yyyymmdd>` represents the date of the experiment, and `<int>` is an incrementing integer starting from 1 for each experiment conducted on that date. This ensures that all your experiment results are preserved and easily accessible for future reference.

In order to know what the output directory components correspond to, you may refer to the respective sections in the :doc:`split`, :doc:`training`, :doc:`evaluation`, and :doc:`embedding` parts of the documentation.