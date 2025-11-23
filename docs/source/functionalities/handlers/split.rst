Handling Datasets
=================

Raw Data Conversion to Parquet Format
--------------------------------------

We recall that for the text, tabular, and time-series modalities, the raw data file must be in CSV or XLSX format. For the image modality, the raw data must be organized in a directory structure where each subdirectory represents the train, validation and test split. Inside each split subdirectory, it would contain the class labels containing the respective images.

The following is the generic CLI structure to convert raw data into Parquet format:

.. code-block:: console

    $ python -m handlers.raw_to_parquet_dataset \
        --modality <images_or_text_or_timeseries_or_tabular> \
        --raw_dataset_dir <str> \
        --out <str> \

The output will be stored in the directory specified with the ``--out`` argument,  
using the following naming format: ``<modality>_dataset_<yyyymmdd_hhmm>.parquet``. 

.. note::
  Images are stored in bytes format inside the Parquet file.

Split Your Dataset
------------------

We assume that your dataset (formatted as Parquet File already) will need to be splitted into train/val/test splits as you are going to conduct different experiments with different models using **DeepTune**. Given that **DeepTune** expects the labels to be numerically encoded, the ``split_dataset`` function automatically by default encodes your label column. If you want to disable this functionality, use the ``--disable-numerical-encoding`` option. 

The following is the generic CLI structure to split the dataset:

.. code-block:: console

    $ python -m handlers.split_dataset \
        --df <str> \
        --train_size <float> \
        --val_size <float> \
        --test_size <float> \
        --out <path> \
        --target_column <str> \
        --[fixed-seed] \
        --[disable-numerical-encoding] \
        --[disable-target-column-renaming] \

.. note::
   It is important to use the ``--fixed-seed`` flag to regenerate the same train/val/test splits everytime you run the above command.

The output will be stored in the directory specified with the ``--out`` argument,  
using the following naming format: ``data_splits_<yyyymmdd_hhmm>``.  
This directory will contain the split files, which will be used later for training and evaluation:

.. code-block:: text

   output_directory
   ├── data_splits_<yyyymmdd_hhmm>
       ├── train_split.parquet
       ├── test_split.parquet
       └── val_split.parquet
       └── label_mapping.json
       └── test_indices.csv

.. note::

    Except for timeseries datasets, **DeepTune** renames the target column to 'labels' for consistency across modalities by default. If you want to keep the original target column name, use the ``--disable-target-column-renaming`` option.


.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``--df <str>``
     - Path to dataset to split (must be a parquet file).
   * - ``--train_size <float>``
     - Percentage of the training dataset w.r.t. the whole data.
   * - ``--val_size <float>``
     - Percentage of the validation dataset w.r.t. the whole data.
   * - ``--test_size <float>``
     - Percentage of the testing dataset w.r.t. the whole data.
   * - ``--out <str>``
     - Path to the directory where you want to save the results.
   * - ``--target_column <str>``
      - The name of the target column in your dataset. Default is 'labels' if not provided by the user.
   * - ``--fixed-seed``
     - *(Flag)* Ensures that a fixed random seed is set for reproducibility.
   * - ``--disable-numerical-encoding``
     - *(Flag)* Disables the default numerical label encoding when generating splits.
   * - ``--disable-target-column-renaming``
     - *(Flag)* Disables the automatic renaming of the target column to 'labels'. By default, **Deeptune** renames the target column to 'labels' for consistency across modalities.
.. note::

    For the ``test_indices.csv`` file, it includes an additional indices column that maps the entry of each test sample to the original dataset, making it easier to track where they are located in ``--df`` file. While the ``label_mapping.json`` file contains the mapping between the original labels and their corresponding numerical encodings if applied.

Get the Intersection Between Two Datasets
---------------------------------------------

This feature is mainly implemented to be integrated with `df-analyze <https://github.com/stfxecutables/df-analyze>`_, where `df-analyze <https://github.com/stfxecutables/df-analyze>`_ relies by default on 40% of the input dataset as the test set. Since the AutoML framework uses **DeepTune**'s embeddings for `df-analyze <https://github.com/stfxecutables/df-analyze>`_, it is sometimes necessary to extract the intersection between the two dataframes.

In order to achieve this, we use the following command: 

.. code-block:: console

    $ python -m handlers.get_intersection \
        --df_parquet_path <str> \
        --df_csv_path <str> \
        --out <str> \

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``--df_parquet_path <str>``
     - Path to first dataset as parquet file (usually **DeepTune**'s embeddings extracted).
   * - ``--df_csv_path <str>``
     - Path to second dataset as csv file (usually the subset 40% obtained df-analyze).
   * - ``--out <str>``
     - Path to the directory where you want to save the results.

The output will be stored in the directory specified with the ``--out`` argument, using the following naming format: ``intersection_<yyyymmdd_hhmm>.parquet``.