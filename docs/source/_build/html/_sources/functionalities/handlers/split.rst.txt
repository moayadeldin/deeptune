Handling Datasets
=================

Split Your Dataset
------------------

We assume that your dataset formatted as Parquet File will need to be splitted into train/val/test splits as you are going to conduct different experiments with different models using *DeepTune*. Given that *DeepTune* expects the labels to be numerically encoded, the ``split_dataset`` function automatically by default encodes your label column. If you want to disable this functionality, use the ``--disable-numerical-encoding`` option. 

The following is the generic CLI structure to split the dataset:

.. code-block:: console

    $ python -m handlers.split_dataset \
        --df <str> \
        --train_size <float> \
        --val_size <float> \
        --test_size <float> \
        --out <path> \
        --[fixed-seed] \
        --[disable-numerical-encoding]

.. note::
   It is important to use the ``--fixed-seed`` flag to regenerate the same train/val/test splits everytime you run the above command.

The output will be stored in the directory specified with the ``--out`` argument,  
using the following naming format: ``data_splits_<yyyymmdd_hhmm>``.  
This directory will contain the split files, which will be used later for training and evaluation:

.. code-block:: text

   output_directory
   ├── data_splits_<yyyymmdd_hhmm>
       ├── cli_arguments.json
       ├── train_split.parquet
       ├── test_split.parquet
       └── val_split.parquet
       └── test_indices.csv


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
   * - ``--fixed-seed``
     - *(Flag)* Ensures that a fixed random seed is set for reproducibility.
   * - ``--disable-numerical-encoding``
     - *(Flag)* Disables the default numerical label encoding when generating splits.

.. note::

    For the ``test_indices.csv`` file, it includes an additional indices column that maps the entry of each test sample to the original dataset, making it easier to track where they are located in ``--df`` file.

Getting the Intersection Between Two Datasets
---------------------------------------------

This feature is mainly implemented to be integrated with `df-analyze <https://github.com/stfxecutables/df-analyze>`_, where `df-analyze <https://github.com/stfxecutables/df-analyze>`_ relies by default on 40% of the input dataset as the test set. Since the AutoML framework uses `DeepTune`'s embeddings for `df-analyze <https://github.com/stfxecutables/df-analyze>`_, it is sometimes necessary to extract the intersection between the two dataframes.

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
     - Path to first dataset as parquet file (usually `DeepTune'`s embeddings extracted).
   * - ``--df_csv_path <str>``
     - Path to second dataset as csv file (usually the subset 40% obtained df-analyze).
   * - ``--out <str>``
     - Path to the directory where you want to save the results.

The output will be stored in the directory specified with the ``--out`` argument, using the following naming format: ``intersection_<yyyymmdd_hhmm>.parquet``.