.. _timeseriesconfig.rst:

Time Series Forecasting Parameters Reference
----------------------------------------------

The following are the configuration parameters for the Time Series Forecasting task in **DeepTune**:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--modality``
     - Specifies the input data modality (i.e., ``timeseries``).
   * - ``--target``
     - Name of the target column containing the prediction labels.

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
   * - ``--time_idx_column``
     - The column index of the time series data.
   * - ``--target``
     - The column index of the target variable in the time series data.


Optional Execution Flags
~~~~~~~~~~~~~~~~~~~~~~~~

The following flags enable additional execution and
fine-tuning capabilities.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Flag
     - Description
   * - ``--fixed-seed``
     - Enables fixed random seeds to improve experiment
       reproducibility.
   * - ``--raw-data``
     - Enables processing of raw (non-parquet) input data.

Predefined Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~
To simplify the initial experience with **DeepTune**, a set of predefined hyperparameters is used. These hyperparameters have been chosen based on common practices and are intended to provide a good starting point for most experiments. When you run the unified CLI command without specifying certain hyperparameters, **DeepTune** will automatically apply these predefined values. The predefined hyperparameters are as follows:

- **Number of Epochs**: 10
- **Train/Validation/Test Split Ratios**: 70% / 10% / 20%

Unified CLI Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After running the unified CLI command, **DeepTune** will generate an output directory containing the results of your experiment. The structure of the output directory will be as follows:

.. code-block:: text

   output_directory
   ├── deeptune-<yyyymmdd>-exp<int>
       ├── <modality>_dataset_<yyyymmdd_hhmm>.parquet
       ├── data_splits_<yyyymmdd_hhmm>
       ├── <yyyymmdd_hhmm>
       │       ├── hparams.yaml
       ├── data_splits_<yyyymmdd_hhmm>
       │   ├── train_split.parquet
       │   ├── val_split.parquet
       │   ├── test_split.parquet
       │   └── test_indices.csv
       ├── trainval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── cli_arguments.json
       │   ├── prediction_output.json
       │   ├──deepar-<model_details>.ckpt
       │   ├── training_details.json
       │   ├── training_log.csv
       ├── eval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── full_metrics.json
       │   ├── cli_arguments.json
       │   └── evaluation_details.json
       │   └── prediction_output.json
       ├── embed_output_<model_version>_<yyyymmdd_hhmm>
           ├── <model_version>_cls_embeddings.parquet
           ├── cli_arguments.json 
           ├── embeddings_details.json