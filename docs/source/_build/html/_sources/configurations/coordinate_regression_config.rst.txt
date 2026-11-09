.. _coordinate_regression_config:

Coordinate Regression Configuration Parameters
==============================================

The following are the configuration parameters for the coordinate regression task in **DeepTune**:

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
   * - ``--heatmap_size``
     - Size of the heatmap used for coordinate regression tasks.

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
     - Enables fixed random seeds to improve experiment reporducibility.

Unified CLI Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After running the unified CLI command, **DeepTune** will generate an output directory containing the results of your experiment. The structure of the output directory will be as follows:

.. code-block:: text

   output_directory
   ├── deeptune_coordinate_regression-<yyyymmdd_hhmm>-exp<int>
       ├── data_splits_<yyyymmdd_hhmm>
       │   ├── train_split.parquet
       │   ├── val_split.parquet
       │   └── test_split.parquet
       │   └── label_mapping.json
       │   └── split_summary.json
       ├── coordinate_regression_output_<model_version>_<yyyymmdd_hhmm>
       │   └── test_log.txt
       │   └── cli_arguments.json
       │   └── model_weights.pth
       │   └── trainval_performance_log.csv