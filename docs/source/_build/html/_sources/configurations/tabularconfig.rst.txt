.. _tabularconfig:

Tabular Classification & Regression Parameters Reference
----------------------------------------------------------

Dataset and Task Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--modality``
     - Specifies the input data modality (i.e., ``tabular``).
   * - ``--mode``
     - Defines the machine learning task for TabPFN (i.e., ``cls``
       for classification and ``reg`` for regression).
   * - ``--type``
     - Defines the machine learning task for GANDALF (i.e., ``classification``
       for classification and ``regression`` for regression).
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
   * - ``--num_epochs``
     - Number of training epochs.
   * - ``--learning_rate``
     - Learning rate used by the optimizer.
   * - ``--continuous_cols``
     - Names of the continuous columns in the dataset (only for GANDALF).
   * - ``--categorical_cols``
     - Names of the categorical columns in the dataset (only for GANDALF).
   * - ``-gflu_stages``
     - Number of Gated Feature Learning Unit (GFLU) stages in the GANDALF model (only for GANDALF).


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
   * - ``--finetuning-mode``
     - Enables fine-tuning mode for TabPFN, allowing the model to adapt to new data while retaining knowledge from pre-trained weights.
   * - ``--raw-data``
     - Enables processing of raw (non-parquet) input data.

Predefined Hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~
To simplify the initial experience with **DeepTune**, a set of predefined hyperparameters is used. These hyperparameters have been chosen based on common practices and are intended to provide a good starting point for most experiments. When you run the unified CLI command without specifying certain hyperparameters, **DeepTune** will automatically apply these predefined values. The predefined hyperparameters are as follows:

- **Learning Rate**: 0.0001
- **Number of Epochs**: 10
- **GFLU Stages**: 6 (only for GANDALF)
- **Train/Validation/Test Split Ratios**: 70% / 10% / 20%
- **Disable Numerical Encoding**: False (i.e., numerical encoding is applied by default as part of preprocessing).

Unified CLI Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After running the unified CLI command, **DeepTune** will generate an output directory containing the results of your experiment. The structure of the output will differ whether you are using TabPFN or GANDALF. The output directory will be as follows:



.. code-block:: text

   output_directory
   └── deeptune-<yyyymmdd>-exp<int>
       ├── <modality>_dataset_<yyyymmdd_hhmm>.parquet
       ├── data_splits_<yyyymmdd_hhmm>
       │   ├── train_split.parquet
       │   ├── val_split.parquet
       │   ├── test_split.parquet
       │   ├── test_indices.csv
       │   ├── label_mapping.json
       │   └── split_summary.json
       │
       │   // GANDALF
       ├── trainval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── GANDALF_model
       │   │   ├── callbacks.sav
       │   │   ├── config
       │   │   ├── custom_params.sav
       │   │   ├── datamodule.sav
       │   │   └── model.ckpt
       │   ├── cli_arguments.json
       │   ├── training_details.json
       │   └── training_log.csv
       │
       │   // TabPFN
       ├── train_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── cli_arguments.json
       │   ├── <training_or_finetuning>_details.json
       │   ├── <trained_or_finetuned>_TABPFN_<cls_or_reg>.joblib
       │   └── training_log.csv
       │
       ├── eval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── full_metrics.json
       │   ├── cli_arguments.json
       │   └── evaluation_details.json
       └── embed_output_<model_version>_<yyyymmdd_hhmm>
           ├── <model_version>_<cls_or_reg>_embeddings.parquet
           ├── cli_arguments.json
           └── embeddings_details.json

