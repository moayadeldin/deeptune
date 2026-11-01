.. _textconfig.rst:

Text Classification Reference
------------------------------------------------------

Dataset and Task Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--modality``
     - Specifies the input data modality (i.e., ``text``).
   * - ``--mode``
     - Defines the machine learning task (i.e., ``cls``
       for classification only).
   * - ``--target``
     - Name of the target column containing the prediction labels.
   * - ``--num_classes``
     - Number of target classes for classification. Only needed for BERT as GPT-2 can infer the number of classes from the dataset.

Model and Training Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``--model``
     - Specifies the model version. Either ``bert`` or ``gpt2``.
   * - ``--batch_size``
     - Number of samples processed in each training batch.
   * - ``--num_epochs``
     - Number of training epochs.
   * - ``--learning_rate``
     - Learning rate used by the optimizer.

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
   * - ``--use-peft``
     - Enables parameter-efficient fine-tuning (PEFT). Only applicable for BERT as GPT-2 does not support PEFT in DeepTune.
   * - ``--raw-data``
     - Enables processing of CSV/XLSX (non-parquet) input data.
   * - ``--freeze-backbone``
     - Freezes the backbone parameters during training,
       allowing the prediction head to be trained.

.. note :: 
    The predefined hyperparameters for text classification are similar to those used for image classification and regression tasks. When running the unified CLI command without specifying certain hyperparameters, **DeepTune** will automatically apply these predefined values.

Raw Data Handling
~~~~~~~~~~~~~~~~~~
**DeepTune** supports CSV and XLSX formats for text classification tasks. When using raw data, the library automatically handles the conversion to Parquet format, ensuring efficient processing and compatibility with the training pipeline.

Unified CLI Output Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
After running the unified CLI command, **DeepTune** will generate an output directory containing the results of your experiment. The structure of the output directory in case of GPT2 will differ from that of BERT given the training weights that are generated. The structure will be as follows:

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
   ------------------------------------------------------------------------
       // GPT2
       ├── trainval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── tokenizer
       │      ├── merges.txt
       │      ├── vocab.json
       │      └── tokenizer_config.json
       │      └── special_tokens_map.json
       │   ├── bert_model
       │      ├── model.safetensors
       │      ├── config.json
       │   ├── cli_arguments.json
       │   ├── model_weights.pth
       │   ├── model_config.json
       │   ├── training_details.json
       │   ├── training_log.csv

       
       // BERT
       │   ├── trainval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── tokenizer
       │      ├── merges.txt
       │      ├── vocab.json
       │      └── tokenizer_config.json
       │      └── special_tokens_map.json
       │   ├── bert_model
       │      ├── adapter_config.json
       │      ├── adapter_model.safetensors
       │   ├── cli_arguments.json
       │   ├── model_weights.pth
       │   ├── model_config.json
       │   ├── training_details.json
       │   ├── training_log.csv   
   ------------------------------------------------------------------------
       ├── eval_output_<model_version>_<yyyymmdd_hhmm>
       │   ├── full_metrics.json
       │   ├── cli_arguments.json
       ├── embed_output_<model_version>_<yyyymmdd_hhmm>
           ├── <model_version>_cls_embeddings.parquet
           ├── cli_arguments.json 
           ├── embeddings_details.json
