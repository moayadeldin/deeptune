Using `DeepTune` for Training
==============================

Images
------


The following is the generic CLI structure of running `DeepTune` on images dataset stored in Parquet file as bytes format for training:

.. code-block:: console

    $ python -m trainers.vision.train \
    --train_df <str> \
    --val_df <str> \
    --model_version <str> \
    --batch_size <int> \
    --num_classes <int> \
    --num_epochs <int> \
    --learning_rate <float> \
    --added_layers <int> \
    --embed_size <int> \
    --out <str>
    --mode <cls_or_reg> \
    [--fixed-seed] \
    [--use-peft] \
    [--freeze-backbone]

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``--train_df <str>``
     - Path to your training dataset (must be a parquet file).

   * - ``--val_df <str>``
     - Path to your validation dataset (must be a parquet file).

   * - ``--<model>_version <str>``
     - The model to use along with its respective architecture version.  
       You may refer to the *Supported Models* table for available options.

   * - ``--batch_size <int>``
     - Number of samples per batch.

   * - ``--num_classes <int>``
     - Number of classes in your dataset.

   * - ``--num_epochs <int>``
     - Number of training epochs.

   * - ``--learning_rate <float>``
     - Learning rate used for optimization.

   * - ``--added_layers <int>``
     - Number of layers added on top of the model for transfer learning (either with or without using PeFT).  
       Only values **1** or **2** are supported currently.

   * - ``--embed_size <int>``
     - Size of the intermediate embedding layer (applicable when using two added layers).

   * - ``--out <str>``
     - Path to the directory where you want to save the results.

   * - ``--fixed-seed``
     - *(Flag)* Ensures that a fixed random seed is set for reproducibility.

   * - ``--mode <str>``
     - Task mode: either ``cls`` for classification or ``reg`` for regression.

   * - ``--use-peft``
     - *(Flag)* Enables **Parameter-Efficient Fine-Tuning (PeFT)**.

   * - ``--freeze-backbone``
     - *(Flag)* Determines whether to train only the added layers or update all model parameters during training.

.. note::
    For using PeFT just add the `--use-peft` switch to the previous command.

For example, suppose that we want to train our model with ResNet18, and apply transfer learning to update the whole model's weights, and an embedding layer of size 1000. Hence, we run the command as follows:

.. code-block:: console

    $ python -m trainers.vision.train --train_df <str> --val_df <str> --model_version resnet18 --batch_size 4 --num_classes 2 --num_epochs 10 --learning_rate 0.0001 --added_layers 2 --embed_size 1000 --out <str> --mode cls --fixed-seed

If everything is set correctly, you should expect an output in the same format:

.. code-block:: text

   > os.environ['PYTHONHASHSEED'] set to 42.
   > np.random.seed(42) set.
   > torch.manual_seed(42) set.
   > torch.cuda.manual_seed(42) set.
   > torch.cuda.manual_seed_all(42) set.
   > torch.backends.cudnn.benchmark set to False.
   > torch.backends.cudnn.deterministic set to True.
   > Dataset is loaded!
   > Data splits have been saved and overwritten if they existed.
   > The Trainer class is loaded successfully.

   > 4%|████    | 459/855 [00:17<01:07,  5.89it/s, loss=0.43]

Text
------

Since ``DeepTune`` currently supports only two models for text classification, the way they are called in the CLI differs from that of image models. Apart from this, the CLI structure remains largely the same:

.. note::
    GPT-2 model does not support PeFT right now in ``DeepTune``.

.. code-block:: console
    $ python -m trainers.nlp.[train_multilinbert/train_gpt2] \
    --train_df <str> \
    --val_df <str> \
    --batch_size <int> \
    --num_classes <int> \
    --num_epochs <int> \
    --learning_rate <float> \
    --added_layers <int> \
    --embed_size <int> \ 
    [--fixed-seed]
    [--freeze_backbone]

.. note::
    There is no need to specify the `--added_layers` and `--embed_size` switches with using GPT-2 **as they are already statically fixed due to design constraints**.

Tabular
-------

Currently, ``DeepTune`` offers only support for GANDALF (Gated Adaptive Network for Deep Automated Learning of Features) model to provide predictions on your own tabular data. You can read more about GANDALF through the paper `here <https://arxiv.org/abs/2207.08548>`_.

The generic CLI workflow for applying GANDALF in ``DeepTune`` requires specifying certain columns before training can begin, which is mainly determining the continuous columns in your dataset and the categorical columns as an input.

.. note::
    Currently, GANDALF implementation does not support transfer learning in the way we commonly applied to images or text above. Instead, it follows the standard training scheme, starting from scratch.

.. code-block:: console

    $ python -m trainers.tabular.train_gandalf \
    --train_df <str> \
    --val_df <str> \
    --batch_size <int> \
    --num_epochs <int> \
    --learning_rate <float> \
    --out <str>
    [--fixed-seed] \
    --categorical_cols \
    --continuous_cols \
    --tabular_target_column <str> \
    --gflu_stages <int>

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``--categorical_cols <str> (space-separated)``
     - Column names of the categorical fields to treat differently.

   * - ``--continuous_cols <str> (space-separated)``
     -  Column names of the numeric fields.

   * - ``--tabular_target_column <str>``
     -  Target column within the dataset.

   * - ``--gflu_stages <int>``
     - Number of GFLU stages for GANDALF.


.. note::
    ``--gflu_stages`` is a hyperparameter related to internal GANDALF working, according to the documentation on `PyTorch Tabular <https://pytorch-tabular.readthedocs.io/en/latest/apidocs_model/#pytorch_tabular.models.GANDALFConfig>`_, it is the number of layers in the feature abstraction layer. The documentation defaults to 6 and we advise the same.

Training Output
---------------

After training completes, you may find the results in the directory specified with the `--out` directory. Alternatively, ``DeepTune`` will create an output directory named  `deeptune_results` (if it does not already exist). Inside this directory, the results are organized in a subfolder using the following naming convention: ``trainval_output_<FINETUNED/PEFT>_<model_version>_<mode>_<yyyymmdd_hhmm>`` with the following output:

.. code-block:: console

    output_directory
    ├── trainval_output_<FINETUNED/PEFT>_<model_version>_<mode>_<yyyymmdd_hhmm>
        └── cli_arguments.json
        └── model_weights.pth
        └── training_log.csv
        └── training_details.json

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``cli_arguments.json``
     - Records the CLI arguments you entered to run ``DeepTune``.

   * - ``model_weights.pth``
     - The fine-tuned model weights (used later for evaluation).

   * - ``training_log.csv``
     - A performance log reporting training and validation accuracies and errors for each epoch.

   * - ``training_details.json``
     - Stores the amount of time needed between starting and completing the training.



For text GPT-2 and BERT models, instead of the ``model_weights.pth`` file, you may find a whole subdirectory containing the tokenizer and model weights files.

.. code-block:: console

    output_directory
    └── trainval_output_<BERT/GPT2>_<yyyymmdd_hhmm>
        ├── tokenizer
        │   └── ...
        ├── model
        │   └── ...
        ├── model_weights.pth
        └── training_log.csv

For tabular data using GANDALF, the directory will be named as ``trainval_output_<GANDALF>_<mode>_<yyyymmdd_hhmm>`` with the weights stored in ``GANDALF_model`` subdirectory. 
