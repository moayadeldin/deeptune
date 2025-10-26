Using `DeepTune` for Evaluation
=========

Images
------

Evaluating your model on a separete holdout test dataset is referred to as evaluation. The reason is to simply not confuse the terms as testing in `DeepTune` documentation context also refers to testing the model functionality (e.g, writing test cases).

After using `DeepTune` to apply transfer learning on one of the models the package support, now we need to evaluate the performance of the tuned model for images.

The following is the generic CLI structure of running DeepTune for evalaution of image datasets:

.. code-block:: console

  $ python -m evaluators.vision.evaluate \
  --eval_df <str> \
  --model_version <str> \
  --batch_size <int> \
  --num_classes <int> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  --out <str> \
  [--use-peft] \
  [--freeze-backbone]

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``--eval_df <str>``
     - Path to your test dataset. It should be the ``test_split_<yyyymmdd>_<hhmm>.parquet `` you got from the previous `DeepTune` for splitting data run.
   * - ``--model_weights <str>``
     - Path to your model's weights. It should be ```model_weights.pth`` you got from the previous `DeepTune` for training run.

.. note::
    If you used one of the switches ``--freeze_backone`` or ``--use_peft`` or both in the previous run, you should use them while doing your evaluation here again. Also, you feed the evaluator here the same ``--added_layers`` and ``--embed_size`` you used for your previous training run of DeepTune. Otherwise, a mismatch error will occur.

If everything is set correctly, and evaluation is done, you should expect an output in the same format:

.. code-block:: text

   > Model into the path is loaded. 100%|█████████████████████████████████████████████████████| 107/107 [00:10<00:00, 10.03it/s] 98.18713450292398 0.05557631243869067 INFO | Test accuracy: 98.18713450292398% 
   > {'loss': 0.05557631243869067, 'accuracy': 0.9818713450292398, '0': {'precision': 1.0, 'recall': 0.967479674796748, 'f1-score': 0.9834710743801653, 'support': 123.0},'auroc': 0.9997672516861436}
   > Test results saved successfully!

Text
-----

In `DeepTune`, the text SoTA models save the weights of both the models, and the tokenizers. The tokenizer role is to split sentences into smaller units (we call them tokens) that can be more easily assigned meaning. On the other hand, the model is responsible for handling the part of interpreting these tokens.

The generic CLI structure of running *DeepTune* for evalaution of text datasets:

.. code-block:: console

    $ python -m evaluators.nlp.evaluate_<multilingualbert/gpt> \
     --eval_df <str> \
    --batch_size <int> \
    --num_classes <int> \
    --model_weights <str> \
    --added_layers <int> \
    --embed_size <int> \
    --out <str> \
    [--use-peft] \
    [--freeze-backbone]

.. note::
    For the ``--model_weights`` argument, we feed the whole output directory we got from running `DeepTune` for training (``trainval_output_<BERT/GPT2>_<yyyymmdd_hhmm>``)

.. note:: 
    For GPT-2 model, the switches ``--added_layers`` and ``embed_size`` are set by default as we tweaked the model architecture in order to be properly ready for training due to design constrains, so you don't have to set these to a specific input.


Tabular
-------
The generic CLI structure of running *DeepTune* for evalaution of text datasets using GANDALF is:

.. code-block:: console

    $ python -m evaluators.tabular.evaluate_gandalf \
    --eval_df <str> \
    --model_weights <str>
    --out <str>
.. note::
    You feed the path to ``GANDALF_model`` subdirectory that you obtained after training as a parameter to ``--model_weights``.


Evaluation Output
-----------------


After evaluation is done, you may find the results in the directory specified with the ``--out`` directory or ``deeptune_results`` initiated in your `DeepTune` path. Inside this folder, you will find the following output directory:

.. code-block:: console

    output_directory
    ├── eval_output_FINETUNED/PEFT-<model_version>_<yyyymmdd>_<hhmm>
        └── cli_arguments.json
        └── full_metrics.json
        └── evaluation_details.json

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``cli_arguments.json``
     - Records the CLI arguments you entered to run ``DeepTune``.

   * - ``full_metrics.json``
     - The full metrics as appeared to you in the CLI while using the model.

   * - ``evaluation_details.json``
     - Stores the amount of time needed between starting and completing the training.

Similarly, the output of the text and tabular datasets will have the same structure but with respectively different directory namings.
