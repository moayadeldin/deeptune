Using `DeepTune` for Knowledge Extraction
=========================================

After we trained and evaluated our fine-tuned model, the user may need to obtain intermediate knowledge representation of their data for further postprocessing according to their own choice. This practice is widely adpoted in medical imaging applications, where fine-tuned deep learning models serve as encoders for both image data and associated clinical metadata.

We integrate `DeepTune` with `df-analyze <https://github.com/stfxecutables/df-analyze>`_ with both being part of the MIB Lab @StFX open-source software contributions as we show in :doc:`dfanalyze` section.

Images
-------

The following is the generic CLI structure of running DeepTune for embeddings extraction of image datasets:

.. code-block:: console

    $ python -m embed.vision.embed \
    --df <path_to_df> \
    --batch_size <int> \
    --num_classes <int> \
    --out <str> \
    --model_version <str> \
    --model_weights <str> \
    --added_layers <int> \
    --embed_size <int> \
    --mode <cls_or_reg> \
    --use_case <finetuned_or_pretrained_or_peft> \

The ``--use_case`` argument specifies on which use case you want to use DeepTune for:

  - `pretrained`: Using the exact weights of the model as it is without any further training. This option allows you to use DeepTune with skipping the training and evaluation parts (You don't need to specify ``--added_layers``, ``--embed_size``, and ``--model_weights``).
  - `finetuned`: If you ran DeepTune for Transfer Learning without PeFT.
  - `peft`: If you ran DeepTune for Transfer Learning with PeFT.

  .. note::
    You feed the evaluator here the same ``--added_layers`` and ``--embed_size`` you used for your previous training run of DeepTune. Otherwise, a mismatch error will occur.

If everything is set correctly, and evaluation is done, you should expect an output in the same format:

Text
----

The following is the generic CLI structure of running DeepTune for embeddings extraction of text datasets:

.. code-block:: console

    $ python -m embed.nlp.<gpt2/multilingualbert>_embeddings \
    --batch_size <int> \
    --num_classes <int> \
    --df <path_to_df> \
    --model_weights <str> \
    --added_layers <int> \
    --embed_size <int> \
    --use_case <finetuned_or_pretrained_or_peft> \

.. note::

    We recall the same note mentioned in :doc:`training` and :doc:`evaluation` sections which is that the arguments ``--added_layers`` and ``--embed_size`` for GPT-2 model are set by default due to design constraints.

Tabular
-------

The following is the generic CLI structure of running DeepTune for embeddings extraction of text datasets:

.. code-block:: console

    $ python -m embed.vision.embed \
    --df <path_to_df> \
    --batch_size <int> \
    --out <str> \
    --tabular_target_column <str> \
    --model_weights <str> \
    --categorical_cols \
    --continuous_cols \

.. note::

    As GANDALF relies on the standard training scheme without applying transfer learning, we do not use Parameter Efficient Fine-Tuning (PeFT), Fine-Tuning, or Pretrained options. Indeed, we only use the already trained model's weights to obtain the embeddings.


For images, text, or tabular data embeddings, you you may find the results in the directory specified with the `--out` or default `DeepTune` directory as follows: 


