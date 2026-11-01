.. _textsentiment:

Text Classification
=====================

Text Classification is a natural language processing (NLP) task that involves determining the sentiment or emotional tone expressed in a piece of text. It is commonly used to analyze sentiment analysis, customer reviews, social media posts, and other forms of textual data to understand public opinion, customer satisfaction, and brand perception.

Running Text Classification with DeepTune
-----------------------------------------

**DeepTune** supports the following Text Classification models:

- `Bidirectional Encoder Representations from Transformers (BERT) <https://arxiv.org/abs/1810.04805>`_ (HuggingFace Transformers). The model versions supported are: 'bert_multilingual_uncased' which is called 'bert' in the CLI.

- `Generative Pre-trained Transformer 2 (GPT-2) <https://huggingface.co/docs/transformers/en/model_doc/gpt2>`_ (HuggingFace Transformers). The model versions supported are: 'gpt2' which is called 'gpt2' in the CLI.

.. note ::

For BERT, the number of classes must be specified in the CLI command, while for GPT-2, it is not required as it is a generative model that can handle multiple classes without explicit specification. Execute the following command to run sentiment analysis using BERT in **DeepTune**:

.. code-block:: console

    (.deeptune) $ python -m deeptune \
        --modality text \
        --mode cls \
        --model bert \
        --df <dataset_path> \
        --target <target_column_name> \
        --batch_size <batch_size> \
        --num_classes <num_classes> \
        --num_epochs <num_epochs> \
        --learning_rate <learning_rate> \
        --out <output_directory> \
        --fixed-seed \
        --use-peft \
        --freeze-backbone \
        --raw-data \

For GPT-2, the **DeepTune** command is as follows:
 
.. code-block:: console

    (.deeptune) $ python -m deeptune \
        --modality text \
        --mode cls \
        --model gpt2 \
        --df <dataset_path> \
        --target <target_column_name> \
        --batch_size <batch_size> \
        --num_epochs <num_epochs> \
        --learning_rate <learning_rate> \
        --out <output_directory> \
        --fixed-seed \
        --freeze-backbone \
        --raw-data \

.. note::

    There is no need to specify the ``--added_layers`` and ``--embed_size`` switches when using GPT-2 as they are already statically fixed due to design constraints, to 2 layers and 1000 respectively.

For a detailed information on the configuration parameters and execution options for image classification and regression tasks, please refer to the :doc:`../configurations/textconfig` section.