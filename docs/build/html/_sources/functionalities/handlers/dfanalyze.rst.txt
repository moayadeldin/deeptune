[EXTRA] Integration with df-analyze
====================================

`df-analyze <https://github.com/stfxecutables/df-analyze>`_ is a command-line tool developed in the same Medical Imaging Bioinformatics lab at St. Francis Xavier University for automating Machine Learning tasks on small to medium-sized tabular datasets (less than about 200 000 samples, and less than about 50 to 100 features) using Classical ML algorithms.

After you successfully allocate your embeddings file, either after running `DeepTune` on image dataset or text one, you may install df-analyze —instructions on how to do that is found on the software repository link— and run the following command:

.. code-block:: console

   $ python df-analyze.py --df "path\test_set_<use_case>_<model>_embeddings_cls.parquet" --outdir = ./deeptune_results --mode=classify --target label --classifiers lgbm rf sgd knn lr mlp dummy --embed-select none linear lgbm

If you ran df-analyze for regression task on images, you may change the command to be as follows:

.. code-block:: console

    $ python df-analyze.py --df "path\test_set_<use_case>_<model>_embeddings_reg.parquet" --target label --mode=regress --regressors knn lgbm elastic lgbm sgd dummy mlp --feat-select wrap --outdir=./deeptune_results

.. note::

    Do not forget to change the ``--df`` switch value according to the path of the embeddings file. More information can be found on the `df-analyze <https://github.com/stfxecutables/df-analyze>`_ repository page.