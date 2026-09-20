Using Coordinate Regressors for Images in **DeepTune**
=======================================================

Coordinate regressors are neural networks that learn to predict the coordinates of specific objects of interest in images. The regressor is built on top of a backbone ResNet18 architecture with a heatmap regression map that is dedicated to preserving the spatial information of the input images. The heatmap upsamples the 7 x 7 feature map produced by the ResNet18 backbone to by default a 56 x 56 heatmap (or the specified size by the user), through multiple deconvolutional operations, which is then used to predict the coordinates of the objects of interest. The regressor is trained using a mean squared error loss function, which measures the difference between the predicted coordinates and the ground truth coordinates.

Similar to the other functionalities in DeepTune, the input is expected to be a parquet file containing the images in byte-format, and the coordinates of the objects of interest in that specfiic image (i.e., [image, coord] pair). 

To use the coordinate regressor in **DeepTune**, you can follow the following command:

.. code-block:: console

    $ python -m coordinate_regressors.regressor \
    --df <str> \
    --train_size <float> \
    --val_size <float> \
    --test_size <float> \
    --num_epochs <int> \
    --learning_rate <float> \
    --heatmap_size <int> \
    --fixed_seed <bool> \
    --out <str> \

.. note::
    The ``--heatmap_size`` flag is optional and can be set to any integer value to specify the size of the heatmap output for coordinate regression. By default, it is set to 56. The ``--fixed_seed`` flag is optional and can be set to `True` to use a fixed seed for reproducibility, or `False` to use a random seed. By default, it is set to `True`.

After training is done, the output directory specified with the ``--out`` flag will contain the following files:

.. code-block:: console

    output_directory
       ├── data_splits_<yyyymmdd_hhmm>
       │   ├── train_split.parquet
       │   ├── val_split.parquet
       │   ├── test_split.parquet
       │   └── test_indices.csv
       ├── coordinate_regression_output_resnet18_<yyyymmdd_hhmm>
       │   ├── cli_arguments.json
       │   ├── model_weights.pth
       │   ├── trainval_performance_log.csv
       │   └── test_log.txt
