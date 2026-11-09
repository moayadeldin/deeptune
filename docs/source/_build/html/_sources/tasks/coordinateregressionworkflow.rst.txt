
Coordinate Regression Workflow Components
=========================================

DeepTune provides a dedicated workflow for image-based coordinate
regression, organized into model construction, training, validation,
and evaluation. The workflow uses pretrained vision backbones with
a heatmap-based regression head to predict spatial coordinates.


Model Training
--------------

**Entry point:** ``coordinate_regressors.regressor.main``

The main execution function initializes the regression model,
splits the dataset into training, validation, and testing subsets,
and coordinates model training and evaluation.

.. py:currentmodule:: coordinate_regressors.components.backbone

.. py:class:: RegressorBackbone(backbone="resnet18", pretrained=True, heatmap_size=28)

   Constructs a coordinate regression model using a configurable
   torchvision vision backbone and a heatmap prediction head.

   :param backbone: Identifier of the selected vision architecture.
   :param pretrained: Whether to initialize the backbone with
       pretrained weights.
   :param heatmap_size: Spatial dimensions of the output heatmap.

   The model extracts image features using the selected backbone,
   applies transposed convolutions to generate a single-channel
   heatmap, and estimates normalized coordinates using a
   differentiable spatial soft-argmax operation.

   .. py:method:: forward(x)

      Performs the forward pass and returns predicted coordinates
      of shape ``(B, 2)`` and heatmaps of shape
      ``(B, heatmap_size, heatmap_size)``.

      The predicted coordinates are normalized to ``[0, 1]``.


Coordinate Regression Loss
--------------------------

**Entry point:** ``coordinate_regressors.components.loss.coord_loss``

.. py:currentmodule:: coordinate_regressors.components.loss

.. py:function:: coord_loss(pred_coords, target_coords, loss_mode="mse")

   Computes the regression loss between predicted and target
   coordinates.

   :param pred_coords: Predicted coordinates of shape ``(B, 2)``.
   :param target_coords: Ground-truth coordinates of shape ``(B, 2)``.
   :param loss_mode: Loss function, either ``mse`` or ``mae``.

   The workflow uses Mean Squared Error (MSE) by default,
   with Mean Absolute Error (MAE) available as an alternative.


Training and Validation
-----------------------

**Entry point:** ``coordinate_regressors.regressor.train``

The training function prepares image DataLoaders and optimizes
the regression model using Adam and coordinate-based MSE loss.

.. py:currentmodule:: coordinate_regressors.regressor

.. py:function:: train(num_epochs, learning_rate, batch_size, train_path, val_path, args, coordinate_regression_output, model)

   Executes the training loop, performs validation after each
   epoch, logs training and validation losses, and saves model
   parameters to ``model_weights.pth``.

.. py:function:: validate(val_path, batch_size, model)

   Evaluates the model on the validation dataset without
   gradient computation and returns the average validation loss.

.. py:function:: saveModel(model, performance_logger, path)

   Saves the model parameters to the specified checkpoint path.

Training and validation losses are recorded in
``trainval_performance_log.csv``.


Model Evaluation
----------------

**Entry point:** ``coordinate_regressors.regressor.test``

The evaluation function uses the trained model to predict
coordinates on the held-out testing dataset.

.. py:currentmodule:: coordinate_regressors.regressor

.. py:function:: test(test_path, batch_size, model)

   Performs inference without gradient computation and computes
   the average coordinate regression loss over the testing
   batches.

   The final test loss is saved to ``test_log.txt``.
