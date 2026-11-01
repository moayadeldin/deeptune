.. _configpreface.rst: 

Configuration and Parameter Reference section
=============================================

This section provides a comprehensive reference for the configuration and parameters used in **DeepTune**. It serves as a guide for users to understand the available options and their implications on model training and evaluation.

Except the input dataset path and the output directory path, all other parameters are task-specific and can be found in the respective task section. The following parameters are common across all tasks:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description

   * - ``--df``
     - Path to the input dataset.
    
   * - ``--out``
     - Path to the output directory where results will be saved.

Task-Specific Configuration
---------------------------

Select a task below to explore its supported configuration
parameters and execution options.

.. toctree::
   :hidden:
   :maxdepth: 1

   imageconfig

* :doc:`imageconfig` — Explore image-based models configurations and
  fine-tuning parameters.

