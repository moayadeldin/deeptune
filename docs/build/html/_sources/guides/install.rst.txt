Installation
============

Prerequisites
-------------

Kindly note that you have to install PyTorch version that matches your CUDA and GPU specifications. You can find the corresponding versions for your GPU `here <https://pytorch.org/get-started/locally/>`_.

The software was heavily tested on Windows 11 with Python 3.8, and Ubuntu 20.04 with Python 3.10. We expect **DeepTune** to function properly on either of these two operating systems.

Creating Virtual Environment (Recommended)
-------------------------------------------
.. note::

   It is recommended to use a virtual environment to avoid dependencies issues. You can directly install the ``requirements.txt`` if you want to avoid using virtual envs.


On Windows 11, Python 3.8 and Conda 24.9.2:

.. code-block:: console

    $ conda create -n deeptune
    $ conda activate deeptune

or using venv on Ubuntu 20.04:

.. code-block:: console

    $ python -m venv deeptune
    $ source deeptune/bin/activate

Install dependencies
--------------------

.. code-block:: console

    (.deeptune) $ pip install -r requirements.txt

