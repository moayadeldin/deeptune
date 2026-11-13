Installation
============

Prerequisites
-------------

Kindly note that you have to install PyTorch version that matches your CUDA and GPU specifications. You can find the corresponding versions for your GPU `here <https://pytorch.org/get-started/locally/>`_.

The software was heavily tested on both Windows 11 and Ubuntu 22.04 with Python 3.10. We expect **DeepTune** to function properly on either of these two operating systems. **DeepTune** was not tested on MacOS, consider using a virtual machine or Docker container if you are using MacOS.

Creating Virtual Environment (Recommended)
-------------------------------------------
.. note::

   It is recommended to use a virtual environment to avoid dependencies issues. You can directly install the ``requirements.txt`` if you want to avoid using virtual envs.


On Windows 11, Python 3.10 and Conda:

.. code-block:: console

    $ conda create -n deeptune python=3.10
    $ conda activate deeptune

or using venv ensuring that you have Python 3.10 installed:

.. code-block:: console

    $ python -m venv deeptune
    $ source deeptune/bin/activate

Install dependencies
--------------------

.. code-block:: console

    (.deeptune) $ pip install -r requirements.txt

