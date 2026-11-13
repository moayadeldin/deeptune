# *DeepTune*

[![deeptune tests](https://github.com/moayadeldin/deeptune/actions/workflows/test.yml/badge.svg)](https://github.com/moayadeldin/deeptune/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/deeptune/badge/?version=latest)](https://deeptune.readthedocs.io/en/latest/)

**DeepTune** is a comprehensive, automated deep learning toolkit supporting computer vision, natural language processing, video analysis, tabular data, and time series. It provides state-of-the-art algorithms and ready-to-use CLI and GUI interface, enabling users to exploit machine learning workflows across diverse applications, including medical imaging, natural language understanding, and action classification. A detailed documentation of DeepTune is available through [docs](https://deeptune.readthedocs.io/en/latest/).

## Features

- **Unified Pipeline**: An automated framework supporting multiple tasks across different domains.
- **Multimodal Support**: Supports image, text, video, tabular, and time-series datasets.
- **Extensive Model Support**: Supports a wide range of models for +40 state-of-the-art deep learning models for vision, NLP, tabular, time-series, and video tasks.
- **Flexible Training Paradigms**: Supports multiple training and fine-tuning options, including end-to-end training, transfer learning, linear readout, and parameter-efficient fine-tuning (PEFT) with hyperparameter tuning, including projection-layer configuration.
- **User-Friendly Interface**: Provides both command-line interface (CLI) and graphical user interface (GUI) for ease of use.

## **DeepTune** Supported Tasks & Models
<p align="center">
  <img width="1822" height="826" alt="deeptune_models" src="https://github.com/user-attachments/assets/750eec30-0b85-4472-a0fe-002a9eb5d4c0" />
</p>

## How to Install?

Given that your device is running on Python 3.10:

```bash
    $ python -m venv deeptune
    $ source deeptune/bin/activate
```

then:
```bash
    (.deeptune) $ git clone https://github.com/moayadeldin/deeptune
    (.deeptune) $ pip install -r requirements.txt
```

or:
```bash
    (.deeptune) $ git clone https://github.com/moayadeldin/deeptune
    (.deeptune) $ cd deeptune
    (.deeptune) $ pip install -e .
```

## Getting Started
1. To start on the GUI:
```bash
    (.deeptune) $ cd deeptune
    (.deeptune) $ python.exe desktop/app.py

```
<p align="center">
  <img width="1907" height="1010" alt="deeptune_gui" src="https://github.com/user-attachments/assets/d6da9d1b-79f5-42f1-940b-7a6a7536c3da" />
</p>

2. To start on the CLI:
```bash
    (.deeptune) $ python -m deeptune --modality <model_name> --model <model_name> --df <dataset_path>
```

Each task (Image Classification & Regression, Sentiment Analysis, Clip-Level Action Classification, ...) has its own set of supported models and specific parameters that must be set in order to exeucte DeepTune properly. You can find the list of supported models with their corresponding parameters in the respective task section in the documentation.

## Documentation

DeepTune is being under active development and mainteneance with a user-friendly comprehensive documentation for easier usage. The documentation can be accessed [here](https://deeptune.readthedocs.io/en/latest/).

Acknowledgements
----------------
This software package was developed as part of work done by Moayadeldin Hussain at Medical Imaging Bioinformatics lab under the supervision of Jacob Levman at St. Francis Xavier Univeristy, Nova Scotia, Canada.

## Citation

If you find *DeepTune* useful, please give us a star ⭐ on GitHub for support.

Also if you find this repository helpful, please cite it as follows:

```bibtex
 @software{DeepTune,
 author  = {Moayadeldin Hussain, Ahmed Saad, John Kendall, Youssoff Mahfouz, Yazan Hamad, Xuchen Guo, and Jacob Levman},
 title   = {DeepTune: Cutting-edge Tool automating state-of-the-art deep learning models for cross-modal applications},
 year = {2026},
 url = {https://github.com/moayadeldin/deeptune},
 }
