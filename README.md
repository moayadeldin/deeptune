# DeepTune

[![deeptune tests](https://github.com/moayadeldin/deeptune-scratch/actions/workflows/test.yml/badge.svg)](https://github.com/moayadeldin/deeptune-scratch/actions/workflows/test.yml)

**DeepTune** is a full compatible library to automate Computer Vision and Natural Language Processing algorithms on diverse images and text datasets.

As a cutting-edge software library that has been specifically designed for use in different Machine Learning Tasks, inclduing but not limited to image classification, transfer learning, and embedding extraction. 

**DeepTune** is currently going under the process of extensive testing, and offers multiple features including ability to apply transfer learning via fine-tuning for advanced classification algorithms for images, and texts, including Parameter Efficient Fine-Tuning with LoRA, and latent feature extraction as embedding vectors. This offers a massive assistance for users to take full advantage of what their case studies may offer with simple commands.

## Features

- Fine-tuning SoTA Computer Vision algorithms for Image Classification.
- Fine-tuning SoTA NLP algorithms.
- Fine-tuning SoTA Time Series algorithms.
- Providing PEFT with LoRA support for Computer Vision algorithms implemented, enabling state-of-the-art models that typically require substantial computational resources to perform efficiently on lower-powered devices. This approach not only reduces computational overhead but also enhances performance.
- Extracting meaningful feature embeddings with SoTA algorithms for image and text classification tasks.

## Models DeepTune Supports Up to Date


<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Transfer Learning with Adjustable Embedding Layer?</th>
      <th>Support PEFT with Adjustable Embedding Layer?</th>
      <th>Support Embeddings Extraction?</th>
      <th>Task</th>
      <th>Modality</th>
      <th>Supported Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ResNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>ResNet18, ResNet34, ResNet50, ResNet101, ResNet152</td>
    </tr>
    <tr>
      <td>DenseNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>DenseNet121, DenseNet161, DenseNet169, DenseNet201</td>
    </tr>
    <tr>
      <td>Swin</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>Swin_t, Swin_b, Swin_s</td>
    </tr>
    <tr>
      <td>EfficientNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>EfficientNet-B0, B1, B2, B3, B4, B5, B6, B7</td>
    </tr>
    <tr>
      <td>VGGNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>VGG11, VGG13, VGG16, VGG19</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>ViT_b_16, ViT_b_32, ViT_l_16, ViT_l_32, ViT_h_14</td>
    </tr>
    <tr>
      <td>GPT</td>
      <td>✅</td>
      <td>—</td>
      <td>✅</td>
      <td>Text Classification</td>
      <td>Text</td>
      <td>GPT-2</td>
    </tr>
    <tr>
      <td>BERT</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Sentiment Analysis</td>
      <td>Text</td>
      <td>bert-base-multilingual-cased</td>
    </tr>
    <tr>
      <td>RoBERTa</td>
      <td colspan="4" style="text-align:center; font-weight:bold; font-size:16px;">Only Supports Embedding Extraction</td>
      <td>Text</td>
      <td>XLM-RoBERTa</td>
    </tr>
    <tr>
  <td>DeepAR</td>
  <td colspan="4" style="text-align:center; font-weight:bold; font-size:16px;">Only Supports Time Series Datasets</td>
  <td>Time Series</td>
  <td>DeepAR</td>
</tr>
  </tbody>
</table>

## DeepTune Structure

```plaintext
DeepTune
├── datasets
│   ├── __init__.py
│   ├── image_dataset.py
│   └── text_dataset.py
├── embed
│   ├── nlp
│   │   ├── es_embeddings.py
│   │   └── multilingualbart_embeddings.py
│   │   └── gpt2_embeddings.py
│   └── vision
│       ├── densenet_embeddings.py
│       ├── efficientnet_embeddings.py
│       ├── resnet_embeddings.py
│       └── swin_embeddings.py
│       └── vgg_embeddings.py
│       └── vit_embeddings.py
├── evaluator
│   ├── nlp
│   │   └── evaluate_multilingualbert.py
│   │   └── evaluate_gpt.py
│   └── vision
│       ├── evaluate_densenet.py
│       ├── evaluate_efficientnet.py
│       ├── evaluate_resnet.py
│       ├── evaluate_swin.py
│       ├── evaluate_vgg.py
│       ├── evaluate_vit.py
│       └── evaluator.py
├── src
│   ├── nlp
│   │   ├── E5_roberta.py
│   │   ├── multilingual_bert.py
│   │   └── multilingual_bert_peft.py
│   │   └── gpt2.py
│   └── vision
│       ├── densenet.py
│       ├── densenet_peft.py
│       ├── efficientnet.py
│       ├── efficientnet_peft.py
│       ├── resnet.py
│       ├── resnet_peft.py
│       ├── swin.py
│       └── swin_peft.py
│       ├── vgg.py
│       └── vgg_peft.py
│       ├── vit.py
│       └── vit_peft.py
├── trainers
│   ├── nlp
│   │   └── train_multilingualbert.py
│   │   └── train_gpt2.py
│   └── vision
│       ├── train_densenet.py
│       ├── train_efficientnet.py
│       ├── train_resnet.py
│       └── train_swin.py
│       └── train_vit.py
│       └── train_vgg.py
│       └── trainer.py
├── timeseries
│   ├── deepAR.py
├── tests
│   ├── models
│   │   ├── test_densenet.py
│   │   ├── test_efficientnet.py
│   │   ├── test_resnet.py
│   │   ├── test_swin.py
│   │   ├── test_vgg.py
│   │   ├── test_vit.py
├── .github
│   ├── workflows
│   │   ├── test.yml
├── options.py
├── requirements.txt
└── utilities.py
└── .gitignore
└── update_version.sh
└── VERSION
└── README.md
```


## 📘 User Guide

### 1 Installation & Preface

- [1.1 Prerequisites](#11-prerequisites)
- [1.2 Linux and Windows](#12-linux-and-windows)
- [1.3 Preface](#13-preface)

### 2. Getting Started: Your First DeepTune Run
- [2.1 Using DeepTune for Training](#21-using-deeptune-for-training)
- [2.2 Using DeepTune for Evaluation](#22-using-deeptune-for-evaluation)
- [2.3 Using DeepTune for Embeddings Extraction](#23-using-deeptune-for-embeddings-extraction)
- [ [EXTRA] 2.4 Integration with df-analyze](#..)



## Installation

### 1.1 Prerequisites

Kindly note that you have to install PyTorch version that matches your CUDA and GPU specifications. You can find the corresponding versions for your GPU [here](https://pytorch.org/get-started/locally/).



### 1.2 Linux and Windows

Although the most of development cycle we have been using Windoes 11, DeepTune is expected function properly on Ubuntu 20.04 also.

#### Creating Virtual Environment (Recommended)

> **Note**  
> It is recommended to use a virtual environment to avoid dependencies issues. You can directly install the `requirements.txt` if you want to avoid using virtual envs.

On Windows 11, Python 3.8 and Conda 24.9.2:

```
$ conda create -n deeptune
$ conda activate deeptune
```

#### Install dependencies

To use DeepTune properly, you need to install the package dependencies:

```
$ pip install -r requirements.txt
```

### 1.3 Preface
**Now as you installed the required packages needed for DeepTune, and just before running your First DeepTune program, you may read this preface to get more engaged on what exactly to expect from the program.**


DeepTune is a cutting-edge software maintained to make usage of state-of-the-art algorithms for images and text datasets one step easier!

DeepTune currently supports finetuning 6 different state-of-the-art image classification models to fine-tune your dataset with: ResNet, DenseNet, Swin, EfficientNet, VGGNet, and ViT for images, with BERT, and GPT-2 for images. More details of the supported variants is found in the documentation's Supported Models table.

DeepTune gives you also a wide flexible set of options to choose what you think would suit your case the best. The options are as follows:

- Transfer Learning Mode: DeepTune currently supports applying transfer learning with Parameter Efficient Fine-tuning (PeFT) or without PeFT.

- Adjustable Additional Layer Choices: Fine-tuning is commonly applied in Deep Learning by adding one or more layer(s) on the top of the fine-tuned model. DeepTune gives you the choice of adding one, or two layers on the top of the model. Moreover, for the last layer size (also referred to as Embedding Layer) this is specified by the user choice as a CLI argument.

- Task Type: DeepTune provides initial support for converting classification-based models to work for regression.

- Embeddings Extraction: DeepTune provides a wide support for extracting embeddings for your dataset for all of the models mentioned above. This application is extremely useful if you want to get a meaningful representation of your own dataset to utilize further (e.g, projecting in 2D and see how they correlate, provide them to classical ML approach, etc.)

**Notes:**

> 1. DeepTune also provides support for DeepAR model for time series datasets, but it is only available now for training and testing.

> 2. DeepTune also supports RoBERTa model for text datasets. However, it is only available for embeddings extraction.

> 3. Pre-released version of DeepTune currently doesn't support PEFT for GPT-2.

> 4. Kindly note that DeepTune for images and texts only accepts Parquet files as an input (Time Series datasets are given as CSVs). The parquet file expected is actually containing two columns, If we work with images, then the two columns are [`images`, `labels`] pair. **Images must be in Bytes Format for efficient representation, and labels must be numerically encoded** If we work with text, then the two columns are [`text`, `label`] pair. For text, **Label column must be numerically encoded also.**

## Getting Started: Your First DeepTune Run

### 2.1 Using DeepTune for Training

#### Images & Texts

The following is the generic CLI structure of running DeepTune on images/text dataset stored in Parquet file as bytes format for training:

``` python -m trainers.<vision/nlp>.train_<model> \
  --input_dir <path_to_dataset> \
  --<model>_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --num_epochs <int> \
  --learning_rate <float> \
  --train_size <float> \
  --val_size <float> \
  --test_size <float> \
  --added_layers <int> \
  --embed_size <int> \
  [--fixed-seed] \
  --mode <cls_or_reg> \
  [--use-peft] \
  [--freeze-backbone]
```

`` --input_dir <str>`` : Path to your input dataset (must be parquet file).

``--<model>_version <model_variant>`` : The model to use with its respective version. (Only for Images Datasets, you don't use this flag with text datasets)

``-- batch_size <int>`` : Number of samples per batch.

``-- num_classes <int>`` : Number of your dataset's classes.

``-- num_epochs <int>``: Number of Training epochs.

``-- learning_rate <float>``: Learning rate.

``--train_size <float>``: Percentage of the training dataset w.r.t the whole data.

``--val_size <float>``: Percentage of the validation dataset w.r.t the whole data.

``--test_size <float>``: Percentage of the testing dataset w.r.t the whole data.

``--added_layers <int>``: Number of added layers on the top of the model for transfer learning either with using PeFT or not. Only 1 and 2 are supported for now.

``--fixed-seed``: (Flag) Ensures that a fixed random seed is set for reproducibility.

``--mode <str>``: Task mode: either `cls` for classification, or `reg` for regression.

`--use-peft`: (Flag) Enables Parameter Efficient Fine-tuning (PeFT)

`--freeze-backbone`: (Flag) Determines whether you want only to train the added new layers, or update the whole model parameters during training.

**Note** :
> The CLI command structure to training images and texts datasets models in DeepTune are the same in training, except for ``--<model>_version`` switch where you don't add it with text models. Moreover, the `--num_classes` isn't required as a switch when running GPT-2 model.

For example, suppose that we want to train our model with ResNet18, and apply transfer learning to update the whole model's weights, with 2 added layers, and an embedding layer of size 1000. Hence, we run the command as follows:

```
!python -m trainers.vision.train_resnet --num_classes 8 --resnet_version resnet18 --num_epochs 10 --added_layers 2 --embed_size 1000 --batch_size 16 --learning_rate 0.0001 --train_size 0.8 --val_size 0.1 --test_size 0.1 --input_dir "path/dataset.parquet" --fixed-seed --mode cls
```

If everything is set correctly, you should expect an output in the same format:
```
> os.environ['PYTHONHASHSEED'] set to 42.
> np.random.seed(42) set.
> torch.manual_seed(42) set.
> torch.cuda.manual_seed(42) set.
> torch.cuda.manual_seed_all(42) set.
> torch.backends.cudnn.benchmark set to False.
> torch.backends.cudnn.deterministic set to True.
> Dataset is loaded!
> Data splits have been saved and overwritten if they existed.
> The Trainer class is loaded successfully.

> 4%|████    | 459/855 [00:17<01:07,  5.89it/s, loss=0.43]

```

**Notes**:
> The way you initiate the calls (e.g, python trainers.vision.train_resnet.py, or python -m trainers.vision.train_resnet) may differ within different Operating Systems, what I am using in the format above for Windows 11 CMD. Do not forget to change the path for ``--input_dir`` as your dataset input path also.
> For using PeFT just add the `--use-peft` switch to the previous command.



After training is done, you will find that the output directory folder `deeptune_results` was initiated in your DeepTune path. Inside this folder, you will find the following outputs:

```plaintext
deeptune_results
├── train_split_<yyyymmdd>_<hhmm>.parquet
├── test_split_<yyyymmdd>_<hhmm>.parquet
├── val_split_<yyyymmdd>_<hhmm>.parquet
├── output_directory_trainval_<yyyymmdd>_<hhmm>
    └── cli_arguments.txt
    └── model_weights.pth
    └── training_log.csv
```

Description of Output files:
  - ``train/test/val_split.parquet files``: Store the training, testing, and validation dataset splits with timestamps corresponding to the time the run were initiated.
  -  ``output_directory_trainval_<yyyymmdd>_<hhmm>`` folder:
      - `cli_arguments.txt`: Indicating the CLI arguments you entered to run DeepTune, with the DeepTune version you are running.
      - `model_weights.pth`: The finetuned model weights (which will be used further for testing)
      - `training_log.csv`: A Performance Logger that reports the training and validation accuracies and errors during each epoch of training.
        
**Note**:
> The first three parquet files in ``deeptune_results`` are the splits that were produced from the dataset you have fed to DeepTune to work on, notice that everytime you run DeepTune with ``fixed-seed`` flag it should produce the same exact splits.


### 2.2 Using DeepTune for Evaluation

Evaluating your model on a separete holdout dataset (which we refer to as testing) here is referred to as evaluation. The reason is to simply not confuse the terms as testing in DeepTune documentation context also refers to testing the model functionality (e.g, writing test cases).

### Images

After using DeepTune to apply transfer learning on one of the models the package support, now we need to evaluate the performance of the tuned model for images.

The following is the generic CLI structure of running DeepTune for evalaution of image datasets:

``` python -m evaluators.vision.evaluate_<model> \
  --input_dir <path_to_dataset> \
  --<model>_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --num_epochs <int> \
  --test_set_input_dir <str> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  [--use-peft] \
  [--freeze-backbone]
```


`` --test_set_input_dir <str>`` : Path to your test dataset. It should be the `test_split_<yyyymmdd>_<hhmm>.parquet` you got from the previous DeepTune for training run.

`` --model_weights <str>`` : Path to your model's weights. It should be `model_weights.pth` you got from the previous DeepTune for training run.

**Note**:
> If you used one of the switches `--freeze_backone` or `--use_peft` or both in the previous run, you should use them while doing your evaluation here again.
> You feed the evaluator here the same `--added_layers` and `--embed_size` you used for your previous training run of DeepTune. Otherwise, a mismatch error will occur.

If everything is set correctly, and evaluation is done, you should expect an output in the same format:


```
Model into the path is loaded.
100%|█████████████████████████████████████████████████████| 107/107 [00:10<00:00, 10.03it/s]
98.18713450292398 0.05557631243869067
INFO | Test accuracy: 98.18713450292398%
{'loss': 0.05557631243869067, 'accuracy': 0.9818713450292398, '0': {'precision': 1.0, 'recall': 0.967479674796748, 'f1-score': 0.9834710743801653, 'support': 123.0}, '1': {'precision': 0.9931740614334471, 'recall': 1.0, 'f1-score': 0.9965753424657534, 'support': 291.0}, '2': {'precision': 1.0, 'recall': 0.967741935483871, 'f1-score': 0.9836065573770492, 'support': 155.0}, '3': {'precision': 0.9397163120567376, 'recall': 0.9706959706959707, 'f1-score': 0.954954954954955, 'support': 273.0}, '4': {'precision': 0.9847328244274809, 'recall': 1.0, 'f1-score': 0.9923076923076923, 'support': 129.0}, '5': {'precision': 0.9934640522875817, 'recall': 0.9440993788819876, 'f1-score': 0.9681528662420382, 'support': 161.0}, '6': {'precision': 0.9730538922155688, 'recall': 0.9878419452887538, 'f1-score': 0.9803921568627451, 'support': 329.0}, '7': {'precision': 1.0, 'recall': 0.9959839357429718, 'f1-score': 0.9979879275653923, 'support': 249.0}, 'macro avg': {'precision': 0.985517642802602, 'recall': 0.979230355111288, 'f1-score': 0.9821810715194739, 'support': 1710.0}, 'weighted avg': {'precision': 0.9822626797526258, 'recall': 0.9818713450292398, 'f1-score': 0.981906668565337, 'support': 1710.0}, 'auroc': 0.9997672516861436}
Test results saved successfully!

```

After evaluation is done, you will find that the output directory folder `deeptune_results` was initiated in your DeepTune path. Inside this folder, you will find the following output directory:

```plaintext
deeptune_results
├── output_directory_test_<yyyymmdd>_<hhmm>
    └── cli_arguments.txt
    └── test_accuracy.txt
```

Description of Output files:
  -  ``output_directory_test_<yyyymmdd>_<hhmm>`` folder:
      - `cli_arguments.txt`: Indicating the CLI arguments you entered to run DeepTune, with the DeepTune version you are running.
      - `test_accuracy.txt`: The test accuracy you achieved while using the model.
   
### Text

The process of applying DeepTune for evaluation for Text models it supports (GPT-2 and BERT) is the same in everything except for the CLI argument structure, where there is some changes it is important to highlight in a separate subsection.

In DeepTune, the text SoTA models save the weights of both the models, and the tokenizers. The tokenizer role is to split sentences into smaller units (we call them tokens) that can be more easily assigned meaning. On the other hand, the model is responsible for handling the part of interpreting these tokens.

The output directory from applying transfer learning on BERT or GPT-2 using DeepTune is as follows:

```plaintext
deeptune_results
├── train_split_<yyyymmdd>_<hhmm>.parquet
├── test_split_<yyyymmdd>_<hhmm>.parquet
├── val_split_<yyyymmdd>_<hhmm>.parquet
└── output_directory_trainval_<yyyymmdd>_<hhmm>
    ├── tokenizer
      └── ..
    ├── model
      └── ..
    ├── model_weights.pth
    └── training_log.csv
```

Notice that there are directories saving the update tokenizer, and model files. We need to feed these directories while evaluating test dataset for the fine-tuned model.

Hence, the generic CLI structure of running DeepTune for evalaution of text datasets:

``` python -m evaluators.nlp.evaluate_<model> \
  --input_dir <path_to_dataset> \
  --<model>_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --input_dir <str> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  --adjusted_<bert/gpt2>_dir <str> \
  [--use-peft] \
  [--freeze-backbone]
```

For the ``--adjusted_<bert/gpt2>_dir`` switch, we feed the whole output directory we got from running DeepTune for training (`output_directory_trainval_<yyyymmdd>_<hhmm>`)

**Note:**
> For GPT-2 model, **the switches `--added_layers` and `embed_size` are set by default as we tweaked the model architecture in order to be properly ready for training, so you don't have to set these to a specific input.** More details to follow in further phase of writing the documentation.

### 2.3 Using DeepTune for Embeddings Extraction


After we trained and evaluated our model, now we need to apply embeddings extraction used our fine-tuned model. Whatever you choose to apply embeddings extraction on is the user's/researcher's choice. However, as we integrate our framework with [df-analyze](https://github.com/stfxecutables/df-analyze) as we show in the next subsection, we choose to apply that for the test set we evaluated our model on.

### Images

After using DeepTune to apply transfer learning on one of the models the package support and testing its performance, the user may want to extract further information from their dataset using DeepTune for their future applications.

The following is the generic CLI structure of running DeepTune for embeddings extraction of image datasets:

``` python -m embed.vision.<model>_embeddings \
  --<model>_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --input_dir <str> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  --use_case <finetuned_or_pretrained_or_peft> \
  [--use-peft] \
  [--freeze-backbone]
```

The `--use_case` switch specifies on which use case you want to use DeepTune for:
  - pretrained: Using the exact weights of the model as it is without any further training. **This option allows you to use DeepTune with skiping the training and evaluation parts (You don't need to specify `--added_layers`, `--embed_size`, and `--model_weights`).**
  - finetuned: If you ran DeepTune for Transfer Learning without PeFT.
  - peft: If you ran DeepTune for Transfer Learning with PeFT.

**Note**:
> You feed the evaluator here the same `--added_layers` and `--embed_size` you used for your previous training run of DeepTune. Otherwise, a mismatch error will occur.

If everything is set correctly, and evaluation is done, you should expect an output in the same format:


```
100%|█████████████████████████████████████████████████████| 107/107 [00:09<00:00, 10.76it/s]
The shape of the embeddings matrix in the dataset is (1710, 1000)
```

**Note**:
> The shape of embeddings in the output is actually `(n_samples, embed_size)`. `n_samples` here is the number of elements in the test set, while `embed_size` is the number we set in the CLI which in this case is 1000.

You will find the output in a format of Parquet file in the following path: `deeptune_results/test_set_<use_case>_<model>_embeddings_<mode>.parquet`.

### Text
The process of Embeddings Extraction with text datasets would actually be the same except a small change in the CLI structure due to the reasons we illustrated in Section 2.2.

The following is the generic CLI structure of running DeepTune for embeddings extraction of text datasets:

``` python -m embed.nlp.<model>_embeddings \
  --batch_size <int> \
  --num_classes <int> \
  --input_dir <str> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  --use_case <finetuned_or_pretrained_or_peft> \
  --adjusted_<bert/gpt2>_dir <str> \
  --use_case <finetuned_or_peft>
  [--use-peft] \
  [--freeze-backbone]
```

**Notes:**
> We recall the same note mentioned in Section 2.2 that **the switches `--added_layers` and `embed_size` for GPT-2 model are set by default as we tweaked the model architecture in order to be properly ready for training, so you don't have to set these to a specific input. Also, For the ``--adjusted_<bert/gpt2>_dir`` switch, we feed the whole output directory we got from running DeepTune for training (`output_directory_trainval_<yyyymmdd>_<hhmm>`)**.
> Using text models for embeddings extraction directly with skipping Sections 2.1, and 2.2 (as we may do in images) isn't supported in DeepTune. If you want to extract the embeddings directly, you may use RoBERTa model. Moreover, this version of DeepTune doesn't yet support using peft case for GPT-2. The later part will be added in a future version of DeepTune.


You will find the output in a format of Parquet file in the following path: `deeptune_results/test_set_<use_case>_<model>_embeddings_<mode>.parquet`.






## Acknowledgments
This software package was developed as part of work done at Medical Imaging Bioinformatics lab under the supervision of Jacob Levman at St. Francis Xavier Univeristy, Nova Scotia, Canada.


Thanks to John's Work: [PEFT4Vision](https://github.com/johnkxl/peft4vision) for providing their code.


## Citation

If you find this repository helpful, please cite it as follows:


```bibtex
@software{DeepTune,
  author = {Moayadeldin Hussain},
  title = {DeepTune: Cutting-edge library to automate Computer Vision and Natural Language Processing algorithms.},
  year = {2025},
  url = {https://github.com/moayadeldin/deeptune-scratch},
  version = {1.0.0}
}
