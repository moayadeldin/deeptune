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

## Models DeepTune Supports (Up to Date)


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
      <td>'resnet18', 'resnet34', 'resnet50', 'resnet101', or 'resnet152'</td>
    </tr>
    <tr>
      <td>DenseNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'densenet121', 'densenet161', 'densenet169', or 'densenet201'</td>
    </tr>
    <tr>
      <td>Swin</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'swin_t', 'swin_s', or 'swin_b'</td>
    </tr>
    <tr>
      <td>EfficientNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', or 'efficientnet_b7'</td>
    </tr>
    <tr>
      <td>VGGNet</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'vgg11', 'vgg13', 'vgg16', or 'vgg19'</td>
    </tr>
    <tr>
      <td>ViT</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32' or 'vit_h_14'</td>
    </tr>
      <tr>
      <td>SiGLip</td>
      <td>✅</td>
      <td>✅</td>
      <td>✅</td>
      <td>Classification & Regression</td>
      <td>Image</td>
      <td>siglip</td>
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
  <td>GANDALF</td>
  <td colspan="2" style="text-align:center; font-weight:bold; font-size:16px;">Only Supports Regular Training</td>
  <td>✅</td>
  <td>Classification & Regression</td>
  <td>Tabular</td>
  <td>GANDALF</td>
</tr>
  </tbody>
</table>

## 📘 User Guide

### 1 Installation & Preface

- [1.1 Prerequisites](#11-prerequisites)
- [1.2 Linux and Windows](#12-linux-and-windows)
- [1.3 Preface](#13-preface)

### 2. Getting Started: Your First DeepTune Run
- [2.0 Splitting Your Dataset](#20-splitting-your-dataset)
- [2.1 Using DeepTune for Training](#21-using-deeptune-for-training)
- [2.2 Using DeepTune for Evaluation](#22-using-deeptune-for-evaluation)
- [2.3 Using DeepTune for Embeddings Extraction](#23-using-deeptune-for-embeddings-extraction)
- [2.4 [EXTRA] Integration with df-analyze](#24-extra-integration-with-df-analyze)




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

> 2. Pre-released version of DeepTune currently doesn't support PEFT for GPT-2.

> 3. Kindly note that DeepTune for images and texts only accepts Parquet files as an input (Time Series datasets are given as CSVs). The parquet file expected is actually containing two columns, If we work with images, then the two columns are [`images`, `labels`] pair. **Images must be in Bytes Format for efficient representation, and labels must be numerically encoded** If we work with text, then the two columns are [`text`, `label`] pair. For text, **Label column must be numerically encoded also.**

## Getting Started: Your First DeepTune Run

### 2.0 Splitting Your Dataset

We assume that your dataset formatted as Parquet File will need to be splitted into train/val/test splits as you are going to conduct different experiments with different models using DeepTune. 

The following is the generic CLI structure to split the dataset:
```
python -m split_dataset \
  --df <path_to_df> \
  --train_size <float> \
  --val_size <float> \
  --test_size <float> \
  --out <path> \
  --[fixed-seed] 
```

`` --df <str>`` : Path to dataset to split (must be parquet file).

``--train_size <float>``: Percentage of the training dataset w.r.t the whole data.

``--val_size <float>``: Percentage of the validation dataset w.r.t the whole data.

``--test_size <float>``: Percentage of the testing dataset w.r.t the whole data.

``--out <output_path``: Path to the directory where you want to save the results.

``--fixed-seed``: (Flag) Ensures that a fixed random seed is set for reproducibility.

**Note** :
> It is important to use the `--fixed-seed` flag to regenerate the same train/val/test splits everytime you run the above command.

The output will be stored in the directory specified with the `--out` argument, using the following naming format: ``data_splits_<yyyymmdd_hhmm>`` with the splits inside as follows, which we will use further for training and evaluation:

```
output_directory
├── data_splits_<yyyymmdd_hhmm>
    └── cli_arguments.json
    └── train_split.parquet
    └── test_split.parquet
    └── val_split.parquet
```

### 2.1 Using DeepTune for Training

#### Images

The following is the generic CLI structure of running DeepTune on images dataset stored in Parquet file as bytes format for training:

```
python -m trainers.vision.train \
  --train_df <path_to_train_df> \
  --val_df <path_to_val_df> \
  --model_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --num_epochs <int> \
  --learning_rate <float> \
  --added_layers <int> \
  --embed_size <int> \
  --out <output_path>
  [--fixed-seed] \
  --mode <cls_or_reg> \
  [--use-peft] \
  [--freeze-backbone]
```

`` --train_df <str>`` : Path to your train set (must be parquet file).

`` --val_df <str>`` : Path to your validation dataset (must be parquet file).

``--<model>_version <model_variant>`` : The model to use with its respective version architecture. You may use the model versions as in [Table](#models-deeptune-supports-up-to-date) above.

``-- batch_size <int>`` : Number of samples per batch.

``-- num_classes <int>`` : Number of your dataset's classes.

``-- num_epochs <int>``: Number of Training epochs.

``-- learning_rate <float>``: Learning rate.

``--added_layers <int>``: Number of added layers on the top of the model for transfer learning either with using PeFT or not. Only 1 and 2 are supported for now.

``--embed_size <int>``: Size of the intermediate embedding layer in case you choose to put 2 added layers on top of the tuned model.

``--out <output_path``: Path to the directory where you want to save the results.

``--fixed-seed``: (Flag) Ensures that a fixed random seed is set for reproducibility.

``--mode <str>``: Task mode: either `cls` for classification, or `reg` for regression.

`--use-peft`: (Flag) Enables Parameter Efficient Fine-tuning (PeFT)

`--freeze-backbone`: (Flag) Determines whether you want only to train the added new layers, or update the whole model parameters during training.

For example, suppose that we want to train our model with ResNet18, and apply transfer learning to update the whole model's weights, and an embedding layer of size 1000. Hence, we run the command as follows:

```
!python -m trainers.vision.train --train_df <path_to_train_df> --val_df <path_to_val_df> --model_version resnet18 --batch_size 4 --num_classes 2 --num_epochs 10 --learning_rate 0.0001 --added_layers 2 --embed_size 1000 --out <path_to_out_directory> --mode cls --fixed-seed
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

#### Text

Since DeepTune currently supports only two models for text classification, the way they are called in the CLI differs from that of image models. Apart from this, the CLI structure remains largely the same:

```
!python -m trainers.nlp.[train_multilinbert/train_gpt2] --train_df <path_to_train_df> --val_df <path_to_val_df> --model_version resnet18 --batch_size 4 --num_classes 2 --num_epochs 10 --learning_rate 0.0001 --added_layers 2 --embed_size 1000 --mode cls --fixed-seed
```

**Note**: 
> GPT2 model does not support PeFT right now in DeepTune.

After training completes, you may find the results in the directory specified with the `--out` directory. Alternatively, DeepTune will create an output directory named  `deeptune_results` (if it does not already exist). Inside this directory, the results are organized in a subfolder using the following naming convention: `trainval_output_<FINETUNED/PEFT>_<model_version>_<mode>_<yyyymmdd_hhmm>` with the following output:

```
output_directory
├── trainval_output_<FINETUNED/PEFT>_<model_version>_<mode>_<yyyymmdd_hhmm>
    └── cli_arguments.json
    └── model_weights.pth
    └── training_log.csv
```
**Description of Output files:**

- `cli_arguments.json`: Records the CLI arguments you entered to run DeepTune, along with the DeepTune version.
- `model_weights.pth`: The fine-tuned model weights (used later for testing).
- `training_log.csv`: A performance log reporting training and validation accuracies and errors for each epoch.

**Note**: 
> The text directory in the output will be named as follows: `trainval_output_<BERT/GPT2>_<yyyymmdd_hhmm>`


### 2.2 Using DeepTune for Evaluation

Evaluating your model on a separete holdout dataset (which we refer to as testing) here is referred to as evaluation. The reason is to simply not confuse the terms as testing in DeepTune documentation context also refers to testing the model functionality (e.g, writing test cases).

After using DeepTune to apply transfer learning on one of the models the package support, now we need to evaluate the performance of the tuned model for images.

The following is the generic CLI structure of running DeepTune for evalaution of image datasets:

```
python -m evaluators.vision.evaluate \
  --eval_df <path_to_dataset> \
  --<model>_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --num_epochs <int> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  --out <output_path> \
  [--use-peft] \
  [--freeze-backbone]
```

`` --eval_df <str>`` : Path to your test dataset. It should be the `test_split_<yyyymmdd>_<hhmm>.parquet` you got from the previous DeepTune for splitting data run.

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

After evaluation is done, you may find the results in the directory specified with the `--out` directory or `deeptune_results` was initiated in your DeepTune path. Inside this folder, you will find the following output directory:

```
deeptune_results
├── eval_output_FINETUNED/PEFT-<model_version>_<yyyymmdd>_<hhmm>
    └── cli_arguments.json
    └── full_metrics.json
```

Description of Output files:
  -  ``output_directory_test_<yyyymmdd>_<hhmm>`` folder:
      - `cli_arguments.json`: Indicating the CLI arguments you entered to run DeepTune, with the DeepTune version you are running.
      - `full_metrics.json`: The full metrics as appeared to you in the CLI while using the model.
   
### Text

In DeepTune, the text SoTA models save the weights of both the models, and the tokenizers. The tokenizer role is to split sentences into smaller units (we call them tokens) that can be more easily assigned meaning. On the other hand, the model is responsible for handling the part of interpreting these tokens.

The output directory from applying transfer learning on BERT or GPT-2 using DeepTune is as follows:

```
deeptune_results
└── trainval_output_<BERT/GPT2>_<yyyymmdd_hhmm>
    ├── tokenizer
      └── ..
    ├── model
      └── ..
    ├── model_weights.pth
    └── training_log.csv
```

Notice that there are directories saving the update tokenizer, and model files. We need to feed these directories while evaluating test dataset for the fine-tuned model.

Hence, the generic CLI structure of running DeepTune for evalaution of text datasets:

```
python -m evaluators.nlp.evaluate_<multilingualbert/gpt> \
  --eval_df <path_to_dataset> \
  --<model>_version <model_variant> \
  --batch_size <int> \
  --num_classes <int> \
  --num_epochs <int> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --mode <cls_or_reg> \
  --out <output_path> \
  [--use-peft] \
  [--freeze-backbone]
```

For the ``--model_weights`` switch, we feed the whole output directory we got from running DeepTune for training (`trainval_output_<BERT/GPT2>_<yyyymmdd_hhmm>`)

**Note:**
> For GPT-2 model, **the switches `--added_layers` and `embed_size` are set by default as we tweaked the model architecture in order to be properly ready for training, so you don't have to set these to a specific input.** More details to follow in further phase of writing the documentation.

### 2.3 Using DeepTune for Embeddings Extraction


After we trained and evaluated our model, now we need to apply embeddings extraction used our fine-tuned model. Whatever you choose to apply embeddings extraction on is the user's/researcher's choice. However, as we integrate our framework with [df-analyze](https://github.com/stfxecutables/df-analyze) as we show in the next subsection, we choose to apply that for the test set we evaluated our model on.

### Images

After using DeepTune to apply transfer learning on one of the models the package support and testing its performance, the user may want to extract further information from their dataset using DeepTune for their future applications.

The following is the generic CLI structure of running DeepTune for embeddings extraction of image datasets:


python -m embed.vision.embed --df "H:\john-pull-request\data_splits_20250803_1252\test_split.parquet" --mode cls --num_classes 2 --out "H:\john-pull-request" --model_version siglip --model_weights "H:\deeptune-beta\deeptune_results\train_output_PEFT-siglip_20250725_2223\custom_siglip_model.pt" --use_case peft --added_layers 2 --embed_size 1000 --batch_size 4


``` python -m embed.vision.embed \
  --df <path_to_df> \
  --batch_size <int> \
  --num_classes <int> \
  --out <path> \
  --model_version <str> \
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


python -m embed.nlp.gpt2_embeddings --df "H:\Moayad\deeptune-scratch\deeptune_results\DATASETS_PARQUET\data_splits_20250803_2007\test_split.parquet" --mode cls --out "H:\john-pull-request" --batch_size 8 --model_weights "H:\john-pull-request\trainval_output_GPT2_20250803_2319"

``` python -m embed.nlp.<gpt2/multilingualbert>_embeddings \
  --batch_size <int> \
  --num_classes <int> \
  --df <path_to_df> \
  --model_weights <str> \
  --added_layers <int> \
  --embed_size <int> \
  --use_case <finetuned_or_pretrained_or_peft> \
  [--use-peft] \
  [--freeze-backbone]
```

**Notes:**
> We recall the same note mentioned in Section 2.2 that **the switches `--added_layers` and `embed_size` for GPT-2 model are set by default as we tweaked the model architecture in order to be properly ready for training, so you don't have to set these to a specific input. Also, For the ``--adjusted_<bert/gpt2>_dir`` switch, we feed the whole output directory we got from running DeepTune for training (`output_directory_trainval_<yyyymmdd>_<hhmm>`)**.

> Using text models for embeddings extraction directly with skipping Sections 2.1, and 2.2 (as we may do in images) isn't supported in DeepTune. If you want to extract the embeddings directly, you may use RoBERTa model. Moreover, this version of DeepTune doesn't yet support using peft case for GPT-2. The later part will be added in a future version of DeepTune.


You will find the output directory in the following format format in the specified output path: `embed_output_<PRETRAINED/FINETUNED/PEFT>_<model_version>_<mode>_<yyyymmdd_hhmm>`.

```
deeptune_results
├── embed_output_<PRETRAINED/FINETUNED/PEFT>_<model_version>_<mode>_<yyyymmdd_hhmm>
    └── cli_arguments.json
    └── full_metrics.json
```

### 2.4 [EXTRA] Integration with df-analyze

[df-analyze](https://github.com/stfxecutables/df-analyze) is a command-line tool developed in the same Medical Imaging Bioinformatics lab at St. Francis Xavier University for automating Machine Learning tasks on small to medium-sized tabular datasets (less than about 200 000 samples, and less than about 50 to 100 features) using Classical ML algorithms.

If you want to further see how would your images/text dataset perform using Classical ML algorithms, it would be very difficult to achieve without having an intermediate representation of each sample. DeepTune provides you the way to get this intermediate representation using Embeddings Extraction as illustrated in Section 2.3, which allows you right now to run [df-analyze](https://github.com/stfxecutables/df-analyze)!

After you successfully allocate your embeddings file, either after running DeepTune on image dataset or text one, you may install df-analyze —instructions on how to do that is found on the software repository link— and run the following command:

```
python df-analyze.py --df "path\test_set_<use_case>_<model>_embeddings_cls.parquet" --outdir = ./deeptune_results --mode=classify --target label --classifiers lgbm rf sgd knn lr mlp dummy --embed-select none linear lgbm
```

If you ran df-analyze for regression task on images, you may change the command to be as follows:
```
python df-analyze.py --df "path\test_set_<use_case>_<model>_embeddings_reg.parquet" --target label --mode=regress --regressors knn lgbm elastic lgbm sgd dummy mlp --feat-select wrap --outdir=./deeptune_results
```

**Notes:**
> You may change the output directory in `--outdir` to be any folder.
> Do not forget to change the `--df` switch value according to the path of the embeddings file.



## Acknowledgments
This software package was developed as part of work done at Medical Imaging Bioinformatics lab under the supervision of Jacob Levman at St. Francis Xavier Univeristy, Nova Scotia, Canada.


Thanks to John's Work: [PEFT4Vision](https://github.com/johnkxl/peft4vision) for providing their code.


## Citation

If you find this repository helpful, please cite it as follows:


```bibtex
@software{DeepTune,
  author = {Moayadeldin Hussain, John Kendall, Jacob Levman},
  title = {DeepTune: Cutting-edge library to automate Computer Vision and Natural Language Processing algorithms.},
  year = {2025},
  url = {https://github.com/moayadeldin/deeptune-beta},
  version = {1.0.0}
}
