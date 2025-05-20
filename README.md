# DeepTune

**NOTE: This is a beta pre-release version of DeepTune. Access is restricted to individuals with whom the code has been shared, under the agreement of Dr. Jacob Levman. Redistribution to third parties is strictly prohibited without prior consent.**

**DeepTune** is a full compatible library to automate Computer Vision and Natural Language Processing algorithms on diverse images and text datasets.

As a cutting-edge software library that has been specifically designed for use in different Machine Learning Tasks, inclduing but not limited to image classification, transfer learning, and embedding extraction. 

**DeepTune** is currently going under the process of extensive testing, and offers multiple features including ability to apply transfer learning via fine-tuning for advanced classification algorithms for images, and texts, including Parameter Efficient Fine-Tuning with LoRA, and latent feature extraction as embedding vectors. This offers a massive assistance for users to take full advantage of what their case studies may offer with simple commands.

## Features

- Ability of fine-tuning SoTA Computer Vision algorithms for Image Classification
- Ability of fine-tuning SoTA NLP algorithms.
- Providing PEFT with LoRA for Computer Vision algorithms implemented, enabling state-of-the-art models that typically require substantial computational resources to perform efficiently on lower-powered devices. This approach not only reduces computational overhead but also enhances performance
- Ability of extracting meaningful feature embeddings representing your own dataset with SoTA algorithms for image and text classification tasks.
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
│   └── vision
│       ├── densenet_embeddings.py
│       ├── efficientnet_embeddings.py
│       ├── resnet_embeddings.py
│       └── swin_embeddings.py
├── evaluator
│   ├── nlp
│   │   └── evaluate_multilingualbart.py
│   └── vision
│       ├── evaluate_densenet.py
│       ├── evaluate_efficientnet.py
│       ├── evaluate_resnet.py
│       ├── evaluate_swin.py
│       └── evaluator.py
├── src
│   ├── nlp
│   │   ├── es_roberta.py
│   │   ├── multilingual_bart.py
│   │   └── multilingual_bart_peft.py
│   └── vision
│       ├── densenet.py
│       ├── densenet_peft.py
│       ├── efficientnet.py
│       ├── efficientnet_peft.py
│       ├── resnet.py
│       ├── resnet_peft.py
│       ├── swin.py
│       └── swin_peft.py
├── trainers
│   ├── nlp
│   │   └── train_multilingualbart.py
│   └── vision
│       ├── train_densenet.py
│       ├── train_efficientnet.py
│       ├── train_resnet.py
│       └── train_swin.py
├── .gitignore
├── README.md
├── options.py
├── requirements.txt
├── trainer.py
└── utilities.py
```

## Example

Preliminary: You need to install the needed dependencies:
```
pip install -r requirements.txt
```

1. Suppose you want to utilize DeepTune for fine-tune ResNet18 on your own image dataset for a Classification problem, you want to add 2 additional layers on top of the original architecture, with the ResNet internal features being mapped to an indermediate layer, and then this is to be mapped to the number of classes. You have a limited computation resources so you want only to make the intermediate layer of size 100, freeze the backbone weights, updating only the added layers, and use Parameter Efficient Fine-tuning (PEFT) with LoRA. You want your dataset splits and reruns of the training of your model to be reproducible so you need to have a fixed seed, where the weights are reinitialized each time the same, and dataset splits are also the same. You decided that your dataset is to splitted 70% train, 10% validation, and 20% of test set. DeepTune expects the dataset to be in a Parquet file format, with the images being represented as byte-format, in a column named images, and labels to be encoded to integers, with column named labels. The  The command then would be as follows:

```
python -m trainers.vision.train_resnet \
--input_dir "dataset.parquet" \
--swin_version resnet18 \
--batch_size <batch_size> \
--num_classes <num_classes> \
--num_epochs <num_epochs> \
--learning_rate <lr> \
--train_size 0.7 \
--val_size 0.1 \
--test_size 0.2 \ 
--added_layers 1 \
--embed_size 100 \
--fixed-seed \
--mode cls \ 
--use-peft
```
2. After you complete your run successfully, have your training and validation errors and accuracies respectively, and your results saved in a directory named `deeptune_results` with the run metadata being saved inside a directory called `output_directory_trainval_{yyyymmdd_hhmm}`. Inside the train-validation directory, you will find the `model_weights.pth` which you will use to test your model on the testing dataset, along with a CSV file showing the update of progress of training, and CLI arguments you entered.

You will notice after you complete your run that DeepTune automatically provided you the three train, validation, and test splits in a parquet file format, handling the headache of splitting them yourself. You will find the corresponding test dataset having the following name `test_split_{yyyymmdd_hhmm}.parquet`. You will take the path leading to this dataset (the relative path would be `deeptune_results/test_split_{yyyymmdd}_hhmm.parquet`) and feed it to the specific evaluation mode of DeepTune using ResNet18, following this command:

```
python -m evaluators.vision.evaluate_resnet \
--test_set_input_dir "deeptune_results/test_split_{yyyymmdd}_hhmm.parquet" \ // adjust according to your specific test split file
--batch_size <batch_size> \
--num_classes <num_classes> \
--use-peft \
--model_weights "deeptune_results\output_directory_trainval_{yyyymmdd_hhmm}\model_weights.pth" \ // adjust according to your specific output trainval file
--added_layers 1 \
--embed_size 100 \
--mode cls \
--resnet_version resnet18
```

3. Very Good! Now you have your test set results as you applied Transfer Learning, with PEFT option, using DeepTune on your Image Classification Problem. The output is saved in the following directory `deeptune_results/test_split_{yyyymmdd_hhmm}` with the test accuracy and CLI arguments you entered. If you want to obtain the embeddings format of your test set, to use for further analysis using Classical ML algorithm, or software package as `df-analyze` then you will use the specific embeddings extraction mode of DeepTune using ResNet18, following this command:
```
python -m embed.vision.resnet_embeddings \
--use_case peft \
--resnet_version resnet18 \
--input_dir "deeptune_results/test_split_{yyyymmdd}_hhmm.parquet" \ // adjust according to your specific test split file
--batch_size <batch_size> \
--num_classes <num_classes> \
--model_weights "deeptune_results\output_directory_trainval_{yyyymmdd_hhmm}\model_weights.pth" \ // adjust according to your specific output trainval file
--added_layers 1 \
--embed_size 100 \
--mode cls \
```


You will obtain the following embeddings in the following format `deeptune_results/test_set_peft_resnet_embeddings_cls.parquet`.

At last, after one run of DeepTune, your `deeptune_results` output directory should have the following structure. We assume for example you had these runs on 30 March 2025, and throughout the day starting at 3:30PM :

```
deeptune_results/
├── output_directory_trainval_20250330_1530/
│   ├── cli_arguments.txt
│   ├── training_log.csv
│   └── model_weights.pth
├── output_directory_test_20250330_1639/
│   ├── cli_arguments.txt
│   └── test_accuracy.txt
├── train_split_20250330_1530.parquet
├── val_split_20250330_1530.parquet
├── test_split_20250330_1530.parquet
└── test_set_{use_case}_{model}_embeddings_{mode}.parquet
```





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
