# XVisionHelper

### Easily adjust one of the most popular pre-trained Deep Learners for image classification using your dataset with a single command.

[X-Vision-Helper.webm](https://github.com/user-attachments/assets/36242179-15da-4303-9920-c3cf84c8bc19)


## Features
- **Finetuning the model on your dataset.**: The helper gives you the chance to fine-tune ResNet50 on different datasets straightforwardly, without the burden of writting every component from scratch (e.g. dataloader, training loop, model adjusted architecture, etc.).
- **Finetuning the "PEFT-ed" model on your dataset**: Parameter Efficient Fine-tuning (PEFT) with Low-rank Adaptation (LoRA) provides the ability to freeze the pre-trained model weights, and injects additional low-rank matricies parameters during forward pass, and backward pass.
- **Extract Embeddings from the fine-tuned model**: After fine-tuning ResNet50 on your dataset, you have the option to pull out the embeddings directly from the model. This can be pretty handy for things like dimensionality reduction or similarity searches. For CSCI-525.10, it’s encouraged to give this step a try—you can pass these embeddings to the [df-analyze](https://github.com/stfxecutables/df-analyze) command-line tool that’s used throughout the course. It’s an interesting way to see how other models, like SVMs or logistic regression, perform when you start with these embeddings instead of raw images.

## Notes

- This package requires access to GPU resources, whether it's on your local machine or a server.
- Currently, it only supports fine-tuning a ResNet50 model as the initial version.
- It’s designed to work with supervised datasets, exclusively image data.
- Given the nature of the underlying model, the package is best suited for image classification tasks, where it’s expected to deliver meaningful results.

## Installation

1. Create a virtual environment, and activate it, which is recommended to avoid dependency issues (Optional).

   ```bash
   python -m venv xVisionHelper
   source xVisionHelper/bin/activate
   ```
2. Clone the repository and navigate to its directory
   ```bash
   git clone https://github.com/moayadeldin/X-vision-helper.git
   cd X-vision-helper
   ```
3. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
## Example

Suppose you have an image classification dataset that we want to fine-tune ResNet50 on, at first, X-vision-helper expects your dataset to be organized in the following directory structure:
```plaintext
DatasetName
├── train
│   ├── class1
│   ├── class2
│   └── ...
├── val
│   ├── class1
│   ├── class2
│   └── ...
└── test
    ├── class1
    ├── class2
    └── ...
```

Now you may run the fine-tuning script as follows:
```bash
python3 model_finetuning.py \
    --model <model_name>
    --num_classes <num_classes> \
    --num_epochs <num_epochs> \
    --batch_size <batch_size> \
    --learning_rate <learning_rate> \
    --train_dir <train_dir> \
    --val_dir <val_dir> \
    --test_dir <test_dir>

```
| Hyperparameter   | Purpose                                                                      | Datatype                                |
|------------------|------------------------------------------------------------------------------|-----------------------------------------|
| `model`          | Decide whether to use fine-tuning ResNet18 or PEFT-ResNet18         | `str` (options: `resnet50`, `peft-resnet18`) |
| `num_classes`    | Set this number according to the number of classes in your own dataset       | `int`                                   |
| `num_epochs`     | Number of times you may want the model to be finetuned on the whole training set | `int`                              |
| `batch_size`     | Number of samples that you feed into your model at each iteration of training | `int`                                   |
| `learning_rate`  | Number that controls how fast you want your model to converge                | `float`                                 |
| `train_dir`      | Directory path to your training dataset                                      | `str`                                   |
| `val_dir`        | Directory path to your validation dataset                                    | `str`                                   |
| `test_dir`       | Directory path to your test dataset                                          | `str`                                   |


After you finish running the fine-tuning script, the package will produce `.pth` file that you may use for embeddings extraction:

```bash
python3 extract_embeddings.py \
    --num_classes <num_classes> \
    --batch_size <batch_size> \
    --dataset_dir <dataset_dir> \
    --finetuned_model_pth <finetuned_model_path>
```

At the end, you will obtain two `.csv` files, one for your instances embeddings, and the other for labels embeddings. It is expected to that the dimensions of your instances embeddings output will be `(N,1000)` and the dimensions of the labels embeddings output will be `(N,1)`, where N is the number of samples you extracted embeddings for.

##

This package has been developed to help Master's students taking the CSCI-525.10 Machine Learning Design course at St. Francis Xavier University with their advanced projects. It is not limited to this specific purpose and may be applied to other related use cases. As this project primary goal is to help students, the code implementation is expected to remain neat, short and clean as possible. This project is licensed under the MIT License.

## Citation

If you find this repository helpful in your research, please cite it as follows:

```bibtex
@software{X-vision-helper,
  author = {Moayadeldin Hussain, Jacob Levman},
  title = {X-vision-helper: Adjusting Deep Learners for image classification problems with a single command},
  year = {2024},
  url = {https://github.com/moayadeldin/X-vision-helper},
  version = {1.0.0}
}
