# DeepTune

**DeepTune** is a full compatible library to automate Computer Vision and Natural Language Processing algorithms on diverse images and text datasets.

As a cutting-edge software library that has been specifically designed for use in different Machine Learning Tasks, inclduing but not limited to image classification, transfer learning, and embedding extraction. 

**DeepTune** is currently going under the process of extensive testing, and offers multiple features including ability to apply transfer learning via fine-tuning for advanced classification algorithms for images, and texts, including Parameter Efficient Fine-Tuning with LoRA, and latent feature extraction as embedding vectors. This offers a massive assistance for users to take full advantage of what their case studies may offer with simple commands.

## Features

- Ability of fine-tuning SoTA Computer Vision algorithms for Image Classification
- Ability of fine-tuning SoTA NLP algorithms.
- Providing PEFT with LoRA for Computer Vision algorithms implemented, enabling state-of-the-art models that typically require substantial computational resources to perform efficiently on lower-powered devices. This approach not only reduces computational overhead but also enhances performance
- Ability of extracting meaningful feature embeddings representing your own dataset with SoTA algorithms for image and text classification tasks.

## Algorithms Implemented

| Model         | Transfer Learning with Adjustable Embedding Layer? | Support PEFT with Adjustable Embedding Layer? | Support Embeddings Extraction? | Task                | Modality |
|--------------|-------------------------------------------------|---------------------------------|--------------------------------|----------------------|----------|
| ResNet18     | ✅                                              | ✅                              | ✅                              | Classification       | Image    |
| Siglip       | (SUD)                                           | (SUD)                           | (SUD)                           | Classification       | Image    |
| DenseNet121  | ✅                                              | ✅                              | ✅                              | Classification       | Image    |
| Swin         | ✅                                              | ✅                              | ✅                              | Classification       | Image    |
| XLM-RoBERTa  | ❌                                              | ❌                              | ✅                              | Sentiment Analysis   | Text     |



## Acknowledgments
This software package was developed as part of work done at Medical Imaging Bioinformatics lab under the supervision of Jacob Levman at St. Francis Xavier Univeristy, Nova Scotia, Canada.


Thanks to [John Kendall](https://github.com/johnkxl) for their assistance.


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
