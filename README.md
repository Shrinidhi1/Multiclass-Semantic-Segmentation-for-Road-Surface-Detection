# Multiclass Semantic Segmentation for Road Surface Detection

1. Identified road surfaces and 13 different classes like speed bumps, paved, unpaved, markings, water puddles, potholes, etc.
2. Trained the model for semantic segmentation on Unet architecture along with backbone architectures like Resnet, InceptionNet and VGGnet.
3. Added mask to images to show the classes according to their respective colors.

## Model Training
|Sl. No.| Model| Epochs| Mean IoU Score on CV|
|-|-|-|-|
|1.|Unet|20|0.26527|
|2.|Unet with Resnet34|100|0.7297|
|3.|Unet with InceptionNetV3|20|0.6633|
|4.|Unet with VGGnet16|20|0.6604|

#### Deployed On Hugging Face: [Link](https://huggingface.co/spaces/shrinidhi-rh/Road-Surface-Detection)
