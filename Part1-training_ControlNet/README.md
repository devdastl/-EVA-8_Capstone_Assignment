# Training ControlNet using Stable Diffusion
This readme file specifically focuses on Part1 (Training ControlNet) of the capstone assignment

## Introduction
This is a sub README file to capture the details on Training ControlNet. The complete process of training a ControlNet architecture is dividied into two parts:
- Part1: is dataset preperation, you need source image, target image and a prompt discribing the target image. Here souce image is detected edges of target image.
- Part2: is attaching Control part, training the new architecture and inference after training.

More details are coverd below.

## About ControlNet architecture
Before diving into training one, let quickly look into what is ControlNet architecture
</br>
<img src="output_images/sd.png" alt= “” width="500" height="">
ControlNet, an end-to-end neural network architecture that controls large image
diffusion models (like Stable Diffusion) to learn task-specific input conditions. The ControlNet clones
the weights of a large diffusion model into a "trainable copy" and a "locked copy": the locked copy
preserves the network capability learned from billions of images, while the trainable copy is trained
on task-specific datasets to learn the conditional control. The trainable and locked neural network
blocks are connected with an unique type of convolution layer called "zero convolution", where the
convolution weights progressively grow from zeros to optimized parameters in a learned manner.
Since the production-ready weights are preserved, the training is robust at datasets of different scale.

## Folder structure
Since the folder structure and files are little complex, lets quickly go through the some important files and structure of ControlNet training directory:
```
Part1-training_ControlNet/
  |
  ├── README.md                           <- readme file with documentation about training ControlNet
  |
  ├── output_images/                      <- folder containing inference output of trained model.
  |
  ├── part1_1-Dataset_preparation/        <- sub directory containing files to help in dataset generation for training ControlNet.
  │   ├── Part1_1-prepare_dataset.ipynb   <- Main notebook which should be executed in colab to generate dataset.
  │   ├── annotator/                      <- folder containing tools to generate Canny edge images given s source image.
  │   ├── Captioner/                      <- folder containing tools to generate prompt given source image.
  │   │   ├── blip_caption.py                <- script containing custom written class to perfrom inference using BLIP to generate prompts
  │   │   ├── models/                        <-folder containing model definiation for BLIP prompt generation. 
  │   ├── run_generation.py               <- Custom written script which given folder of images will generate canny images and prompts.
  │   ├── requirements.txt                <- text file containing python requirements.
  │
  ├── part1_2-Training_ControlNet/                   <- Sub directory containing files to help in architecture generation, training and inference
  │   ├── part1_2_training_inference.ipynb            <- Main notebook which should be executed after generating dataset to build, train and inference on ControlNet architecture.
  │   ├── training.py             <- Python file containing lightning code to train ControlNet architecture on a GPU.
  │   ├── inference.py            <- Python file containing pytorch code to run inference on trained ControlNet model on GPU.
  │   ├── tutorial_dataset.py     <- Python file containing pytorch code to generate Dataset
  │   ├── tool_add_control.py
  │   ├── requirements.txt
```

## Getting started

## part1_1: Dataset preperation


## part1_2: Training ControlNet archirecture and inferenceing

## Training output

## Inference output
Below are some inference output from the trained model:

<img src="output_images/output_1.PNG" alt= “” width="500" height="">
<img src="output_images/output_2.PNG" alt= “” width="500" height="">
<img src="output_images/output_3.PNG" alt= “” width="500" height="">

## Conclusion
- Model when trained for lower epochs did not produce good results.
- Using slice attention can greatly increase the capacity to have larger batch size.
- Training for longer time can generate even better results
