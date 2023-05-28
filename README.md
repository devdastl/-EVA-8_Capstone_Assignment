# EVA-8_Capstone_Assignment
This is the capstone assignment of Phase-1 of EVA-8 course from TSAI

## Introduction:
This is the final capstone assignment of EVA-8 course. There are two part of this assignment:
1. Training a ControlNet architecture on a available dataset.
2. Implementing Inpainting by using available trained models.

## Part1: Training a ControlNet architecture on custom dataset
### Objective:
Objective is train a ControlNet model from scratch. For this Part I have referred to a ControlNet implementation as shown [here](https://github.com/lllyasviel/ControlNet).

More on ControlNet training implementation in this repo is documented in [Part1-training_ControlNet/README.md](https://github.com/devdastl/EVA-8_Capstone_Assignment/blob/main/Part1-training_ControlNet/README.md) folder of this repo. 

### Getting started:
Below are steps to get started with training the ControlNet ing this repo:
- First go through the readme [Part1-training_ControlNet/README.md](https://github.com/devdastl/EVA-8_Capstone_Assignment/blob/main/Part1-training_ControlNet/README.md)
- Prepare dataset: follow this notebook in colab [Part1-training_ControlNet/part1_1-Dataset_preperation/Part1_1_prepare_dataset.ipynb](https://github.com/devdastl/EVA-8_Capstone_Assignment/blob/main/Part1-training_ControlNet/part1_1-Dataset_preperation/Part1_1_prepare_dataset.ipynb)
- Training and inferencing: Now train on the generated dataset by following this notebook on colab [Part1-training_ControlNet/part1_2-Training_ControlNet/part1_2_training_inference.ipynb](https://github.com/devdastl/EVA-8_Capstone_Assignment/blob/main/Part1-training_ControlNet/part1_2-Training_ControlNet/part1_2_training_inference.ipynb).

NOTE: Training is memory exaustive and 12GB of free colab might not be enough to load the model. I have used a local system for complete training which can recreated on colab pro.
### Training output:
- Below are some of the important information of training this model:
    - Batch size = 11
    - epochs = 13
    - time taken = 24 hours
    - sliced attention = True
    - Dataset = StandFord Dog dataset with 20k images

Below is the Training log generated while training the model:
 ```
    Epoch 1:  16%|_| 300/1871 [17:33<1:31:56,  3.51s/it, loss=0.192, v_num=0, train/loss_simple_step=0.210, train/loss_vlb_step=0.0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:27<00:00,  1.79it/s]
Epoch 1:  32%|_| 600/1871 [34:52<1:13:53,  3.49s/it, loss=0.202, v_num=0, train/loss_simple_step=0.223, train/loss_vlb_step=0.0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.06it/s]
Epoch 1:  48%|_| 900/1871 [52:08<56:15,  3.48s/it, loss=0.183, v_num=0, train/loss_simple_step=0.163, train/loss_vlb_step=0.000Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.06it/s]
Epoch 1:  64%|_| 1200/1871 [1:09:30<38:52,  3.48s/it, loss=0.175, v_num=0, train/loss_simple_step=0.134, train/loss_vlb_step=0.Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.03it/s]
Epoch 1:  80%|_| 1500/1871 [1:26:50<21:28,  3.47s/it, loss=0.18, v_num=0, train/loss_simple_step=0.206, train/loss_vlb_step=0.0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:27<00:00,  1.84it/s]
Epoch 1:  96%|_| 1800/1871 [1:44:20<04:06,  3.48s/it, loss=0.174, v_num=0, train/loss_simple_step=0.135, train/loss_vlb_step=0.Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.03it/s]
Epoch 2:   0%| | 0/1871 [00:00<?, ?it/s, loss=0.178, v_num=0, train/loss_simple_step=0.147, train/loss_vlb_step=0.000525, trainData shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.07it/s]
Epoch 2:  16%|_| 300/1871 [17:29<1:31:37,  3.50s/it, loss=0.157, v_num=0, train/loss_simple_step=0.318, train/loss_vlb_step=0.0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps

...........
...........

Epoch 12:  64%|_| 1200/1871 [1:09:47<39:01,  3.49s/it, loss=0.163, v_num=0, train/loss_simple_step=0.164, train/loss_vlb_step=0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:25<00:00,  1.98it/s]
Epoch 12:  80%|_| 1500/1871 [1:27:16<21:35,  3.49s/it, loss=0.156, v_num=0, train/loss_simple_step=0.218, train/loss_vlb_step=0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.03it/s]
Epoch 12:  96%|_| 1800/1871 [1:44:27<04:07,  3.48s/it, loss=0.161, v_num=0, train/loss_simple_step=0.155, train/loss_vlb_step=0Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:23<00:00,  2.09it/s]
Epoch 13:   0%| | 0/1871 [00:00<?, ?it/s, loss=0.137, v_num=0, train/loss_simple_step=0.0911, train/loss_vlb_step=0.000447, traData shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:23<00:00,  2.12it/s]
Epoch 13:  16%|_| 300/1871 [17:14<1:30:15,  3.45s/it, loss=0.182, v_num=0, train/loss_simple_step=0.151, train/loss_vlb_step=0.Data shape for DDIM sampling is (4, 4, 32, 32), eta 0.0
Running DDIM Sampling with 50 timesteps
DDIM Sampler: 100%|____________________________________________________________________________| 50/50 [00:24<00:00,  2.07it/s]
Epoch 13:  20%|_| 367/1871 [21:22<1:27:35,  3.49s/it, loss=0.16, v_num=0, train/loss_simple_step=0.0696, train/loss_vlb_step=0.Time limit reached. Elapsed time is 1 day, 0:00:03. Signaling Trainer to stop.
Epoch 13:  20%|_| 368/1871 [21:49<1:29:08,  3.56s/it, loss=0.154, v_num=0, train/loss_simple_step=0.106, train/loss_vlb_step=0.
```
### Inference output:
Below is a gradio output from the trained model.

<img src="Part1-training_ControlNet/output_images/output_3.PNG" alt= “” width="500" height="">

## Part2: Implementation of Inpainting using Stable Diffusion

### Objective:
Objective of Part2 of this capstone assignment is to implement Inpainting from the scratch.
</br>
I have referred to a hugging face implementation of Inpainting and then built upon that. Here is the [link](https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting/tree/main)

### Getting started:
- First go through the separate README file for this part in folder [Part2-Inpainting_implementation/README.md](https://github.com/devdastl/EVA-8_Capstone_Assignment/blob/main/Part2-Inpainting_implementation/README.md)
- Then to directly implement working attempt of inpainting, you can refer to notebook in folder [Part2-Inpainting_implementation/Attempt3_Success.ipynb](https://github.com/devdastl/EVA-8_Capstone_Assignment/blob/main/Part2-Inpainting_implementation/Attempt3_Success.ipynb)

### Inference output:
Below is the inference output of the successful attempt of Inpainting implementation

<img src="Part2-Inpainting_implementation/test_data/inpainting_output1.png" alt= “” width="500" height="">
