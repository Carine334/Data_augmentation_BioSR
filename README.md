## Enhanced AI with hybrid dataset

# Project Overview

New scientific research depends on new data, especially in the field of microscopy, where taking pictures is difficult. The complexity of the process is shown by the construction of high-resolution microscopes. This project try to address these problems by creating synthetic data to improve cellular-level images. The methodology is to select eligible patches from the raw images, which are then augmented to create a comprehensive image dataset. The dataset is then processed through a Variational Autoencoder (VAE) to generate new image data.

## Repository Structure

```
/saved_images                            # Contains images of the model architecture, the model performance, and a visualization of created images 
data_gen_DN                              # Script for generating the training data for the low quality images vector of the initial data 
data_gen_SR                              # Script for generating the training data for the super-resolved images vector of the initial data
augmentation.py                          # Script with the implementation of the different translation techniques (Flipping, transposing, rotating...)
vae_training.py                          # Script with the architecture and VAE training
