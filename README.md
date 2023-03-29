#  Accuracy Neural Network AI
Simple Project that sends back the accuracy of the AI after training it.
This project holds 2 main files
* __CNN__
* __SNN__

## Convolutional

__CNN__ A convolutional neural network (CNN or ConvNet) is a network architecture for deep learning that learns directly from data. CNNs are particularly useful for finding patterns in images to recognize objects, classes, and categories

Here is the Formula for a Convolutional Neural Network

![Formula for CNN](https://user-images.githubusercontent.com/77110462/228610008-5dadf4ff-4924-4b5c-ab06-3e08a82b1423.png)



## Simple
__SNN__ Neural networks can help computers make intelligent decisions with limited human assistance. This is because they can learn and model the relationships between input and output data that are nonlinear and complex.

# Comparing Results

## Convolutional
* Number of epochs: __4__
* Time spent: __40 seconds__
* Training Data Accuracy: __98.57__
* Test Data Accuracy: __98.25__

## Simple
* Number of epochs: __4__
* Time spent: __37.5 seconds__
* Training Data Accuracy: __96.74__
* Test Data Accuracy: __96.13__

# PyTorch
PyTorch is a fully featured framework for building deep learning models,
which is a type of machine learning that's commonly used in applications like image recognition and language processing.
Written in Python, it's relatively easy for most machine learning developers to learn and use.


# Installation 
* Go to [PyTorch](https://pytorch.org/) and you should see a installation area, simply choose the settings that fit you best, and use the pip command that is given.
* Then clone this repository or download it.

# Usage
not complicated and requires no setup.
just run this command to run the file
```bash
py __SNN__.py
```
or if you want to use the Convolutional Neural Network
```bash
py __CNN__.py
```

# Expected Results

```bash
Checking accuracy on traning data
Got 59337 / 60000 with accuarcy 98.89
Checking accuracy on test data
Got 9724 / 10000 with accuarcy 97.24
```

>These numbers are not fixed and will never be the same.

# Modifications

The only thing i recommend for beginners to play around with is the
`num_epochs` variable, increase/decrease and it will affect the accuracy  




