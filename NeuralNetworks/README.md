## PlantDiseaseDetection Using end to end learning


## METHOD

We analyze 1973 images of plant leaves, which have a spread of first 5 classes of 38 class distribution and after this we extend our training to all 38 classes. Each class label is a crop-disease pair, and we make an attempt to predict the crop-disease pair given just the image of the plant leaf. In the approach we resize the images to 224 × 224 pixels, and we perform both the model optimization and predictions on these downscaled images.

We evaluate the applicability of deep convolutional neural networks for the classification problem described above. We focus on the popular architecture, named VGG-16.

We analyze the performance of the model on the PlantVillage dataset by training the model by adapting already trained models (trained on the ImageNet dataset) using transfer learning. In case of transfer learning, we added a new fully connected layer to the network with 5/38 outputs. 

 ## To summarize, 
1. Choice of deep learning architecture:
VGG-16
2. Choice of training mechanism:
Transfer Learning
3. Choice of dataset type:
Colored
4. Choice of training-testing set distribution:
Train: 70%, Val: 30%


We used the following hyper-parameters in all of the experiments:
• Solver type: Adam,
• Base learning rate: 0.01,
• Momentum: beta1=0.9, beta2=0.999,

## RESULT

At the outset, we note that on a dataset with 5 class labels our model achieves an overall accuracy of 71.43% on training set and 80.65% on validation set after one epoch and on a dataset with 38 class labels, an overall accuracy of 50.82% on training set and 60.86% on validation set after running two epochs.





