
# Ensemble of Deep Convolutional Neural Networks for Learning to Detect Retinal Vessels in Fundus Images

Vision impairment due to pathological damage of the retina can largely be prevented through periodic screening using fundus color imaging. However the challenge with large scale screening is the inability to exhaustively detect fine blood vessels crucial to disease diagnosis. This work presents a computational imaging framework using deep and ensemble learning for reliable detection of blood vessels in fundus color images. An ensemble of deep convolutional neural networks is trained to segment vessel and non-vessel areas of a color fundus image. During inference, the responses of the individual ConvNets of the ensemble are averaged to form the final segmentation. In experimental evaluation with the DRIVE database, we achieve the objective of vessel detection with maximum average accuracy of 91.8% (This accuracy is different from the accuracy reported in the paper because of different libraries used)
<hr>

FUNDUS Image             |  Manual Segmentation           | Predicted Segmentation
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/01_test_src.png?raw=True" width="220"> |  <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/01_manual1.png?raw=True" width="220"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/01_test.png?raw=True" width="220">

<hr>
## Proposed Method

**Ensemble learning** is a technique of using multiple models or experts for solving a particular artificial intelligence problem. Ensemble methods seek to promote diversity among the models they combine and reduce the problem related to overfitting of the training data. The outputs of the individual models of the ensemble are combined (e.g. by averaging) to form the final prediction.

**Convolutional neural networks** (CNN or ConvNet) are a special category of artificial neural networks designed for processing data with a gridlike structure. The ConvNet architecture is based on sparse interactions and parameter sharing and is highly effective for efficient learning of spatial invariances in images. There are four kinds of layers in a typical ConvNet architecture: convolutional (conv), pooling (pool), fullyconnected (affine) and rectifying linear unit (ReLU). Each convolutional layer transforms one set of feature maps into another set of feature maps by convolution with a set of filters.

This paper makes an attempt to ameliorate the issue of subjectivity induced bias in feature representation by training an ensemble of 12 Convolutional Neural Networks (ConvNets) on raw color fundus images to discriminate vessel pixels from non-vessel ones. 

**Dataset**: The ensemble of ConvNets is evaluated by learning with the DRIVE training set (image id. 21-40) and
testing over the DRIVE test set (image id. 1-20). 

**Learning mechanism**: Each ConvNet is trained independently on a set of 120000 randomly chosen 3×31×31 patches.
Learning rate was kept constant across models at 5e − 4. Dropout probability and number of hidden units in
the penultimate affine layer of the different models were sampled respectively from U ([0:5; 0:9]) and U ({128; 256; 512}) where U(:) denotes uniform probability distribution over a given range. The models were trained using Adam algorithm with minibatch size 256. Some of these parameters are different from the paper. The user can set some of these parameters using command line arguments which is explained in later sections.

<img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Proposed-Method.png?raw=True" width="800">

<hr>

## Architecture

The ConvNets have the same organization of layers which can be described as: 

**input- [conv - relu]-[conv - relu - pool] x 2 - affine - relu - [affine with dropout] - softmax**

(Schematic representation below)

<img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Architecture.png?raw=True" width="800">

<hr>
## Some Results

FUNDUS Image             |  Magnified Section          | Ground Truth          | Prediction
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified1_1.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified1_2.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified1_3.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified1_4.png?raw=True" width="180"> 
 <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified2_1.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified2_2.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified2_3.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified2_4.png?raw=True" width="180"> 
 <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified3_1.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified3_2.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified3_3.png?raw=True" width="180"> | <img src="https://github.com/Ankush96/Deep-Vessel/blob/master/images/Magnified3_4.png?raw=True" width="180"> 
 
 Note that the 3rd image is not easily visible to the human eye but our network does a good job of recognizing the fine structures.
 
<hr>