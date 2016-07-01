
# Ensemble of Deep Convolutional Neural Networks for Learning to Detect Retinal Vessels in Fundus Images

Vision impairment due to pathological damage of
the retina can largely be prevented through periodic screening
using fundus color imaging. However the challenge with large
scale screening is the inability to exhaustively detect fine blood
vessels crucial to disease diagnosis. This work presents
a computational imaging framework using deep and ensemble
learning for reliable detection of blood vessels in fundus color
images. An ensemble of deep convolutional neural networks
is trained to segment vessel and non-vessel areas of a color
fundus image. During inference, the responses of the individual
ConvNets of the ensemble are averaged to form the final
segmentation. In experimental evaluation with the DRIVE
database, we achieve the objective of vessel detection with
maximum average accuracy of 91.8%.
<hr>

<img src="./images/01_test_src.png?raw=True" width="250">
<img src="./images/01_manual1.png?raw=True" width="250">
<img src="./images/01_test.png?raw=True" width="250">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fundus image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Manual segmentation&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Predicted Segmentation
<hr>
## Proposed Method

<img src="./images/Proposed-Method.png?raw=True" width="800">

<hr>
## Architecture

<img src="./images/Architecture.png?raw=True" width="800">

<hr>
## Some Results

<img src="./images/Magnified1.png?raw=True" width="800">


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Fundus image &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Magnified Section
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ground Truth
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Prediction