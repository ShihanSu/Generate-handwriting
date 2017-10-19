# Generative Handwriting using Pytorch
An attempt to implement the random handwriting generation portion of Alex Graves'[paper][1] using Pytorch.

## Requirement
Pytorch
xml.etree.ElementTree
Numpy

## Files
datahp.py Load data and put in the right format
model.py  Model formulation
train.py  Train and save the model

## Data
You need to [register][2] on the website in order to get the IAM On-Line Handwriting data. 
Unzip lineStrokes-all.tar.gz into data subdirectory, so that you end up with data/lineStrokes/a01 ect.
Afterwards, run python train.py to start the training process. 










[1]:https://arxiv.org/abs/1308.0850
[2]:http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database
