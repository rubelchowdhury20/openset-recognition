## Open Set Recognition

###### This repository is PyTorch implementation of the paper https://arxiv.org/pdf/1811.04110v2.pdf

### Command to train the model
python3 main.py --batch_size 512 --unfreeze_epochs 300

### Dataset
Keep the data inside the dataset folder in the of train and val folders.
Where in each of these there is a folder for a class and one extra folder named "unknown_class"


### Status of the experiment
Here the rendered images are used for the training even for the "unknown_class" and the real images are kept in "val" folder for validation purpose.

They key parameters of the experiment are the the "knownsMinimumnMag", this is nothing but the threshold value which is being used to push the summation of the magnitude of the known classes to be more than this threshold value and for the unknown classes it should be less than this value.
In the latest experiment this value is kept as 5. And we concluded this value after checking what is the summation of maginitude values are coming initially and considering that, this value was selected.

The other important parameters are weights for the two losses named, cross_entropy_loss and ring_loss(objectosphere loss). Right now they are kept at 1 and 0.2.
Reason being, whenever ring_loss was having equal weight to the other loss the total loss was shooting up. So, to keep the training stable, it was kept low and moreover there was difference in the loss values of both of them that's why the weights are different.

The outcome till now, there was a significant difference between the magnitude values found in the validation data for known and unknown classes.
It can be seen in the ipynb file.

But the same when was implemented with the retinanet flow to reduce the number of False Positives, it didn't work.
Possible explanation can be that this network is not trained on the retinanet output crops. As in having the augmentaion of random crops might solve this issue.
And along with that false poitives can be from one the classes from different known class.

###### The modification needed in the retinanet flow for accuracy_check.py, which is the changes in the file updated_eval.py of retinanet utils is also kept in the repo. Something similar should be implemented for retina_infer.py code also.