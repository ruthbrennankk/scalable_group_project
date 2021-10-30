# scalable_group_project

### Compile/Run Instructions 
*Generate* <br/>
Generation for this project is done using a google colab notebook. You should open one of the notebooks in colab and execute the set up code boxes before executing the generate code block. This will create a training set (size 192,000), validation set (size 19,200) and a test set (size 100)using the captcha symbol set 'ABCDeFghijkMnPQRSTUVWXxYZz0123456789#/\[]:><%{}-+' <br/>
<br/>
*Training* <br/>
Training is also done in google colab, executing the training code block will train using the previously generated training and validation image sets in batches of 32 for 6 epochs. <br/>
<br/>
*Classification* <br/>
We convert the model to a tflite model by executing the convert code block and then that tflite is transfered to the raspberry pi using git. And the classify.py script is executed using that model to classify the images.

#### File Retrieval
--TODO

#### Pre and Post Processing
We trialled multiple pre-processing approaches, which can be found in our preprocessing file. We found that less seemed to be more in this case and ended up with the preprocessing suggested in the original code we downloaded from blackboard.

####  Image Set Generation
We generated images in google colab notebooks. <br/>
We altered the provided generate.py to generate images of random length using the numpy rand function in the range 0 to 6 to randomly select the length of all images in the set. We did not use padding at this stage but instead adding spaces to the ends of any image with a length less than 6 while training. <br/>
Since some special characters can't be used for filenames we decided to implement a mapping functionality for the image names. We stored the label for each image in a set_labels.txt file and mapped the image name to it via indexes. This label file is called during training. <br/>
For the final model we used 192,000 training images, this gave us 6,000 batches per epoch. We generated and used 19,200 validation images as well. <br/>

#### Submitty Solving
We started by working with models we had from the previous assignment. We each trained a model over what we deemed to be a whole symbol set by combining what our classmates suggested on piazza with what we observed from our own sets. We then began to reduce that symbol set through eliminations and submissions. <br/>

####  Timing information
| Metric          | Time        |
| -----------     | ----------- |
| *Generate*                    |
| Train 192000    | 13m 58s     |
| Val 19200       | 1m 6s       |
| *Training*                    |
| Per Epoch       | 10-18m      |
| *Classification*              |
| For 2000 images | 10m         |
