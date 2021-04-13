## Assignment 5 - CNNs on cultural image data

__DESCRIPTION__

Multi-class classification of impressionist painters

So far in class, we've been working with 'toy' datasets - handwriting, cats, dogs, and so on. However, this course is on the application of computer vision and deep learning to cultural data. This week, your assignment is to use what you've learned so far to build a classifier which can predict artists from paintings.

You can find the data for the assignment here: https://www.kaggle.com/delayedkarma/impressionist-classifier-data

Using this data, you should build a deep learning model using convolutional neural networks which classify paintings by their respective artists. Why might we want to do this? Well, consider the scenario where we have found a new, never-before-seen painting which is claimed to be the artist Renoir. An accurate predictive model could be useful here for art historians and archivists!

For this assignment, you can use the CNN code we looked at in class, such as the ShallowNet architecture or LeNet. You are also welcome to build your own model, if you dare - I recommend against doing this.

Perhaps the most challenging aspect of this assignment will be to get all of the images into format that can be fed into the CNN model. All of the images are of different shapes and sizes, so the first task will be to resize the images to have them be a uniform (smaller) shape.

You'll also need to think about how to get the images into an array for the model and how to extract 'labels' from filenames for use in the classification report
<br>
<br>


### Before you run the script
Look at the cnn-artists.py script in the src folder.
I was not able to upload the data file, so you will have to upload that on your own. sorry for the inconvinience.

<br>
"cnn-artists.py" is using the LeNet mothod, while the other scrip in the src folder: "ShallowNet_impressionist.py" are using the ShallowNet method. Just look at the cnn-artists.py file, which also contains arg parses.


### Run the script:

- git clone the script to worker02

- run the bash script create_venv_ass5.sh to set up the venv: __bash create_venv_ass5.sh__

- now move to the src folder: __cd src__

- Here you find two scripts, the script you should run is: __cnn-artists.py__
    - Run it with: python cnn-artists.py
    - The script takes 2 arguments: -k: The size of the kernal
    - And -ps: The size you want the pictures sized down to.

