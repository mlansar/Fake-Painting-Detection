This is a sample starting kit for the Persodata challenge. 
It uses paintings from WikiArt (the Visual Art Encyclopedia). The data here was preprocessed : the images were resized to a fixed resolution of (224,244,3) and then feature extraction was conducted using MobileNets transforming each image into a vector of size 1024 containing the essential information of the image.
Half of the images in this data set are fake paintings which were generated through neural style transfer and Generative Adversarial Network. 

This starting kit contains 50 samples for each set : training, validation and testing.
The task for this challenge is a binary classification whose goal is to detect the fake paintings.

References and credits: 
https://www.wikiart.org/

Prerequisites:
Install Anaconda Python 3.6.6 

Usage:

(1) If you are a challenge participant:

- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the Iris challenge. 
At the prompt type:
jupyter-notebook README.ipynb

- modify sample_code_submission to provide a better model

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).

(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)

- the following programs run properly:

    `python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

    `python scoring_program/score.py sample_data sample_result_submission scoring_output`

- the metric identified by metric.txt in the utilities directory is the metric used both to compute performances in README.ipynb and for the challenge.

- your code also runs within the Codalab docker (inside the docker, python 3.6 is called python3):

	`docker run -it -v `pwd`:/home/aux codalab/codalab-legacy:py3`
	
	`DockerPrompt# cd /home/aux`
	`DockerPrompt# python3 ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`
	`DockerPrompt# python3 scoring_program/score.py sample_data sample_result_submission scoring_output`
	`DockerPrompt# exit`
