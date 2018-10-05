# pneumonia-xray-detection
Machine Learning Fall 2018 Project

Run_cnn.py runs the CNN. 

~~~~~~~~~~~~~Parameters~~~~~~~~~~~~~~~~~~~~
Epoch is the number of times the CNN runs through. 
batch size needs to be in multiples of 2
Nodes in the CNN is indicated the code below:

classifier.add(Dense(units = 64, activation = 'relu')) shows that there are 64 nodes.
classifier.add(Dense(units = 1, activation = 'sigmoid'))

Note that 'relu' and 'sigmoid' are rectifier and sigmoid functions for determining weights in the CNN and the last node, respectively.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

input_shape = (100, 100, 3) 
#### indicates the size of the filters can adjust to higher. 3 just means RBG.
target_size = (100, 100),   
#### input_shape and target size have to match.



############################################################################

cam2.py is a class activation map loader that outputs a jpg file with a Class Activation Map, using VG166 CNNs from Oxford. 

#############################################################################

when running using Unix command lines, first make sure you have python and slurm loaded up.
First, make sure that the environment is made and 
run these lines:

python3 -m venv env
source ./env/bin/activate
python -m pip install requirements.txt

Then, submit a job by running this like:

sbatch sbatch.sh

##############################################################################





