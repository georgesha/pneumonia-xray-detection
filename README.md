# pneumonia-xray-detection
Machine Learning Fall 2018 Project

Run_cnn.py runs the CNN. 

Epoch is the number of times the CNN runs through. 

batch size needs to be multiples of 2

input_shape = (100, 100, 3) #### indicates the size of the filters can adjust to higher. 3 just means RBG.
target_size = (100, 100),   #### input_shape and target size have to match.

classifier.add(Dense(units = 64, activation = 'relu')) shows that there are 64 nodes.
classifier.add(Dense(units = 1, activation = 'sigmoid'))
