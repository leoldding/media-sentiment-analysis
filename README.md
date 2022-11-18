The data is split into two sets. The training/validation set consists of 90% of the data while the test set consists of the remaining 10%. 

The train.py file trains the system. 
The system uses iterates over different combinations of parameters to find the optimal set of parameters (essential sklearn's GridSearchCV).
Each parameter set then runs a 10-fold cross validation system on the training/validation data. 
Each set of parameters will then receive an average RMSE score from the 10-fold cross validation scoring.
The lowest average RMSE score and its respective parameters is then used to create the final word/sentiment pair text output.

The test.py file tests the system and outputs the final RMSE score. 
The test system reads in the text output from train.py and creates a dictionary. 
It then runs through the test set of data and compares the true value to the calculated value using the dictionary.
It also compares the true value to a randomly chosen sentiment value to be used as a baseline.
At the end, both the system's normalized RMSE and the baseline's normalized RMSE are outputted.