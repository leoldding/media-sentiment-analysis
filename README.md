#Data

The financial data is already inside of the *data* directory. 
Inside the same directory is the *tweetsPreprocessing.py* file. 
The tweets data is too large to put onto GitHub, but the data can be found on https://www.kaggle.com/datasets/kazanova/sentiment140.
After downloading the data, please rename the file to 'tweetsOriginal.csv', place it inside the *data* directory, and run *tweetsPreprocessing.py*.

#Running the Programs

Use the command line to run the programs. 
Each program has a required argument to denote which data is used.
The argument is either *financial* or *tweets*. For example, *python train.py tweets* trains the tweets model.

#Process
The data is split into two sets. The training/validation set consists of 90% of the data while the test set consists of the remaining 10%. 

The train.py file trains the system through the use of grid search cross validation.
The system uses iterates over different combinations of parameters to find the optimal set of parameters.
Each unique parameter set runs a 10-fold cross validation system on the training/validation data. 
Each set of parameters will receive an average RMSE score from the 10-fold cross validation iterations.
The lowest average RMSE score and its respective parameters is then used to create the final word/sentiment pair text output.

The test.py file tests the system and outputs the final RMSE score. 
The test system reads in the text output from train.py and creates a dictionary. 
It then runs through the test set of data and compares the true value to the calculated value using the dictionary.
It also compares the true value to a randomly chosen sentiment value to be used as a baseline.
At the end, both the system's normalized RMSE and the baseline's normalized RMSE are outputted.