The data is split into 80/20 train test splits.

There are three systems that are running.

The first system reads in words from the train set and increases or decreases the words associated sentiment value based on the sentiment of the overall headline.
Then it will read in headlines from the test set and manipulate a headline sentiment score using the values stored from the first step. 
The final score per headline is just the aggregation of the word sentiment values. 

The second system is pretty much the first system, except it looks at the stems of words instead.

The last system chooses a headline's sentiment at random.

The differences in the predicted and the actual sentiment is calculated using RMSE. 
This is the baseline system.

The first system scores around a 0.471.

The second system scores around a 0.476.

The baseline system scores around a 0.518.