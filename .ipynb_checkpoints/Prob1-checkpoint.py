import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# The function that handles the entires algorithm, taking in the training and test data as well as the alpha value for laplas smoothing
def runBayes(training,testing,lap):
  
    # Get the number of rows and columns in the training set
    trainingRows = len(training.index);
    trainingCol = len(training.columns);
  
    # Store just the index of the row and the "CLASS" of the row
    indexResults = training.iloc[:,trainingCol-1];
  
    # Store how often the result is 1 or 0 (i.e. spam or not)
    spamOrNot = training.iloc[:,trainingCol-1].value_counts(sort=True);
  
    # Append 2 columns to the testing dataset to store what the prediction is and if the prediction was correct or not
    testing = testing.reindex(columns=[*testing.columns.tolist(), 'The Prediction', 'The Accuracy']);
  
    # Get the number of rows and columns in the testing set
    testingRows = len(testing.index);
    testingCol = len(testing.columns);
    
    # Store the prior probabilities
    priors = training.iloc[:,trainingCol - 1].value_counts(normalize=True, sort=True);
    
    # Create a dictionary that will be used to create the training model
    likelyDic = {};
  
    # Loop through each column and figure out the likelihood of each 
    for column in range(1, trainingCol):
 
        # Get the number of unique values in this column (i.e. unique number of times a word appeared in each row)
        uniqueWordCount = len(training[training.columns[column]].unique());
        theColVals = training[training.columns[column]];
      
        # Find the likelihood of the word appearing
        for rowIndex in range (0, trainingRows):

            for theVal in range(0, theColVals.size):
                
                # Get the count of the number of times this word shows up as spam if it cannot be done, set the value to 0
                try:
                    spamCount = spamOrNot[rowIndex];
                except:
                    spamCount = 0;
                    
            
                # Get the likelihood using laplace smoothing
                likelihood = ((spamCount + lap) / (trainingRows + (2 * lap)));
    
                # Create a key for storing the likelihood in the dictionary and store it
                theKey = str(training.columns[column]) + str(theColVals.iloc[theVal]) + str(indexResults.iloc[rowIndex]);
                likelyDic[theKey] = likelihood;

  
    # Calculate the predictions for each value row by row
    for row in range(0, testingRows):
  
        currentMax = 0.0
        thePrediction = indexResults.iloc[0]
  
        for theDex in range (0, len(indexResults)):
  
            # Set the starting value to be the prior probability if one exists
            try:
                theNume = priors[theDex]
            except:
                theNume = 0.1
            
            # Loop through all the values in the columns to caclulate the prediction for the row
            for column in range(1, trainingCol):
                
                rowResult = indexResults.iloc[theDex]
                columnName = testing.columns[column]
                columnVal = testing.iloc[row,column]
                
  
                # Set the key that will be used to search the dictionary
                key = str(columnName) + str(columnVal) + str(rowResult)
        
               # Use the value in the dictionary to continue updating
                try:
                    theNume = theNume * likelyDic[key];
                except:
                    theNume *= 1
            
            # Store the prediction if theNume is greater than the currentMax
            if(theNume > currentMax):
                thePrediction = indexResults.iloc[theDex];
                currentMax = theNume;

        # Add the prediction to the column
        testing.iloc[row,testingCol-2] = thePrediction;
        
        # Record if the prediction was correct or not to later calculate the accuracy
        if(thePrediction == testing.iloc[row,trainingCol-1]):
            testing.iloc[row,testingCol-1] = 1;
        else: 
            testing.iloc[row,testingCol-1] = 0;
 
    # Calculate the accuracy
    accuracy = (testing.iloc[:,testingCol-1].sum())/(testingRows)
  
    # Return the accuracy that is produced
    return  accuracy;

# Retrieve the email subjects csv file
emails = pd.read_csv("dbworld_subjects_stemmed.csv")

# Split the data into 80% training and 20% testing
train, test = train_test_split(emails, test_size=0.2, random_state=67)

# Set the value for laplacian smoothing
lap = 1;

# Get the accuracy and print it out
accuracy = runBayes(train, test, lap);

print("Subjects Accuracy: " + str(accuracy))

# Retrieve the email bodies csv file
emails2 = pd.read_csv("dbworld_bodies_stemmed.csv")

# Split the data into 80% training and 20% testing
train, test = train_test_split(emails2, test_size=0.2, random_state=67)

# Get the accuracy and print it out
accuracy = runBayes(train, test, lap);

print("Bodies Accuracy: " + str(accuracy))