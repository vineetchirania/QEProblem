# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# The problem statement is:
# 
# There is an inter hostel tournament for AOE (Feel free to choose Quake or CS) going on in IIT Delhi. Each hostel’s team comprises of 4 people.  The organizers (who are students who have played a lot of AOE in IIT but are not allowed to participate in this tournament) are trying to be smart and have chosen to create a new map for the tournament (let’s call it map 1).  They know that each map has its own characteristics and different kind of players perform differently on different maps.
# 
# To make things easier for the hostels (so that they can choose a team of 4 among 20 students in each hostel), the orgainzers will be doing the following:
# 
# a.	They will be playing on the new map among themselves (assume there are 30 organizers). They will be playing multiple times and will provide a performance index of each organizer for each game p1ij where i is from 1 to 30 and j is from 1 to 1000 (number of games played)
# 
# b.	They will also be providing data of all the other maps and all the other games that have been played in IITD so far. We assume here that all the potential participants have played sufficient number of games. [pkij - where k is from 2 to 10 (for 9 different maps), i is from 1 to n (where 1 to 30 is for organizers  and 31 to n is for the potential participants), j is from 1 to 10000 which are the number of games played]
# 
# Assume that each hostel wants to choose the 4 people who will have the highest predicted performance index for map 1.
# 
# Help the captains of all the teams in choosing the best team for their hostels for the particular map (map 1).
# 
# You need to mention the methods/models you will be using to solve this problem – Try to minimize, the order of time complexity without compromising with the optima
# 
# 
# 
# 

# <markdowncell>

# This is a regression problem. I am attempting to solve this using linear regression.
# Lets first import some required python libraries

# <codecell>

import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression

# <markdowncell>

# We will build the training dataset first - a numpy array with 3 dimensions. First dim will represent user, second dim will represent map and the third will represent total games played.

# <codecell>

user_count = 200 # 1 to 30 organizers and rest contenders for hostel team
map_count = 9 # Maps for which we know user's score i.e. excluding the test map
game_count = 10000 # Game count per map per player on old maps
new_map_game_count = 1000 # Game count per map per player for new map 
expert_user_count = 30

def build_training_data():
    dataset = np.ndarray(shape=(user_count, map_count, game_count), dtype=np.float32)
    
    # Assuming performance index (aka score) to be a floating point decimal
    
    """Weight distribution: 1
        Players 0 to 14 have mean score of 10 in maps 1-5 and mean score of 6 in maps 6-9
        Players 15 to 29 have mean score of 7 in maps 1-5 and mean score of 10 in maps 6-9
        
        Players 30 to 39 have mean score of 8 in maps 1-5 and mean score of 6 in maps 6-9
        Players 40 to 49 have mean score of 6 in maps 1-5 and mean score of 8 in maps 6-9

        All other players have mean score of 4 in all maps
        
        Standard deviation for all cases is assumed 1 for generating data
    """
    dataset[0:15, 0:5] = np.random.normal(10, 1, size=(15, 5, game_count)) 
    dataset[0:15, 5:9] = np.random.normal(6, 1, size=(15, 4, game_count)) 

    dataset[15:30, 0:5] = np.random.normal(7, 1, size=(15, 5, game_count)) 
    dataset[15:30, 5:9] = np.random.normal(10, 1, size=(15, 4, game_count)) 
    
    dataset[30:40, 0:5] = np.random.normal(8, 1, size=(10, 5, game_count)) 
    dataset[30:40, 5:9] = np.random.normal(6, 1, size=(10, 4, game_count)) 
    
    dataset[40:50, 0:5] = np.random.normal(6, 1, size=(10, 5, game_count)) 
    dataset[40:50, 5:9] = np.random.normal(8, 1, size=(10, 4, game_count)) 
    
    dataset[50:] = np.random.normal(3, 1, size=(150, 9, game_count))
    
    """
    new_map_score is the score for each (organizing) player for the new map
    Players 0 to 15 have mean score of 7 in new map
    Players 16 to 30 have mean score of 9 in new map
    """
    new_map_score = np.ndarray(shape=(expert_user_count, new_map_game_count), dtype=np.float32)
    new_map_score[0:15] = np.random.normal(7, 1, size=(15, new_map_game_count)) 
    new_map_score[15:expert_user_count] = np.random.normal(9, 1, size=(15, new_map_game_count)) 

    return dataset, new_map_score
    
    
all_data, new_map_score = build_training_data()    
print 'Shape of all data is (users, old_map_count, games):',all_data.shape
print 'Shape of new_map_score is (expert_users, games):',new_map_score.shape

# <markdowncell>

# For each person and each map, take the average score for all the games he has played.

# <codecell>

def mean_user_map_score(all_data, new_map_score):
    all_data_averaged = np.average(all_data, axis=2)
    new_map_score_averaged = np.average(new_map_score, axis=1)
    return all_data_averaged, new_map_score_averaged
    
all_data_averaged, new_map_score_averaged = mean_user_map_score(all_data, new_map_score)
print 'Shape of all data after averaging out over all games is (users, old_map_count):',all_data_averaged.shape
print 'Shape of new_map_score after averaging out over all games is (expert_users):',new_map_score_averaged.shape

# <markdowncell>

# Lets normalize the data so that we have a faster convergence

# <codecell>

def normalize(all_data, new_map_score):
    mean = np.mean(all_data)
    std = np.std(all_data)
    print 'Mean:',mean,', Standard deviation:',std
    all_data_normalized = (all_data - mean)/std
    new_map_score_normalized = (new_map_score - mean)/std
    
    return all_data_normalized, new_map_score_normalized
    
all_data_normalized, new_map_score_normalized = normalize(all_data_averaged, new_map_score_averaged)

# <markdowncell>

# Shuffle the data for the organizers as we will need both training and test data from organizers data. So to test the model well we need a good mix.

# <codecell>

np.random.seed(51)
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

# Randomize data
all_data_normalized[:30], new_map_score_normalized[:30] = randomize(all_data_normalized[:30], new_map_score_normalized[:30])
all_data = all_data_normalized
new_map_score = new_map_score_normalized

# <markdowncell>

# We have been able to generate the training data. Now lets fit a linear model. We will split data for the first 30 users into 2 parts - 24 (80%) users data for training and rest 6 for testing and model evaluation.

# <codecell>

def build_and_evaluate_model(all_data, new_map_score):
    # Divide into training and test set
    training_data, test_data = all_data[:24], all_data[24:30]
    training_label, test_label = new_map_score[:24], new_map_score[24:30]
    
    # Build and fit model
    model = LinearRegression()
    print 'Training data shape:',training_data.shape
    print 'Training label shape:',training_label.shape
    model.fit(training_data, training_label)
    print 'Model is:'
    print model
    
    print '\nScore when ran model over training data is:'
    print model.score(training_data, training_label)
    
    print '\nScore when ran model over test data is:'
    print model.score(test_data, test_label)
        
    print '\nScore of organizers in new map is:',new_map_score
    
    predicted_score = model.predict(all_data[30:])
    print '\nPredicted performance index in new map for all participants is:',predicted_score
    print '\nShape of predicted score is:',predicted_score.shape
    
    print '\nNow verifying score:'
    print 'Mean score in new map for users 0 to 30',np.mean(new_map_score[:30])
    print 'Mean score in new map for users 31 to 40',np.mean(predicted_score[:10])
    print 'Mean score in new map for users 41 to 50',np.mean(predicted_score[10:20])
    print 'Mean score in new map for users 51 onwards',np.mean(predicted_score[20:])
    
    return model


model = build_and_evaluate_model(all_data, new_map_score)

# <codecell>

    

