#nearest neighbor algorithm
from random import randrange
import sys
import pandas as pd # type: ignore
import numpy as np # type: ignore
import copy 

def read_data(fileName):  
    #read file using pandas
    df = pd.read_csv(fileName, delimiter=r'\s+', header=None)

    #filter rows by class label; boolean indexing
    df_class1 = df[df.iloc[:, 0].astype(int) == 1]
    df_class2 = df[df.iloc[:, 0].astype(int) == 2]

    df_class1 = df_class1.iloc[:, 1:]
    df_class2 = df_class2.iloc[:, 1:]

    return df_class1, df_class2

#compute distance between two points
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

#finds nearest neighbor in array
def find_nearest(df, features, data):
    myclass = df.iloc[:, features].to_numpy()
    dist = np.array([distance(data, x) for x in myclass])
    index_of_nearest = np.argmin(dist)
    return dist[index_of_nearest]

def feature_search_forward(df1, df2, alg):
    #start empty feature set
    num_features = len(df1.columns)
    best_so_far_accuracy = 0
    if alg == "1":
        curr_features = []
    if alg == "2":
        curr_features = [x for x in range(num_features)]
        best_so_far_accuracy = leave_one_out_cross_validation(df1, df2, curr_features)
    #for i = 1 to i <= feature, search the ith level
    #try combinations of features, select best at each level, move forward to combos of best feature + others
    for i in range(0, num_features):
        print(f"On level {i+1} of the search tree")
        feature_to_add = None
        accuracy = [0] * num_features
        for j in range(0, num_features):
            #once we add a feature, we should not add it again
            if alg == "1":
                if j not in curr_features:
                    features = copy.deepcopy(curr_features)
                    features.append(j)
                    accuracy[j] =  leave_one_out_cross_validation(df1, df2, features)
            if alg == "2":
                    if j in curr_features:
                        features = [x for x in curr_features if x != j ]
                        accuracy[j] =  leave_one_out_cross_validation(df1, df2, features)
 
        #print("accuracy: ", accuracy)

        for j in range(0, num_features):
            #if new accuracy is better than prev accuracy, we can add feature j
            if accuracy[j] > best_so_far_accuracy:
                best_so_far_accuracy = accuracy[j]
                feature_to_add = j

        #add to current features
        if alg == "1" and feature_to_add != None:
            curr_features.append(feature_to_add)
            print(f"On level {i+1}, I added feature {feature_to_add+1} " f"to current set {[x + 1 for x in curr_features]}")

        if alg == "2" and feature_to_add != None:
            curr_features = [x for x in curr_features if x != feature_to_add ]
            print(f"On level {i+1}, I removed feature {feature_to_add+1} " f"from current set {[x + 1 for x in curr_features]}")

    print(f"Finished search!! The best feature subset is {[x + 1 for x in curr_features]} which has an accuracy of {best_so_far_accuracy}.")
    return curr_features 


def leave_one_out_cross_validation(class1_df, class2_df, features): 
    
    hit = 0
    miss = 0
    tie = 0

    for cl in range(2):
        if cl == 0:
            df1 = class1_df
            df2 = class2_df
        else:
            df1 = class2_df
            df2 = class1_df
        df_length = len(df1)
        for index in range(df_length):
            #making a copy so the original df1 remains unchanged
            df_temp = df1.copy()
            
            #saving row to leave out
            row = df_temp.iloc[index]
            
            #temp df w the row dropped
            df_temp = df_temp.drop(df_temp.index[index]).reset_index(drop=True)
        
            d1 = find_nearest(df_temp, features, np.array(row)[features])
            d2 = find_nearest(df2, features, np.array(row)[features])

            
            if d1 < d2:
                hit += 1
            elif d2 < d1:
    
                miss += 1
            else:
                tie += 1
    total = hit + miss + tie
    accuracy = (hit/total) * 100
    accuracy = round(accuracy, 2)
    print(f"Using feature(s) {[x + 1 for x in features]}, accuracy is ", accuracy)
    return accuracy
                        
def main():
    dataset = input("Type in the name of the file to test:" + '\n')
    alg = input("Type in the number of the algorithm you would like to use:" + '\n'
                      "1) Forward Selection" + '\n'
                      "2) Backward Elimination" + '\n')
    class1, class2 = read_data(dataset)
    print(f"This dataset has {len(class1.columns)} features (not including the class attribute), " 
          f"with {len(class1)+ len(class2)} instances.")
    if alg == "1" or alg == "2":
        feature_search_forward(class1, class2, alg)
    else:
        print("Invalid input")
main()