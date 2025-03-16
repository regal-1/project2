#nearest neighbor algorithm
from random import randrange
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

    # df_class1 = df_class1.apply(np.sort, axis=0)
    # df_class2 = df_class2.apply(np.sort, axis=0)

    return df_class1, df_class2

#compute distance between two points
def distance(a, b):
    # print("a in distance = ", a)
    # print("b in distance = ", b)
    return np.sqrt(np.sum((a - b) ** 2))

#finds nearest neighbor in array
def find_nearest(df, features, data):
    # print("feature: ", features)
    # print("point: ", data)
    myclass = df.iloc[:, features].to_numpy()
    # print("myclass = ", myclass)
    dist = np.array([distance(data, x) for x in myclass])
    # print("distance: ", dist)
    index_of_nearest = np.argmin(dist)
    # print("nearest: ", dist[index_of_nearest])
    return dist[index_of_nearest]

def feature_search(df1, df2):
    #start empty feature set
    curr_features = []
    num_features = len(df1.columns)
    best_so_far_accuracy = 0

    #for i = 1 to i <= feature, search the ith level
    #try combinations of features, select best at each level, move forward to combos of best feature + others
    for i in range(0, num_features):
        print(f"On the {i+1} level of the search tree")
        feature_to_add = None
        accuracy = [0] * num_features
        for j in range(0, num_features):
            #once we add a feature, we should not add it again
            if j not in curr_features:
                print(f"-- Considering adding feature {j+1}")
                accuracy[j] =  leave_one_out_cross_validation(df1, df2, curr_features, j)
        
        print("accuracy: ", accuracy)

        for j in range(0, num_features):
            print("acccuracy: ", accuracy[j], "bsf: ", best_so_far_accuracy)
            #if new accuracy is better than prev accuracy, we can add feature j
            if accuracy[j] > best_so_far_accuracy:
                best_so_far_accuracy = accuracy[j]
                feature_to_add = j
                print("feature to add: ", feature_to_add)
                
        #add to current features
        if feature_to_add != None:
            curr_features.append(feature_to_add)
            print(f"On level {i+1}, I added feature {feature_to_add+1} " f"to current set {[x + 1 for x in curr_features]}")

    print(f"Feature set {[x + 1 for x in curr_features]} was the best. Accuracy was {best_so_far_accuracy}.")
    return curr_features

def leave_one_out_cross_validation(class1_df, class2_df, in_features, i):
    features = copy.deepcopy(in_features)
    features.append(i)
    # print("features", features + 1)
    # print(df1)
    
    hit = 0
    miss = 0
    tie = 0

    #for feature in features:
    # print("Processing feature:", features + 1)
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
            # print("Before dropping row at index:", index)
            # print(df_temp.head(3))
            
            #temp df w the row dropped
            df_temp = df_temp.drop(df_temp.index[index]).reset_index(drop=True)
            # print("After dropping row at index:", index)
            # print(df_temp.head(3))
        
            d1 = find_nearest(df_temp, features, np.array(row)[features])
            d2 = find_nearest(df2, features, np.array(row)[features])
            # print("Distance d1 from class1 = ", d1, "for feature = ", features)
            # print("Distance d2 from class2 = ", d2, "for feature = ", features)
            
            if d1 < d2:
                hit += 1
            elif d2 < d1:
                # print("Distance d1 from class1 = ", d1, "for feature = ", features)
                # print("Distance d2 from class2 = ", d2, "for feature = ", features)
                miss += 1
            else:
                tie += 1
    total = hit + miss + tie
    accuracy = (hit/total) * 100
    accuracy = round(accuracy, 2)
    print("accuracy: ", accuracy)
    print("hit: ", hit)
    print("miss = ", miss)
    return accuracy
                        
def debug_csv():
    class1, class2 = read_data("CS170_Small_Data__98.txt")
    print("Class 1:")
    print(class1)
    print("\nClass 2:")
    print(class2)
    sample_data = 2
    feature = 2
    #index = (find_nearest(class1, feature, sample_data))
    #print(index, class1.iloc[index, feature])
    feature_search(class1, class2)


debug_csv()