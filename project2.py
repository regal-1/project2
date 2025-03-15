#nearest neighbor algorithm
from random import randrange
import pandas as pd # type: ignore
import numpy as np # type: ignore

def read_data(fileName):  
    #read file using pandas
    df = pd.read_csv(fileName, header=None, delim_whitespace=True)

    #filter rows by class label; boolean indexing
    df_class1 = df[df.iloc[:, 0].astype(int) == 1]
    df_class2 = df[df.iloc[:, 0].astype(int) == 2]

    df_class1 = df_class1.iloc[:, 1:]
    df_class2 = df_class2.iloc[:, 1:]

    df_class1 = df_class1.apply(np.sort, axis=0)
    df_class2 = df_class2.apply(np.sort, axis=0)

    return df_class1, df_class2

#compute distance between two points
def distance(a, b):
    return np.sqrt(((a - b) ** 2))

#finds nearest neighbor in array
def find_nearest(df, feature, data):
    #print("feature: ", feature)
    #print("point: ", data)
    class1 = df.iloc[:, feature]
    dist = np.array([distance(data, x) for x in class1])
    #print("distance: ", dist)
    index_of_nearest = np.argmin(dist)
    #print("nearest: ", dist[index_of_nearest])
    return dist[index_of_nearest]

def feature_search(df1, df2):
    #start empty feature set
    curr_features = []
    num_features = len(df1.columns)

    #for i = 1 to i <= feature, search the ith level
    #try combinations of features, select best at each level, move forward to combos of best feature + others
    for i in range(0, num_features):
        print(f"On the {i+1}th level of the search tree")
        feature_to_add = None
        best_so_far_accuracy = 0
        for j in range(0, num_features):
            #once we add a feature, we should not add it again
            if j not in curr_features:
                print(f"-- Considering adding the {j+1} feature")
                #data = , need to define this
                accuracy =  leave_one_out_cross_validation(df1, df2, curr_features, j)

                #if new accuracy is better than prev accuracy, we can add feature j
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
                
        #add to current features
        if feature_to_add:
            curr_features.append(feature_to_add)
            print(f"On level {i+1}, I added feature {feature_to_add}+1 " f"to current set {curr_features}")

    return curr_features

def leave_one_out_cross_validation(df1, df2, features, i):
    #return randrange(0, 101, 2)
    features.append(i)
    print("features ", features)
    print(df1)
    df_length = len(df1)


    for feature in features:
        print(feature, df1)
        for index in range(0, len(df1)): 
            #save the row in a variable before dropping
            row = df1.iloc[index]
            # Drop a single row at position i
            df1 = df1.drop(df1.index[i])
            d = find_nearest(df1, feature, np.array(row)[feature])
            if index == 0:
                df1 = pd.concat([pd.DataFrame([row]), df1], ignore_index=True)
            elif index == df_length - 1:
                pd.concat([df1, pd.DataFrame([row])]).reset_index(drop=True)
            else:
                df1 = pd.concat([df1.iloc[:index - 1], pd.DataFrame([row]), df1.iloc[index]]).reset_index(drop=True)
            print(df1)
            print("distance = ", d, "feature = ", feature)
    return randrange(0, 101, 2)
            


def debug_csv():
    class1, class2 = read_data("CS170_Small_Data__10.txt")
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