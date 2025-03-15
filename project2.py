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
    features.append(i)
    print("features", features)
    # print(df1)
    df_length = len(df1)
    num_matches = 0
    miscount = 0

    for feature in features:
        print("Processing feature:", feature)
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
        
            d1 = find_nearest(df_temp, feature, np.array(row)[feature])
            d2 = find_nearest(df2, feature, np.array(row)[feature])
            # print("Distance d1 from class1 = ", d1, "for feature = ", feature)
            # print("Distance d2 from class2 = ", d2, "for feature = ", feature)
            
            if d1 < d2:
                num_matches = num_matches + 1
            else:
                miscount = miscount + 1
    accuracy = (num_matches/(num_matches + miscount)) * 100
    print("accuracy: ", accuracy)
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