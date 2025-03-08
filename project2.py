#nearest neighbor algorithm
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
    class1 = df.iloc[:, feature]
    dist = np.array([distance(data, x) for x in class1])
    nearest = np.argmin(dist)
    return nearest

def debug_csv():
    class1, class2 = read_data("CS170_Small_Data__10.txt")
    print("Class 1:")
    print(class1)
    print("\nClass 2:")
    print(class2)
    sample_data = 2
    feature = 2
    index = (find_nearest(class1, feature, sample_data))
    print(index, class1.iloc[index, feature])

debug_csv()

def feature_search(df):
    #start empty feature set
    curr_features = []
    num_features = df.columns

    #for i = 1 to i <= feature, search the ith level
    #try combinations of features, select best at each level, move forward to combos of best feature + others
    for i in range(num_features):
        print(f"On the {i+1}th level of the search tree")
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0
        for j in range(num_features):
            #once we add a feature, we should not add it again
            if j not in curr_features:
                print(f"-- Considering adding the {j+1} feature")
                #data = , need to define this
                accuracy = leave_one_out_cross_validation(data, curr_features, j)

                #if new accuracy is better than prev accuracy, we can add feature j
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = j
                
        #add to current features
        curr_features.append(feature_to_add)
        print(f"On level {i+1}, I added feature {feature_to_add_at_this_level+1} " f"to current set {curr_features}")

    return curr_features

def leave_one_out_cross_validation(somedata, features, i):
    print("tbd")


