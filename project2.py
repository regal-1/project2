import sys
import pandas as pd  #type: ignore
import numpy as np  #type: ignore
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

def leave_one_out_cross_validation(class1_df, class2_df, features): 
    if not features:
        total = len(class1_df) + len(class2_df)
        minority_count = min(len(class1_df), len(class2_df))
        baseline_accuracy = round((minority_count / total) * 100, 2)
        print(f"For the set of no features ({{}}), the accuracy is {round(baseline_accuracy, 1)}%.")
        return baseline_accuracy
    
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
    accuracy = ((hit + tie) / (hit + miss + tie)) * 100
    
    print(f"Using feature(s) {set([x + 1 for x in features])}, accuracy is {round(accuracy, 1)}%." )
    return accuracy
                        
def feature_search(df1, df2, alg):
    #start empty feature set
    num_features = len(df1.columns)

    #calculate baseline accuracy using all features before starting the search
    baseline_accuracy = leave_one_out_cross_validation(df1, df2, list(range(num_features)))
    print(f"Beginning search. \nRunning nearest neighbor with all {num_features} features, using 'leave-one-out' evaluation, I get an accuracy of: {baseline_accuracy}%")
    
    if alg == "1":
        curr_features = []
    elif alg == "2":
        curr_features = [x for x in range(num_features)]
    
    best_so_far_accuracy = baseline_accuracy
    best_feature_set = curr_features.copy()

    #to store the best accuracy per level
    level_best_accuracies = []

    #for i = 1 to i <= feature, search the ith level
    #try combinations of features, select best at each level, move forward to combos of best feature + others
    for i in range(0, num_features):
        #print(f"\nOn level {i+1} of the search tree")
        accuracy = [0] * num_features
        #once we add a feature we should not add it again
        for j in range(0, num_features):
            if alg == "1":
                if j not in curr_features:
                    features = copy.deepcopy(curr_features)
                    features.append(j)
                    accuracy[j] = leave_one_out_cross_validation(df1, df2, features)
            if alg == "2":
                if j in curr_features:
                    features = [x for x in curr_features if x != j]
                    accuracy[j] = leave_one_out_cross_validation(df1, df2, features)

        #fetermine the best feature for this level regardless of overall accuracy
        level_best_accuracy = 0  
        best_feature = None        
        for j in range(0, num_features):  
            if alg == "1" and j not in curr_features:  
                if accuracy[j] > level_best_accuracy:  
                    level_best_accuracy = accuracy[j]   
                    best_feature = j                    
            elif alg == "2" and j in curr_features:       
                if accuracy[j] > level_best_accuracy:       
                    level_best_accuracy = accuracy[j]       
                    best_feature = j                        

        #update overall best accuracy if the current level's best is higher
        if level_best_accuracy > best_so_far_accuracy:  
            best_so_far_accuracy = level_best_accuracy
            if alg == "1":
                best_feature_set = curr_features.copy() + [best_feature]
            elif alg == "2":
                best_feature_set = [x for x in curr_features if x != best_feature]

        #break if no best feature per level (empty set)
        if best_feature is None:  
            break                

        feature_to_add = best_feature  

        #add to current features
        if alg == "1":
            curr_features.append(feature_to_add)
            print(f"Feature set {set(x + 1 for x in curr_features)} was the best, accuracy is {round(level_best_accuracy, 1)}.")
        if alg == "2":
            curr_features.remove(feature_to_add)
            if len(set(x + 1 for x in curr_features)) != 0:
                print(f"Feature set {set(x + 1 for x in curr_features)} was the best, accuracy is {round (level_best_accuracy, 1)}.")

        level_best_accuracies.append(level_best_accuracy)  

    print(f"\nFinished search!! The best feature subset is {set(x + 1 for x in best_feature_set)} which has an accuracy of {round(best_so_far_accuracy, 1)}%.")
   
    return curr_features  

def main():
    dataset = input("Welcome to Rishika Mundada's Feature Selection Algorithm. \nType in the name of the file to test:\n")
    alg = input("Type in the number of the algorithm you would like to use:\n"
                "1) Forward Selection\n"
                "2) Backward Elimination\n")
    class1, class2 = read_data(dataset)
    print(f"This dataset has {len(class1.columns)} features (not including the class attribute), with {len(class1) + len(class2)} instances.")
    if alg == "1" or alg == "2":
        feature_search(class1, class2, alg)
    else:
        print("Invalid input")

main()
