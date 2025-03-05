#nearest neighbor algorithm
import pandas as pd # type: ignore
import numpy as np # type: ignore

def read_data(fileName):
    
    #read file using pandas
    df = pd.read_csv(fileName, header=None, delim_whitespace=True)

    # Filter rows by class label using boolean indexing
    df_class1 = df[df.iloc[:, 0].astype(int) == 1]
    df_class2 = df[df.iloc[:, 0].astype(int) == 2]

    df_class1 = df_class1.iloc[:, 1:]
    df_class2 = df_class2.iloc[:, 1:]

    #convert to 1d array
    return df_class1.to_numpy().flatten(), df_class2.to_numpy().flatten()

#compute distance between two points
def distance(a, b):
    return np.sqrt(((a - b) ** 2))

#finds nearest neighbor in array
def find_nearest(class1, data):
    dist = np.array([distance(data, x) for x in class1])
    nearest = np.argmin(dist)
    return nearest


def debug_csv():
    class1, class2 = read_data("CS170_Small_Data__10.txt")
    print("Class 1:")
    print(class1)
    print("\nClass 2:")
    print(class2)
    index = (find_nearest(class1, 1.1))
    print(index, class1[index])

debug_csv()