#nearest neighbor algorithm
import pandas as pd # type: ignore

def read_data(fileName):
    
    #read file using pandas
    df = pd.read_csv(fileName, header=None, delim_whitespace=True)

    # Filter rows by class label using boolean indexing
    df_class1 = df[df.iloc[:, 0].astype(int) == 1]
    df_class2 = df[df.iloc[:, 0].astype(int) == 2]

    df_class1 = df_class1.iloc[:, 1:]
    df_class2 = df_class2.iloc[:, 1:]

    return df_class1, df_class2

def debug_csv():
    class1, class2 = read_data("CS170_Small_Data__10.txt")
    print("Class 1:")
    print(class1)
    print("\nClass 2:")
    print(class2)

debug_csv()