import pandas as pd
import os


def merge() :

    name = './csv/yoga1_15fps_.csv'
    if (os.path.isfile(name)) :
        name = './csv/yoga1_1_15fps_.csv'
    
    file_list = os.listdir('./temp_csv/')
    for idx, val in enumerate (file_list) :
        file_list[idx] = './temp_csv/' + val

    dataFrame = pd.concat(map(pd.read_csv, file_list), ignore_index=True)
    dataFrame.to_csv(name, na_rep='None')
    for i in range (len(file_list)) :
        file = file_list.pop()
        os.remove(file)