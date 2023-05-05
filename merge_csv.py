import pandas as pd
import os


def merge(sort, names) :
    name = './csv/' + sort + '_15fps_.csv'
    if (os.path.isfile(name)) :
        name = './csv/' + sort + names + '_15fps_.csv'
    
    file_list = os.listdir('./temp_csv/')
    file_list.sort()
    for idx, val in enumerate (file_list) :
        file_list[idx] = './temp_csv/' + val

    dataFrame = pd.concat(map(pd.read_csv, file_list), ignore_index=True)
    dataFrame.to_csv(name, na_rep='None')
    for i in range (len(file_list)) :
        file = file_list.pop()
        os.remove(file)

    
def merge_land(sort, names) :
    name = './csv/' + sort + names + '_land.csv'
    file_list = os.listdir('./temp_land_csv/')
    file_list.sort()
    for idx, val in enumerate (file_list) :
        file_list[idx] = './temp_land_csv/' + val

    dataFrame = pd.concat(map(pd.read_csv, file_list), ignore_index=True)
    dataFrame.to_csv(name, na_rep='None')
    for i in range (len(file_list)) :
        file = file_list.pop()
        os.remove(file)