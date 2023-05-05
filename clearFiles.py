import os

def clear(sort, name):
    file_list = os.listdir('./images/')
    file_list.sort()
    for i in range(len(file_list)):
        file = file_list.pop()
        os.remove('./images/' + file)

    os.remove('./csv/' + sort + name + '_15fps_.csv')
    os.remove('./csv/' + sort + name + '_land.csv')
        

