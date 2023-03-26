import pandas as pd
import numpy as np

# csv 불러오기
def read_csv(csv_path, file_name, csv) :
    data_frame_raw = pd.read_csv(csv_path+file_name+csv, index_col=0, na_values=['None'])
    data_frame_nan = data_frame_raw.replace({np.nan: None})
    data_frame = np.array(data_frame_nan)
    return data_frame
