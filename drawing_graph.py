import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

x = []
for i in range (999) :
    x.append(i)
data_frame_raw = pd.read_csv('./csv/yoga1_15fps_.csv', index_col=0, na_values=['None'])
data_frame_nan = data_frame_raw.replace({np.nan: None})
#data_frame = np.array(data_frame_nan)
data_frame_nan.info()
data_frame = data_frame_nan.mean(axis='columns')

data_frame_raw1 = pd.read_csv('./csv/yoga1_1_15fps_.csv', index_col=0, na_values=['None'])
data_frame_nan1 = data_frame_raw1.replace({np.nan: None})
#data_frame = np.array(data_frame_nan)
data_frame_nan1.info()
data_frame1 = data_frame_nan1.mean(axis='columns')

plt.plot( data_frame, label='data_frame')
plt.plot( data_frame1, label='data_frame1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




