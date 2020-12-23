import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

csv_path = '/home/member/Workspace/haimd/classfication_pytorch/train.csv'

data = pd.read_csv(csv_path)
x, y = data.iloc[:, 0], data.iloc[:, 1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

df_train = pd.DataFrame.from_dict({'image_id': x_train, 'label':y_train})
df_test = pd.DataFrame.from_dict({'image_id': x_test, 'label':y_test})
df_train.to_csv('tn.csv', index=False)
df_test.to_csv('tt.csv', index=False)
# import ipdb; ipdb.set_trace()