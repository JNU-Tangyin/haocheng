import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import stockstats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from tensorflow.keras.losses import MeanSquaredError
from keras.layers import Dropout
from keras.optimizers import SGD

df = pd.read_csv('./Data/SHEA/SHEA.csv')
df.set_index('Date', inplace=True)

LookBack=20

def validate(d, required, typeof=[pd.DataFrame, stockstats.StockDataFrame],index_type=None):
    '''
    validate whether or not:
    1.The type of d in list'typeof'
    2.d should not be empty
    3.d at least contain columns desinated by`required`
    4.The type of d.index in list`index type`
    '''

    if d.empty:
        raise ValueError("The DataFrame must not be empty.")
    if not any(isinstance(d,i)for i in typeof):# 必须是其中之一
        raise TypeError(f"Type must be either of {typeof}")
    if not all(f in d.columns for f in required):# 所有字段都必须存在
        raise ValueError(f"The DataFrame must contain columns:{required}")
    if index_type is not None:
        if not any(isinstance(d.index,i)for i in index_type):# index的类型
            raise ValueError(f"Index type must be one of {index_type}")
        

def reward(landscape:pd.Series)->float:
    return landscape.rank(pct=True,ascending=False).iloc[0]#reward

def attach_reward(df,col_name,look_ahead):
    validate(df, ['Open','Close','Low','High','Volume'])#根据自己的数据集调整
    data = df.copy()
    data['reward'] = df[str(col_name)].rolling(window=look_ahead+1, min_periods=look_ahead+1).apply(reward)
    data['reward']= data['reward'].shift(-look_ahead) #计算好以后往上移，与close并列
    return data

windows_size=5

df=attach_reward(df,'Close',windows_size)
df.dropna(inplace=True)
labels=df.iloc[:,-1]

class model():
    def __init__(self, data,LookBack):
        # 数据标准化
        scaler = StandardScaler()
        self.data = data
        self.features = [x.values for x in self.data.iloc[:,:-1].rolling(LookBack)][LookBack-1:]
        self.features = np.array([np.array(x.astype(np.float32)).flatten() for x in self.features])
        self.features=scaler.fit_transform(self.features)
        self.labels=self.data.iloc[LookBack-1:,-1]
        self.labels=np.array(self.labels)
        #self.labels.reset_index(drop=True, inplace=True)
        # 构建 MLP 模型
        self.model = Sequential()
        self.model.add(Dense(256, input_dim=len(self.features[0]), activation='LeakyReLU',kernel_regularizer=l2(0.01)))
        self.model.add(BatchNormalization())  # 添加BatchNormalization层
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(512, activation='LeakyReLU',kernel_regularizer=l2(0.01)))
        self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.1))
        self.model.add(Dense(1024, activation='LeakyReLU',kernel_regularizer=l2(0.01)))
        self.model.add(BatchNormalization())
        #self.model.add(Dropout(0.1))
        #self.model.add(Dense(1, activation='linear'))  
        self.model.add(Dense(1, activation='sigmoid'))  # 输出层
    def train(self):
        # 编译模型
        self.model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-5), loss='mean_absolute_error')#SGD(learning_rate=0.01, decay=1e-5)
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.3, random_state=42)
        # 训练模型
        self.model.fit(X_train, y_train,validation_split=0.2, epochs=600,batch_size=256, verbose=1)
        # 评估模型
        y_pred = self.model.predict(X_train)
        loss = mean_squared_error(y_train, y_pred)
        print(f'Train Loss: {loss}')
        loss=r2_score(y_train, y_pred)
        print(f'Train R2 Score: {loss}')
        mae = mean_absolute_error(y_train, y_pred)
        print(f'Train MAE: {mae}')
        msae = mean_squared_log_error(y_train, y_pred)
        print(f'Train MSAE: {msae}')
        y_pred = self.model.predict(X_test)
        loss = mean_squared_error(y_test, y_pred)
        print(f'Test Loss: {loss}')
        loss=r2_score(y_test, y_pred)
        print(f'Test R2 Score: {loss}')
        mae = mean_absolute_error(y_test, y_pred)
        print(f'Test MAE: {mae}')
        msae = mean_squared_log_error(y_test, y_pred)
        print(f'Test MSAE: {msae}')
    
    def predict(self, input_data):
        # 预测动作序列
        action=self.model.predict(input_data)
        return action
    
    def save_model(self, path):
        # 保存模型
        self.model.save(path)
        return

def rolling_mlp(df, model,LookBack):
    scaler = StandardScaler()
    test_input = [x.values for x in df.iloc[:,:-1].rolling(LookBack)][LookBack-1:]
    test_input = np.array([np.array(x.astype(np.float32)).flatten() for x in test_input])
    test_input=scaler.fit_transform(test_input)
    action_sequence = [model.predict(np.array([bb])).item() for bb in test_input]
    return action_sequence


model_reward=model(df,20)
model_reward.train()
model_reward.save_model('./Data/rolling_mlp/SHEA_model_reward')
#model_reward_diff=model(df,20)
#model_reward_diff.train()
#model_reward_diff.save_model('./Data/rolling_mlp/HBEA_model_reward')
#model = tf.keras.models.load_model('./Data/rolling_mlp/HBEA_model')
action_sequence = rolling_mlp(df, model_reward,LookBack)
df=df[LookBack-1:]
action_sequence=pd.DataFrame(action_sequence,columns=['pred'])
action_sequence.index=df.index
action_sequence=pd.concat([action_sequence,df],axis=1)
action_sequence.to_csv('./Data/rolling_mlp/SHEA_pred.csv')