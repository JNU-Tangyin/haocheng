import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import stockstats
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

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
    '''
    close=landscape.iloc[0]
    rk = landscape[1:].rank(pct=True)# 取slide window第一行以外的数据
    df=pd.DataFrame({"rk":rk,"price":landscape[1:]})
    reward=(
            df[df.price>close].rk.min() if not df[df.price>close].empty else 0
            +df[df.price<close].rk.max() if not df[df.price<close].empty else 0
            )/2
    '''
    return landscape.rank(pct=True).iloc[0]#reward

def attach_reward(df,col_name,look_ahead):
    validate(df, ['Open','Close','Low','High','Volume'])#根据自己的数据集调整
    data = df.copy()
    data['reward'] = df[str(col_name)].rolling(window=look_ahead+1, min_periods=look_ahead+1).apply(reward)
    data['reward']= data['reward'].shift(-look_ahead) #计算好以后往上移，与close并列
    return data




# 读取数据
df = pd.read_csv('./Data/HBEA/HBEA.csv')  # 替换为实际的文件路径

# 数据预处理
# 假设 'Date' 字段不需要，'Volume' 和 'reward' 是数值型
df['Date'] = pd.to_datetime(df['Date'])
df['Open_1'] = df['Open'].shift(1)
df['High_1'] = df['High'].shift(1)
df['Low_1'] = df['Low'].shift(1)
df['Close_1'] = df['Close'].shift(1)
df['Adj Close_1'] = df['Adj Close'].shift(1)
df['Volume_1'] = df['Volume'].shift(1) 
# 提取年份、月份、星期几等作为特征
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday
#df = df.drop(columns=['Date'])
df.set_index('Date', inplace=True)
# 选择窗口大小
window_size = 5
# 使用rolling函数计算滚动窗口的统计量
df['Open_rolling_mean'] = df['Open'].rolling(window=window_size).mean()
df['High_rolling_mean'] = df['High'].rolling(window=window_size).mean()
df['Low_rolling_mean'] = df['Low'].rolling(window=window_size).mean()
df['Close_rolling_mean'] = df['Close'].rolling(window=window_size).mean()
df['Adj Close_rolling_mean'] = df['Adj Close'].rolling(window=window_size).mean()
df['Volume_rolling_mean'] = df['Volume'].rolling(window=window_size).mean()

df.to_csv('./Data/HBEA/data_processed.csv')

df.dropna(inplace=True)

label_window=10

labels=attach_reward(df,'Close',label_window)
#labels=labels.reset_index(drop=True)
labels.to_csv('./Data/HBEA/HBEA_label.csv',index=True)

# 特征选择
X = df.copy()
X=X[:-label_window]
y = labels.copy().dropna()['reward']

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建 MLP 模型
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='LeakyReLU'))
model.add(Dense(128, activation='LeakyReLU'))
model.add(Dense(64, activation='LeakyReLU'))
model.add(Dense(1, activation='linear'))  # 输出层

# 编译模型
model.compile(optimizer='rmsprop', loss='mean_absolute_error')

# 训练模型
model.fit(X_train, y_train,validation_split=0.2, epochs=600,batch_size=256, verbose=1)

# 评估模型
#loss = model.evaluate(X_test, y_test, verbose=0)
scaler = MinMaxScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
y_pred = model.predict(X_train)
y_pred = scaler.fit_transform(y_pred)
loss = mean_squared_error(y_train, y_pred)
print(f'Train Loss: {loss}')

y_pred = model.predict(X_test)
y_pred = scaler.fit_transform(y_pred)
loss = mean_squared_error(y_test, y_pred)
print(f'Test Loss: {loss}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {mae}')
msae = mean_squared_log_error(y_test, y_pred)
print(f'Test MSAE: {msae}')

y_pred = model.predict(X)
y_pred = scaler.fit_transform(y_pred)
loss = mean_squared_error(y, y_pred)
print(f'Total Loss: {loss}')


y_pred=pd.DataFrame(y_pred,columns=['pred'])
y_pred.index=y.index
y_pred.to_csv('./Data/HBEA/HBEA_pred.csv',index=True)
