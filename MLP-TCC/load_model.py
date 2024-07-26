import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

#with open('./Data/HBEA/HBEA_model.pkl','rb') as f:
    #model = joblib.load(f)

model = tf.keras.models.load_model('./Data/HBEA/HBEA_model')

df = pd.read_csv('./Data/HBEA/HBEA.csv')

# 数据预处理
df.dropna(inplace=True)
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
df.dropna(inplace=True)
X = df.copy()
label_window=10
X=X[:-label_window]
X_origin=X.copy()
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Now you can use the loaded_model for predictions or further processing
#model.summary()
#feature = model.booster_.feature_name()
y_pred = model.predict(X)

y_pred=pd.DataFrame(y_pred,columns=['pred'])
y_pred.index=X_origin.index
y_pred=pd.concat([X_origin.dropna(),y_pred],axis=1)
y_pred.to_csv('./Data/HBEA/HBEA_pred.csv',index=True)
