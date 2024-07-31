import pandas as pd
import numpy as np
import stockstats
from functools import partial

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
    return landscape.rank(pct=True,ascending=False).iloc[0]#reward

def attach_reward(df,col_name,look_ahead):
    validate(df, ['Open','Close','Low','High','Volume'])#根据自己的数据集调整
    data = df.copy()
    data['reward'] = df[str(col_name)].rolling(window=look_ahead+1, min_periods=look_ahead+1).apply(reward)
    data['reward']= data['reward'].shift(-look_ahead) #计算好以后往上移，与close并列
    return data

reward5 =partial(attach_reward,look_ahead=5)
reward10 = partial(attach_reward,look_ahead=10)
reward20= partial(attach_reward,look_ahead=20)

data=pd.read_csv('./Data/SHEA/SHEA.csv')
labels=attach_reward(data,'Close',5)#导入df,列名为Close,窗口大小为5
#labels=labels.reset_index(drop=True)#重置索引
labels.to_csv('./Data/rolling_mlp/SHEA_label_5.csv',index=False)#保存标签数据