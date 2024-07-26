import stockstats
import pandas as pd

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