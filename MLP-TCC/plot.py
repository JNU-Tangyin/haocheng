from evaluate import Evaluator
import pandas as pd

label_window = 10  # 标签窗口大小
df = pd.read_csv('./Data/HBEA/data_processed.csv', parse_dates=['Date'], index_col='Date') 
df=df.dropna()[:-label_window]
df_label=pd.read_csv('./Data/HBEA/HBEA_pred.csv', parse_dates=['Date'], index_col='Date') 

hbea=pd.concat([df['Close'],df_label],axis=1)
hbea.columns=['close','reward']
hbea['action'] = hbea.reward.apply(lambda x: 1 if x>0.75 else -1 if x<0.25 else 0)
hbea.reset_index(inplace=True)
hbea.columns=['date','close','reward','action']
hbea.set_index('date', inplace=True)

hbea=hbea[['close','action']][-100:]

ev = Evaluator(hbea)
ev.plot(title ="None")
#ev.short_days()