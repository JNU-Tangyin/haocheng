import pandas as pd

#将预测好的reward放到交易框架中测试

cash = 1000  # 初始资金
holding=0
label_window = 10  # 标签窗口大小
df = pd.read_csv('./Data/HBEA/data_processed.csv', parse_dates=['Date'], index_col='Date') 
df=df.dropna()[:-label_window]
df_label=pd.read_csv('./Data/HBEA/HBEA_pred.csv', parse_dates=['Date'], index_col='Date') 
df.reset_index(inplace=True)
df_label.reset_index(inplace=True,drop=True)

for i in range(904,len(df)):
    if df_label.iloc[i].values<0.5:
        share=cash//df.loc[i,'Close']
        cash-=share*df.loc[i,'Close']
        holding+=share
    elif df_label.iloc[i].values>0.5:
        share=holding
        cash+=share*df.loc[i,'Close']
        holding=0
    elif cash<0:
        break
    else:
        continue
    print(f"Date: {df.loc[i,'Date']}, Label: {df_label.iloc[i].values}, Holding: {holding}, Cash: {cash},Total: {cash+holding*df.loc[i,'Close']}")
    

print(f"Final Holding: {holding}, Final Cash: {cash}")
# 交易策略：当标签大于0.8时，买入；当标签小于0.2时，卖出；否则，继续持有。