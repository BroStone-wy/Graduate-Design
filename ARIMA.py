import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import acf,pacf,plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

#画出总的时序图
data=pd.read_csv('C:\\Users\\Administrator\\Desktop\\Purchase Redemption Data\\user_balance_table.csv', index_col='report_date',parse_dates=['report_date'])
df=data.groupby(['report_date'])['total_redeem_amt'].sum()
df1=data.groupby(['report_date'])['total_purchase_amt'].sum()
total_purchase_amt=pd.Series(df1,name='value')
total_redeem_amt=pd.Series(df,name='value')


fig=plt.figure(figsize=(12,8))
plt.plot(total_purchase_amt,label='每日申购量',color='blue')
plt.plot(total_redeem_amt,label='每日赎回量',color='red')
plt.legend(loc='best')
plt.title('第三区间每日申购赎回时序图')
plt.xlabel('日期')
plt.ylabel('单位：分')
plt.rcParams['font.sans-serif'] = [u'SimHei']#显示中文
plt.rcParams['axes.unicode_minus'] = False
plt.show()


#选择训练集和测试集
purchase_seq_train = total_purchase_amt['2014-04-01':'2014-07-31']
purchase_seq_test = total_purchase_amt['2014-08-01':'2014-08-30']
#purchase_seq_test.to_csv('C:\\Users\\Administrator\\Desktop\\Purchase Redemption Data\\purchase_seq_test.csv')

#对原序列和差分序列进行平稳性检验
timeseries_diff1 = purchase_seq_train.diff(1)
timeseries_diff1 = timeseries_diff1.fillna(0)

timeseries_adf = ADF(purchase_seq_train.tolist())
timeseries_diff1_adf = ADF(timeseries_diff1.tolist())

print('timeseries_adf : ', timeseries_adf)
print('timeseries_diff1_adf : ', timeseries_diff1_adf)

plt.figure(figsize=(12, 8))
plt.plot(purchase_seq_train, label='Original', color='blue')
plt.plot(timeseries_diff1, label='Diff1', color='red')
plt.legend(loc='best')
plt.show()

#白噪声序列检验

#print(acorr_ljungbox(timeseries_diff1,lags=7,boxpierce=False))

#作一阶差分序列ACF和PACF图

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
sm.graphics.tsa.plot_acf(timeseries_diff1,lags=20,ax=ax1)
ax1.set(title='申购一阶差分序列自相关图')
ax2=fig.add_subplot(212)
sm.graphics.tsa.plot_pacf(timeseries_diff1,lags=20,ax=ax2)
ax2.set(title='申购一阶差分序列偏自相关图')
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()

#利用BIC和Aic找p和q值
time_evaluate=sm.tsa.arma_order_select_ic(timeseries_diff1,ic=['aic','bic'],trend='nc',max_ar=5,max_ma=5)
print('trend AIC',time_evaluate.aic_min_order)
print('trend BIC',time_evaluate.bic_min_order)

#ARIMA模型训练
def ARIMA_Model(timeseries,order):
    model =ARIMA(timeseries,order=order)
    return model.fit(disp=0)

time_model=ARIMA_Model(purchase_seq_train,(3,1,5))
time_fit_seq=time_model.fittedvalues
resid_seq=time_model.resid


diff_shift_ts = purchase_seq_train
diff_recover_1 = diff_shift_ts.add(time_fit_seq)
time_predict_seq=time_model.predict(start='2014-8-01',end='2014-08-30',dynamic=True)

#模型检验，画出残差qq图
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid_seq, line='q', ax=ax, fit=True)
plt.show()
#进行D-W检验
print(sm.stats.durbin_watson(resid_seq.values))

#进行Ljung-Box检验,观察Q值
r,q,p=sm.tsa.acf(resid_seq.values.squeeze(),qstat=True)
d=np.c_[range(1,41),r[1:],q,p]
table=pd.DataFrame(d,columns=['lag','AC','Q','Prob(>Q)'])
print(table.set_index('lag'))

#拟合训练集
fig=plt.figure(figsize=(12,5))
plt.plot(diff_recover_1,color='red',label='fit_seq')
plt.plot(purchase_seq_train,color='blue',label='purchase_seq_train')
#plt.title('RMSE: %.4f'% np.sqrt(sum((diff_recover_1-purchase_seq_train)**2)/diff_recover_1.size))
plt.legend(loc='best')
plt.show()

#预测测试集
predict_dates=pd.Series(['2014-08-01','2014-08-02','2014-08-03','2014-08-04','2014-08-05','2014-08-06','2014-08-07','2014-08-08',
                         '2014-08-09','2014-08-10','2014-08-11','2014-08-12','2014-08-13','2014-08-14','2014-08-15',
                         '2014-08-16','2014-08-17','2014-08-18','2014-08-19','2014-08-20','2014-08-21','2014-08-22',
                         '2014-08-23','2014-08-24','2014-08-25','2014-08-26','2014-08-27','2014-08-28','2014-08-29','2014-08-30']).apply(lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d')  )
time_predict_seq.index=predict_dates
predict_diff_seq=pd.Series(time_predict_seq,index=time_predict_seq.index)

#拟合测试集
diff_shift_ts = purchase_seq_test
predict_seq = diff_shift_ts.add(predict_diff_seq,fill_value=0)
#predict_seq.to_csv('C:\\Users\\Administrator\\Desktop\\Purchase Redemption Data\\redeem.csv')

fig=plt.figure(figsize=(12,5))
plt.plot(predict_seq,color='red',label='预测序列')
plt.plot(purchase_seq_test,color='blue',label='赎回测试集')
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title('RMSE: %.4f'% np.sqrt(sum(((purchase_seq_test-predict_seq)/pow(10,8))**2)/predict_seq.size))
plt.legend(loc='best')
plt.show()
