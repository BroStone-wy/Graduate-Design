# graduate-design
## 问题
通过了解余额宝用户的申购和赎回行为数据以及其他用户信息，预测未来一个月每日资金流入流出情况。  
## 数据集介绍
蚂蚁金服提供余额宝三万用户在13个月（2013年7月-2014年8月）的经过脱敏加密处理的完整行为（购买/赎回）数据以及其他数据信息。
## 方法
通过建立ARIMA模型和LSTM神经网络模型，分别对申购和赎回序列未来一个月数据序列预测。
## 概括
实验结果表明，两种模型均可以较好地对资金流入流出金额进行预测，其中LSTM模型在时间序列预测上效果更优。
