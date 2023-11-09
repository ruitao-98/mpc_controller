import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_excel("data.xlsx")
# print(type(df1))
# df1 = df1[0:20000,0]
cut = 100000
time_series = df1['clo']
time_series_np = np.array(time_series)
time_series_np1 = time_series_np[0:cut ]
time_series_np2 = time_series_np[cut :]
print(time_series_np.shape[0])



# 获取线性趋势线的系数
coefficients = np.polyfit(np.arange(time_series_np1.shape[0]), time_series_np1, 1)

# 创建趋势线df1.index.values
trendline = np.arange(time_series_np1.shape[0]) * coefficients[0] + coefficients[1]

detrended = (time_series_np1 - trendline)

coefficients2 = np.polyfit(np.arange(time_series_np2.shape[0]), time_series_np2, 1)

# 创建趋势线df1.index.values
trendline2 = np.arange(time_series_np2.shape[0]) * coefficients2[0] + coefficients2[1]

# 从原始序列中减去趋势线
detrended2 = time_series_np2 - trendline2

offset = detrended[-1] - detrended2[0]
detrended2 = detrended2 + offset
index = np.arange(cut , cut + detrended2.shape[0])

plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(time_series_np, label='Original', linewidth = 0.5)
plt.plot(trendline, color='red', label='Trendline')
plt.plot(index, trendline2, color='red', label='Trendline')
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(index, detrended2 , label='After detrend', linewidth = 0.5)
plt.plot(detrended , label='After detrend', linewidth = 0.5)

# plt.plot(trendline , label='baseline', linewidth = 0.5)
plt.legend(loc='best')

plt.savefig("data.png", dpi=600)

plt.tight_layout()
plt.show()

# df2 = df1.diff()

# window = 5

# df1['target'] = df1['clo'].rolling(window).mean()
# df1['detrended'] = df1['clo'] - df1['target']
# df1 = df1.dropna()