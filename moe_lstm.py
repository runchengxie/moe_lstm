'''
本脚本展示如何构建并评估一个混合专家（Mixture-of-Experts, MoE）结构的 LSTM 模型，用于预测标普 500 指数的短期上涨趋势。
主要流程包括：
1. 数据获取与预处理：读取原始数据、对价格和技术指标进行计算（如移动平均线、RSI、MACD、布林带等）。
2. 特征工程：对多种特征进行聚类（K-Means、SOM）或直接使用手工特征集，选出合适特征输入模型。
3. 模型构建与训练：使用多个 LSTM 专家（Experts）和门控网络（Gating Network）组合来得到最终的预测结果（上涨或下跌）。
4. 回测与评估：根据预测结果，使用简单的多空策略进行回测，计算策略收益率、最大回撤与夏普率等指标，并与基准（Buy & Hold）进行对比。

通过多个专家网络来分别提取不同信息维度，并用门控网络融合它们，可在一定程度上提升对市场走势的捕捉能力。
'''

# ## 明确问题陈述（Problem Statement）
# 
# - 所选择的标的: Standard & Poor 500
# - 预测的时间频率：日频
# - 数据区间和长度：Jan 13, 2020 - Jan 10, 2025
# - 具体的预测目标：下一日涨跌（0代表下跌或者不涨，1代表上涨）


# 载入必要的库

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
!pip install minisom
from minisom import MiniSom
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Lambda, Flatten
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


# 全局参数
LOOKBACK = 5
THRESHOLD = 0.002
N_CLUSTERS = 5  # K-Means聚类数

# 读取数据
url = 'https://raw.githubusercontent.com/runchengxie/quant_practice_3/refs/heads/main/historical_data.csv'
df = pd.read_csv(url)
df.head()

# ## 初步数据查看与基础统计
# 
# 在开始可视化或相关性分析之前，需要先对数据做基本了解，包括：数据结构、缺失值、数值范围、数据时间跨度等。
# 
# 1. 查看数据概况：
# 
#   - df.head()：查看前几行，初步了解表格列名及示例数据。
#   - df.tail()：查看最后几行，确认数据尾部是否有缺失或异常。
#   - df.info()：检测每列的数据类型、非空计数，将帮助你确认是否有大量空值(NaN)或数据类型不符合预期。
#   - df.describe()：查看数值列的计数、均值、标准差、最小值、最大值、分位数等；有助于快速识别是否存在极端数值。
# 
# 2. 缺失值检查：
# 
#   - df.isnull().sum()：列出各列缺失值数量，若有缺失值，需要考虑补全或剔除。
#   - 对于金融时间序列数据，若缺失值过多，可删除相应日期或采用插值(如前值填补 forward fill)。

print("===== 数据前几行 =====")
print(df.head())

print("\n===== 数据信息 =====")
print(df.info())

print("\n===== 描述统计 =====")
print(df.describe())

print("\n===== 缺失值统计 =====")
print(df.isnull().sum())

# ## 时间序列可视化（收盘价走势）
# 
# 为了直观地观察资产价格变化，可以先画出收盘价 (Close) 随时间演变的折线图。这样能够看出整体趋势、极端点，以及数据是否存在跳跃性变化。
# 
# - 绘制收盘价随时间曲线：常用 Matplotlib 或 Seaborn。将 Date 设为 x 轴，Close 设为 y 轴。
# - 如果你对开高低收 (OHLC) 都感兴趣，还可以分别画出多条曲线对比。
# - 观察走势期间是否有断点（无交易日或数据缺失），或者有异常价格对走势产生突出影响。

df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df.sort_values(by="Date", inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Close/Last"], label="Close Price")

# 使用 matplotlib.dates 的Locator 和 Formatter
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # 每6个月显示一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # 刻度格式为 年-月

plt.title("Close Price over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)  # 让标签倾斜，避免重叠
plt.show()

# ## 生成标签：下一日涨跌 (Up/Down)
# 
# - 标签定义：为了避免微小波动的噪声，我们设定当天到下一天的涨幅超过 0.2% 即视为上涨 (Target=1)。
# - 新增特征：使用了常用技术指标（MA, RSI, MACD 等），帮助模型捕捉趋势、波动和超买/超卖信息。
# - 合理性：
#   - 微小涨幅可能无法覆盖手续费和滑点，故设置阈值有助于减少噪声；
#   - 引入多个技术指标提供更丰富的上下文；
#   - 平衡数据有助于模型学到更全面的上涨/下跌模式。

threshold = THRESHOLD

# 1. 计算简单移动平均线 (MA)
df["MA5"] = df["Close/Last"].rolling(window=5).mean()
df["MA10"] = df["Close/Last"].rolling(window=10).mean()

# 2. 计算指数移动平均线 (EMA)
df["EMA5"] = df["Close/Last"].ewm(span=5, adjust=False).mean()
df["EMA10"] = df["Close/Last"].ewm(span=10, adjust=False).mean()

# 3. 计算相对强弱指数 (RSI)
# RSI 常用14天计算，这里简单示例:
window_length = 14
delta = df["Close/Last"].diff()
gain = (delta.clip(lower=0)).abs()
loss = (-delta.clip(upper=0)).abs()

avg_gain = gain.rolling(window=window_length).mean()
avg_loss = loss.rolling(window=window_length).mean()

rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# 4. 布林带 (Bollinger Bands)
# 计算MA20为中轨，2倍标准差为上轨和下轨
df["MA20"] = df["Close/Last"].rolling(window=20).mean()
df["STD20"] = df["Close/Last"].rolling(window=20).std()
df["BB_upper"] = df["MA20"] + 2 * df["STD20"]
df["BB_lower"] = df["MA20"] - 2 * df["STD20"]

# 5. MACD (12, 26, 9)
ema12 = df["Close/Last"].ewm(span=12, adjust=False).mean()
ema26 = df["Close/Last"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema12 - ema26
df["Signal_line"] = df["MACD"].ewm(span=9, adjust=False).mean()

# 6. 其他可加入的特征，如日内波动、收益率等
df["Daily_Return"] = df["Close/Last"].pct_change()
df["Volatility"] = df["Daily_Return"].rolling(window=10).std()

# 注意：上面计算完指标后通常需要丢弃最前面的 NaN 行
df.dropna(inplace=True)

# 计算涨幅：以当日Close对比前一日Close
df["Pct_Change"] = df["Close/Last"].pct_change()

# 用下一日的Pct_Change做数据对齐，所以需要shift(-1)
df["Tomorrow_Change"] = df["Pct_Change"].shift(-1)

# 在生成标签时，如果小于等于0.2% 就标记为 0，否则标记为 1
def label_function(x):
    if x > threshold:
        return 1
    else:
        return 0

df["Target"] = df["Tomorrow_Change"].apply(label_function)

# 去掉无法计算下一日涨幅的最后一行
df.dropna(subset=["Tomorrow_Change"], inplace=True)

# 现在 df 中就有一个新的 Target，体现了「微小涨幅」不计入“上涨”。
print(df[["Date", "Close/Last", "Pct_Change", "Tomorrow_Change", "Target"]].head(10))

# ## 查看类别数量及比例
# 
# - 使用 value_counts() 查看标签列各个值出现的频次，然后进一步计算占比。
# - 若出现数据不平衡，将应用 SMOTE 过采样平衡数据，并在训练过程中观察 balanced accuracy

balance_counts = df['Target'].value_counts()
total_samples = len(df)

print("标签分布：")
print(balance_counts)
print("\n占比：")
print(balance_counts / total_samples)

# ## 准备特征：构造LSTM输入
# - 选定多列特征

manual_feature_cols = [
    "Close/Last",
    "MA5",
    "MA10",
    "EMA5",
    "EMA10",
    "RSI",
    "BB_upper",
    "BB_lower",
    "MACD",
    "Signal_line",
    "Daily_Return",
    "Volatility",
    "Open",
    "High",
    "Low"
]

feature_data = df[manual_feature_cols].values  # shape: (num_samples, num_features)

# ## 多特征可视化与多重共线性分析
# 
# 当你已经构造了 MA、EMA、RSI、MACD 等衍生特征后，建议做以下几点：
# 
# 1. 查看分布：
# 
#   - 可以用直方图(histogram)或核密度图(kdeplot)查看各特征的分布形态：是否偏态分布、是否存在重尾等。
# 
# 2. 特征-特征散点图：
# 
#   - 如果你想直观了解任意两个特征之间的关系，如 (RSI, Daily_Return)、(MACD, Volatility) 等，可用散点图比较。
#   - 将有助于发现“非线性”关系或聚集性。
# 
# 3. 相关矩阵(heatmap)：
# 
#   - 通过 df.corr() 计算各特征之间的相关系数。若有一些特征高度相关(例如相关系数>0.9)，可能要么删掉其中一个，要么通过降维(如聚类、SOM)来做合并。

corr_matrix = df[manual_feature_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Selected Features")
plt.show()

# ## 缩放特征
# 
# - 这里需要对多列一起做缩放，保证不同量纲的特征可以在同一尺度上被LSTM处理

scaler = MinMaxScaler(feature_range=(0, 1))
feature_data_scaled = scaler.fit_transform(feature_data)

# 生成标签（Target），如果你之前已经有 df["Target"] 了，可以这样：
labels = df["Target"].values

# ## K均值聚类进行特征选择
# 
# ### 对特征进行聚类
# 
# 将每个特征视为高维空间中的点，通过K均值分组：

# 转置特征矩阵，形状为 (n_features, n_samples)
X_features = feature_data_scaled.T

# 使用K均值聚类（假设分为5个簇）
n_clusters = N_CLUSTERS
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_features)
cluster_labels = kmeans.labels_

# ### 选择代表特征
# 
# 从每个簇中选择与簇中心最接近的特征：

kmeans_feature_cols = []
for i in range(n_clusters):
    indices = np.where(cluster_labels == i)[0]
    distances = np.linalg.norm(X_features[indices] - kmeans.cluster_centers_[i], axis=1)
    closest_idx = indices[np.argmin(distances)]
    kmeans_feature_cols.append(manual_feature_cols[closest_idx])

print("K-Means Selected Features:", kmeans_feature_cols)

# ## 自组织映射（SOM）进行特征选择
# 
# ### 训练SOM模型
# 
# 使用MiniSom库将特征映射到二维网格：

# 定义SOM网格大小（例如3x3）
som = MiniSom(3, 3, X_features.shape[1], sigma=0.5, learning_rate=0.5, random_seed=42)
som.train_random(X_features, 100)  # 训练100次迭代

# 获取每个特征的BMU（Best Matching Unit）
bmu_indices = [som.winner(f) for f in X_features]

# ### 根据BMU分组并选择特征

# 1. 准备特征数据（每个特征为一个样本，维度为时间序列长度）
X_features = feature_data_scaled.T  # 形状: (n_features, n_samples)

# 2. 训练SOM
som = MiniSom(3, 3, X_features.shape[1], sigma=0.5, learning_rate=0.5, random_seed=42)
som.train_random(X_features, 100)

# 3. 获取每个特征的BMU坐标
bmu_indices = [som.winner(f) for f in X_features]

# 4. 按BMU分组
cluster_map = defaultdict(list)
for idx, bmu in enumerate(bmu_indices):
    cluster_map[bmu].append(idx)

# 5. 选择每个BMU组的代表特征
som_feature_cols = []
for bmu, cluster in cluster_map.items():
    # 获取BMU的权重向量（形状为 (n_samples,)）
    bmu_weights = som.get_weights()[bmu[0], bmu[1], :]

    # 计算距离
    distances = [
        np.linalg.norm(X_features[idx] - bmu_weights)
        for idx in cluster
    ]

    # 选择最近的特征
    closest_idx = cluster[np.argmin(distances)]
    som_feature_cols.append(manual_feature_cols[closest_idx])

print("SOM Selected Features:", som_feature_cols)

# ## 封装模型训练与预测函数
# 
# 将模型训练和预测过程封装为函数，便于复用

def train_and_predict_moe(feature_list, model_name, lookback):
    """训练模型并返回预测结果及分类指标"""
    # 提取特征数据
    feature_data = df[feature_list].values

    # 独立标准化（每个特征集单独处理）
    scaler = MinMaxScaler()
    feature_data_scaled = scaler.fit_transform(feature_data)

    # 构造时序窗口
    X, y = [], []
    for i in range(len(feature_data_scaled) - lookback):
        X.append(feature_data_scaled[i : i + lookback])
        y.append(labels[i + lookback])
    X = np.array(X)
    y = np.array(y)

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # MOE LSTM模型
    input_seq = Input(shape=(lookback, len(feature_list)), name="Input_Sequence")

    # LSTM Experts
    lstm_expert_1 = LSTM(64, return_sequences=True, name="LSTM_Expert_1")(input_seq)
    lstm_expert_2 = LSTM(64, return_sequences=True, name="LSTM_Expert_2")(input_seq)
    lstm_expert_3 = LSTM(64, return_sequences=True, name="LSTM_Expert_3")(input_seq)

    # Flatten LSTM outputs
    flat_expert_1 = Flatten(name="Flatten_Expert_1")(lstm_expert_1)
    flat_expert_2 = Flatten(name="Flatten_Expert_2")(lstm_expert_2)
    flat_expert_3 = Flatten(name="Flatten_Expert_3")(lstm_expert_3)

    # Gating Network
    gating_network = Dense(3, activation="softmax", name="Gating_Network")(input_seq[:, -1, :])

    # Weighted sum of experts
    weighted_sum = Lambda(lambda x: K.sum(x[0] * K.expand_dims(x[1], axis=-1), axis=1), name="Weighted_Sum")([
        Concatenate(name="Concatenate_Experts")([flat_expert_1, flat_expert_2, flat_expert_3]),
        gating_network
    ])

    # Final output layer
    final_output = Dense(1, activation="sigmoid", name="Final_Output")(weighted_sum)

    # Compile model
    moe_model = Model(inputs=input_seq, outputs=final_output, name="MOE_LSTM_Model")
    moe_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # 训练模型
    moe_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    # 生成预测结果
    y_pred_proba = moe_model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 计算分类指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)

    # 输出详细分类报告（可选）
    print(f"\n===== {model_name} 分类报告 =====")
    print(classification_report(y_test, y_pred))

    # 返回结果
    return {
        "model_name": model_name,
        "y_test": y_test,
        "predictions": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "model": moe_model
    }

# ## 封装回测函数
# 
# 统一回测逻辑，计算策略收益和关键指标。

def backtest(predictions, df_test, model_name, fee=0.0005):
    """
    执行策略回测的函数。

    参数:
    ----------
    predictions : array-like
        模型输出的预测序列(0,1)。
    df_test : pandas.DataFrame
        对应测试集数据，包含日期和真实收益率。
    model_name : str
        策略/模型名称，便于区分。
    fee : float, optional
        每次交易的固定费率，默认为0.0005。

    返回:
    ----------
    dict
        包含回测关键指标和累积收益序列的字典。
    """
    # 对齐测试集数据
    df_test = df_test.copy()
    df_test["Prediction"] = predictions

    # 生成持仓信号（1=做多，-1=做空）
    df_test["Position"] = np.where(df_test["Prediction"] == 1, 1, -1)

    # 计算策略收益率（扣除手续费）
    df_test["Strategy_Return"] = df_test["Daily_Return"] * df_test["Position"]
    df_test["Strategy_Return"] = df_test["Strategy_Return"] - fee  # 扣除手续费

    # 计算累计净值
    df_test[f"{model_name}_Cumulative"] = (1 + df_test["Strategy_Return"]).cumprod()

    # 计算最大回撤
    roll_max = df_test[f"{model_name}_Cumulative"].cummax()
    df_test[f"{model_name}_Drawdown"] = (df_test[f"{model_name}_Cumulative"] - roll_max) / roll_max

    # 汇总指标
    total_return = df_test[f"{model_name}_Cumulative"].iloc[-1] - 1
    max_drawdown = df_test[f"{model_name}_Drawdown"].min()
    sharpe_ratio = (df_test["Strategy_Return"].mean() / df_test["Strategy_Return"].std()) * np.sqrt(252)

    return {
        "Model": model_name,
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Cumulative": df_test[f"{model_name}_Cumulative"]
    }

# ## 运行三个模型并回测/收集指标

# 全局参数
LOOKBACK = 5  # 确保此前已定义

# 训练并评估每个模型
results = []
for feature_cols, model_name in zip(
    [manual_feature_cols, kmeans_feature_cols, som_feature_cols],
    ["Manual", "KMeans", "SOM"]
):
    # 训练模型并获取结果
    model_result = train_and_predict(feature_cols, model_name, LOOKBACK)

    # 执行回测
    predictions = model_result["predictions"]

    # 对齐测试集时间索引
    test_start_index = len(df) - len(predictions) - LOOKBACK
    df_test = df.iloc[test_start_index:].copy()
    df_test = df_test.tail(len(predictions))

    # 注意：以下语句要与上面的 for 循环对齐
    backtest_result = backtest(predictions, df_test, model_name)

    combined_result = {
        "Model": model_name,
        "y_test": model_result["y_test"],
        "predictions": model_result["predictions"],
        "Accuracy": model_result["accuracy"],
        "Precision": model_result["precision"],
        "Recall": model_result["recall"],
        "F1": model_result["f1"],
        "AUC-ROC": model_result["auc_roc"],
        "Total Return": backtest_result["Total Return"],
        "Max Drawdown": backtest_result["Max Drawdown"],
        "Sharpe Ratio": backtest_result["Sharpe Ratio"]
    }

    # 这一步很重要，把回测里的净值序列也放进combined_result
    combined_result["Cumulative"] = backtest_result["Cumulative"]

    # 将该模型的所有结果记录到results
    results.append(combined_result)

# 循环结束后，可在这里对results做进一步处理或打印
print(results)

# ## 输出对比表格
# 
# 生成一个综合对比表格，包含分类指标和回测指标：

# 生成对比表格
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df[[
    "Model",
    "Accuracy", "Precision", "Recall", "F1", "AUC-ROC",
    "Total Return", "Max Drawdown", "Sharpe Ratio"
]]

print("\n===== 模型性能对比 =====")
print(metrics_df.round(3))

# ## 可视化混淆矩阵（可选）
# 
# 为每个模型绘制混淆矩阵：

plt.figure(figsize=(15, 4))
for i, result in enumerate(results):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(result["y_test"], result["predictions"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down", "Up"],
                yticklabels=["Down", "Up"])
    plt.title(f"{result['Model']} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
plt.tight_layout()
plt.show()

# ## 可视化对比结果
# 
# 绘制三种模型的净值曲线和回撤曲线。

plt.figure(figsize=(14, 7))

# 绘制净值曲线
for result in results:
    plt.plot(df_test["Date"], result["Cumulative"], label=result["Model"])

# 绘制基准（Buy & Hold）
plt.plot(df_test["Date"], (1 + df_test["Daily_Return"]).cumprod(), label="Buy & Hold", linestyle="--")

plt.title("Cumulative Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# 绘制回撤曲线
plt.figure(figsize=(14, 5))
for result in results:
    plt.plot(df_test["Date"], result["Cumulative"].pct_change(), label=result["Model"])

plt.title("Daily Returns Comparison")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.show()

# ## 输出关键指标
# 
# 生成表格对比三种模型的性能。

# 汇总指标
metrics_df = pd.DataFrame([{
    "Model": result["Model"],
    "Total Return": result["Total Return"],
    "Max Drawdown": result["Max Drawdown"],
    "Sharpe Ratio": result["Sharpe Ratio"]
} for result in results])

print(metrics_df)

# -- Replace previous loop with a single MOE call --
moe_result = train_and_predict_moe(manual_feature_cols, "MOE", LOOKBACK)

test_start_index = len(df) - len(moe_result["predictions"]) - LOOKBACK
df_test_moe = df.iloc[test_start_index:].copy()
df_test_moe = df_test_moe.tail(len(moe_result["predictions"]))

moe_backtest_result = backtest(moe_result["predictions"], df_test_moe, "MOE")
print("\n===== MOE 模型回测指标 =====")
print({
    "Total Return": moe_backtest_result["Total Return"],
    "Max Drawdown": moe_backtest_result["Max Drawdown"],
    "Sharpe Ratio": moe_backtest_result["Sharpe Ratio"]
})


