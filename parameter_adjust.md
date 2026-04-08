===============================================================================================================================
1. 症狀：欠擬合 (Underfitting)

    現象：Training Loss 降不下來，停在一個很高的數值（例如你的 5.54），模型連訓練集都學不會。

    處方依據：代表「大腦容量不夠」或「不想學」。

    調整動作：增加神經元數量（Filters / Dense）、增加網路層數、降低 Dropout（放開煞車）、或者稍微調高學習率（LR）。

2. 症狀：過擬合 (Overfitting)

    現象：Training Loss 一直往下降（逼近 0），但 Validation Loss 卻在某個時間點開始往上飆高。

    處方依據：代表模型在「死背」資料的雜訊，失去了舉一反三的能力。

    調整動作：調高 Dropout、加入 L2 正則化、或者最有效的——增加訓練資料量。

3. 症狀：震盪不休

    現象：Loss 曲線像心電圖一樣劇烈上下跳動，無法穩定收斂。

    處方依據：代表下山的步伐太大，或每次看的樣本太少。

    調整動作：調低學習率（LR）、調大 Batch Size。
===============================================================================================================================

第一次測試 v1
dropout: 0.4 -> 0.1
learning rate: 5e-4 -> 1e-4

加入資料集來源機構模型(https://github.com/ANSSI-FR/ASCAD)中的ResNet
主要差別如下
資料label
原本的資料讀入只讀取plainttxt & key -> 官方model中使用multitask learning，預先計算預先算好了真正的乘法掩碼 (alpha_mask)、加法掩碼 (beta_mask) 以及洗牌順序

模型
保留 MixedDataGenerator (解決 ShiftRows 問題)，並把裡面的大腦從 4 層 CNN 換成法國官方的 19 層 ResNet (將build_cnn_model函式刪除採用官方的ResNet架構)

預期結果
深度從 4 層暴增到 19 層

對抗洗牌 (Shuffling)：官方這套 ResNet 專門設計來硬扛 ASCADv2 波形洗牌問題的。

第二次調整 v2
再跑少數的Epoch(3次)就發現accuracy提高0.03%，所以將profiling_limit從10000調整到50000 (法國官方使用80w筆)

第三次調整 v3
訓練到第10個卡住後shut down訓練、將profiling_limit調整回10000

第四次調整
再次卡住，推測是OOM(out of memory)
嘗試解決方法: 開啟 Generator 的多核心加速
# 加上下面這兩行，讓 CPU 用多核心幫忙搬運資料
    workers=4,                  
    use_multiprocessing=True