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

第四次調整 v4
再次卡住，推測是OOM(out of memory)
嘗試解決方法: 降低batch size，因為loss已從5.4 -> 5.0

第五次調整 v5
XLA 為了追求極致速度，有時候會把算式融合得「太過激進」。在某些筆電版 GPU 或特定的 NVIDIA 驅動程式版本上，連續執行幾萬次這種激進的融合指令後，可能會觸發記憶體越界或運算錯誤，導致 GPU 崩潰。
解決方案（關閉 XLA 超頻）tf.config.optimizer.set_jit(False)

第六次調整 v6
多次訓練結果acc都在1%徘徊、推測為原版程式只讀了 plaintext 和 key，完全忽略 ASCADv2 的 Shuffling 問題。Shuffling 的意思是：每筆 trace 中，16 個 byte 的計算順序是隨機打亂的。所以 plaintext[0] 對應的波形特徵，在不同 trace 裡出現在完全不同的時間點。模型學到的只是雜訊，label 和 trace 根本對不上，這才是 acc ≈ 0.39% 的真正原因，跟模型深淺無關。
本次改動主要增強label
.h5 有 permutation + masks  →  策略 C（最準確）
.h5 只有 masks              →  策略 B（次佳）
.h5 什麼都沒有              →  策略 A（baseline，但效果差）
新增_inspect_metadata_keys()事先查看h5欄位來讓程式自動選擇label
差別如下
項目         |  原版                             | 新版
Conv        |  Blocks4 層，filters 最大 256      | 3 層，最大 128Pool size44（前兩層）
分類前層     |  Flatten → 大量參數                | GlobalAveragePooling → 少 10x 
參數激活函數 |  ReLU                             | Swish（對微弱洩漏訊號更平滑）
正則化       | Dropout only                      | Dropout + L2
batch_size  |  128                              |  64（防 OOM）
參數量       |  ~2M                              |  ~200K

第七次調整 v7
問題及修改
1. 標籤與攻擊邏輯的數學脫節
    # 訓練目標：
    vals = sbox_out ^ beta_mask
    return vals, to_categorical(vals, num_classes=256)
    # 攻擊推導邏輯：
    sbox_vals = AES_SBOX[key_hypotheses ^ pt_byte]
    log_scores += np.log(pred_probs[i, sbox_vals] + eps)
    在攻擊階段（Attack Phase），我們是不知道遮罩 $M$ 是多少的。recover_key 函式假設模型輸出的是「乾淨的 $Sbox$ 機率」，但模型實際輸出的是「$Sbox \oplus M$ 的機率」。
    拿著「乾淨 Sbox 的索引」去查「混淆後的機率表」，查出來的值完全是錯位的。
解 -> 強制使用乾淨的S-Box，因為模型目前為單頭的模型，沒辦法分開學習 mask 跟 masked-key 

2. GlobalAveragePooling1D 摧毀了物理時間線
    ASCADv2 的波形有 100,000 個採樣點。經過 3 次 pool_size=2 的縮減後，時間軸還有大約 $12,500$ 個點。GlobalAveragePooling1D 的物理意義是：「把這 12,500 個不同時間點的電壓值，全部加起來除以 12,500，變成一個平均數」。AES 加密的洩漏特徵只發生在特定的幾十個奈秒內（形成幾個微小的 Peak）。你把它跟其他幾萬個冷氣雜訊、記憶體載入雜訊全部「平均」掉，特徵瞬間被稀釋到連神仙都找不到。
解 -> 拿掉

3. 模型太輕
    模型降級到了 3 層 Conv，最大 filter 只有 128，所以結果嚴重欠缺擬合
解 -> 多一層捲曲並且使用 Flatten()，將基礎資料量從 10000 -> 30000

修改輸出圖表格式以及位置

第八次調整 v8
