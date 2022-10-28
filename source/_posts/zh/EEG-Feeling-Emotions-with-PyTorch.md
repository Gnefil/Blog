---
title: EEG Feeling Emotions with PyTorch
catalog: true
lang: zh
date: 2022-10-26 12:24:38
subtitle: 
header-img: https://raw.githubusercontent.com/Gnefil/Blog/main/img/post_images/eeg_emotions_bg.jpg
tags: [EEG, PyTorch, 机器学习]
categories: [EEG]
---

# 目标
今天，我们将构建一个机器学习模型，它接受 EEG（ElectroEncephaloGram）信号，并决定这些信号的来源是正面情绪（快乐）还是负面情绪（悲伤）。

## 数据来源
来自 [Kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions) 的 EEG 数据和完整的笔记本可以在 [这里](https://colab.research.google.com/drive/1uHlS2GPhqjeEn1hAN_8pgxZpO9LQr5Nw?usp=sharing)找到。


# EEG 感知情绪

<a href="https://colab.research.google.com/drive/1uHlS2GPhqjeEn1hAN_8pgxZpO9LQr5Nw?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 1. 环境
---
在动手之前，请准备好**工具**🧰！

### 1.1 导入
要导入我们需要的工具（库），请运行以下命令。

```py
# Import libraries
import torch, pandas, numpy, os, requests, zipfile, io, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
```

### 1.2 开启 GPU 
使用GPU（让代码运行更快），左上角（如果使用Colab）：
- 点击*Edit*
- 点击*Notebook settings*
- 从下拉菜单选择*Hardware accelerator*
- 选择**GPU**
- 保存

准备好了吗？ 通过运行以下代码检查 GPU 是否准备就绪。 输出应该是 `Using cuda.`，通过它使用 GPU。

```py
# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}.")
```

### 1.3 定位

去哪里作业呢？让我们创建一个名为 `EEG` 的文件夹，然后在其中构建所有内容。 结果应该是`/content/EEG 文件夹`。

```sh
# Create and get to `EEG` folder
! mkdir -p /content/EEG
%cd /content/EEG/
```

## 2. 数据
---
调出数据✨，让我们看看它有几斤几两！

### 2.1 下载数据
结果数据存储在 `/content/EEG/data/emotions.csv`.

```sh
# Download the eeg data
! mkdir -p ./data
! wget -q -O ./data/emotions.csv https://github.com/UOMDSS/workshops-2022-2023/raw/main/semester-1/Week-7-EEG-Feeling-Emotions-Logistic-Regression/data/emotions.csv
! ls

```

### 2.2 读取数据
现在我们将数据保存在一个文件夹中，然后将其读入我们的笔记本环境。

```py
# Read csv file
data = pandas.read_csv("./data/emotions.csv")
data
```

我们看到 *mean*、一些省略的列、*fft*、*label*。
本笔记本对数据进行了浅层探索，为了更好地理解每一行和每一列的含义，请参阅数据集本身地文档。
例如，其实末尾的“a”和“b”是指数据的来源是来自实验人员“a”还是实验人员“b”。

### 2.3 理解数据
这个数据集里面有什么？ 维度？ 类型？ 哪些数据是有用的？

```py
# Print data about the data
print(data.shape)
print(data.columns)
# for (i, col) in enumerate(data.columns):
#   print(i, col)
print(data.describe())
```

> ***旁注***：fft 代表快速傅立叶变换。 该数据使我们能够将一种*波*图（时域信号，即时间与幅度）表示为另一种*波*图（频率与幅度）。  
> 这里的要点是**我们可以将其可视化！**

### 2.4 数据可视化
FFT 列代表处理过的信号，因此，它使得我们可以将大脑信号绘制为图表。

```py
# Extract fft data
# ranges is tweakable
start = 0 # min = 0
end = 749 # max = 749
ranges = [(f"fft_{start}_a", f"fft_{end}_a"), ("label", "label")]
fft_data = pandas.concat([data.loc[:, i:j] for i, j in ranges], axis = 1) [data["label"] != "NEUTRAL"].reset_index()
fft_data
```

现在数据大小已减少到：
- 1416 行
- 752 列
为什么？

```py
# Plot fft brain signals
# moment (row) is tweakable
moment = 0
fft_data.iloc[moment, 1:-1].plot(figsize=(20, 15), label="Fast Fourier Transform at interval 0s to 1s")
plt.legend()
```

```py
# Brain signals over time
fig = plt.figure(figsize=(25, 15))
for i in range(0, 6):
    plt.subplot(3, 2, i+1)
    plt.plot(fft_data.iloc[i, 1:-1])
    plt.title("Fast Fourier Transform on " + fft_data.loc[i, "label"] + " emotion")
```

单单用肉眼看，你能找到正面或负面情绪的大脑信号之间的共同点吗？

## 3. 创造学习模型
---

现在是时候编写**逻辑回归**模型了​⚒️！

> ***旁注***：我们可以根据需要编写模型一行一行把代码写出来。 但大多数时候，我们倾向于使用*框架*，一个已经准备好的代码库，已经有模型的骨架。 它们方便且易于使用。  
> 这个 notebook 使用了 [**PyTorch**](https://pytorch.org/)，一种这样的框架。

### 3.1 Dataset 类
如果我们想要输入的数据集可以以任何形式出现，那 PyTorch 该如何处理每种情况？
答案是它并**不**处理，我们作为用户负责根据 PyTorch **数据集接口**来处理数据。
以下代码超出了本次研讨会的范围。 只需运行它就ok了，但如果您有兴趣可以看看。

```py
# Define emotion dataset in PyTorch
class EmotionDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.data = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data.iloc[index, :-1].values.astype(numpy.float64)
        y = self.data.iloc[index, -1]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
```

### 3.2 Model 类
我们的目标是用**逻辑回归**来预测情绪，不是吗？  
幸运的是，PyTorch 能够帮助您以难以置信的速度编写这个模型。

> ***旁注***：  
> - `torch.nn.Linear(x, y)` 在回归线中将 `x` 映射到 `y`。
> - `torch.sigmoid(x)` 在应用 sigmoid 函数后计算 x 的值。


```py
# Build Logistic Regression model
class EmotionLogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(EmotionLogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
```

## 4. 训练模型
---
最激烈的部分（对计算机来说）来了！ 我们将训练🏋模型。

### 4.1. 超参数
与训练后获得的“参数”不同，“超参数”是我们人类选择且对全局模型造成影响的值。  
例如，选择 `cuda`、`GPU` 作为我们的硬件资源，您可以将其视为一种“超参数”。

```py
# Choose hyperparameters
fft_start = 0
fft_end = 20
ranges = [(f"fft_{fft_start}_a", f"fft_{fft_end}_a"), (f"fft_{fft_start}_b", f"fft_{fft_end}_b"), ("label", "label")]
train_proportion = 0.8 
batch_size = 64
learning_rate = 0.01 
epochs = 50
```

### 4.2 训练循环
训练模型之前的最后一件事是将所有训练周期和测试周期包装为函数。 这为我们提供了更简洁易读的代码。  
你不需要完全理解里面发生了什么，因为它有一些复杂的概念。 但是，如果您想知道，请随时询问。

```py
# Create train cycle
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

```py
# Create test cycle
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {100*(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")
```

### 4.3 训练
最后！ 以下代码初始化模型并训练模型。 学习数据可能需要一些时间，请耐心等待。 同样，有些功能是 PyTorch 特有的，你不需要完全理解它们。 猜一猜，它们是什么意思？

```py
# Initialise
# Create dataset
data = pandas.read_csv("eeg_data.csv")
data = pandas.concat([data.loc[:, i:j] for i, j in ranges], axis=1)[data["label"] != "NEUTRAL"]
input_size = data.shape[1] - 1
output_size = 1
dataset = EmotionDataset(data, transform=lambda x: torch.from_numpy(x).float(), target_transform=lambda x: torch.tensor([0]).float() if x == "POSITIVE" else torch.tensor([1]).float())
train_size = int(train_proportion * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create model
model = EmotionLogisticRegressionModel(input_size, output_size).to(device)

# Loss and optimizer
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

```

```py
# Train model!
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)
print("Done!")
```

# 结语
今天我们用 PyTorch 建立了一个逻辑回归模型，根据一个人的 EEG 来预测情绪。 如果输入合成的EEG数据会有点抽象，因此这里不会有预测步骤。

这篇文章的原始格式是在一个 Google Colab 笔记本中，这篇文章的风格有别于之前的博文。