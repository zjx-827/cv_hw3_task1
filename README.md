# 代码说明
自监督预训练：python pretain_res18.py

自监督训练：python train_ssl.py

有监督训练:python train.py

此Python代码旨在对比监督学习和自监督学习在图像分类任务上的性能表现。它包括对基线训练、cutout、mixup和cutmix等不同训练方法的支持，以提高模型的鲁棒性和性能。


## 监督学习部分

### 特性

- **ResNet-18架构**:18层的深度残差网络。
- **数据增强**:支持随机裁剪，水平翻转，和正常化。
- **高级训练技术**:
  - 剪切:随机从输入图像中删除一个正方形区域。
  - 混淆:以一定的比例组合两个图像及其标签。
  - Cutmix:一种混合的变种，也混合了图像补丁。
- **日志和可视化**:使用TensorBoard进行可视化和CSV日志跟踪性能指标。
- **检查点**:保存模型检查点，供以后评估或进一步培训使用。

### 要求

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib(用于绘图，可选)
- tqdm(用于进度条)
- csv

您可以使用pip安装所需的包:
```bash
pip install -r requirements.txt
```

### 使用

1. **准备CIFAR-100数据集**:脚本将自动下载并准备CIFAR-100数据集。

2. **配置训练参数**:修改脚本底部的“args”字典，以配置训练参数，如批量大小、学习率和训练方法。

3. **运行脚本**:使用Python执行代码。脚本将训练模型并保存结果。

```bash
python train.py
```

4. **评估模型**:训练结束后，可以使用‘test()’函数评估模型在测试集上的性能。

### 参数

- dataset: 要使用的数据集(在本例中是'cifar100')。
- model:模型架构('resnet18')。
- batch_size:每批样品的数量。
- epoch:要训练的epoch的数量。
- learning_rate:优化器的初始学习率。
- n_holes:要切割的孔数(用于切割)。
- length:要切割的正方形的长度(用于切割)。
- alpha: beta分布在mixup中使用的参数。
- cutmix_prob:使用cutmix的概率。
- cuda:如果cuda可用于GPU加速，则设置为True。

你可以通过运行TensorBoard来可视化训练过程。


保存和加载模型

- 脚本将模型检查点保存在'./ mycheckpoints”目录中。
- 要加载一个模型检查点，使用'torch.load()'和检查点文件的路径。

## 自监督学习部分

### CIFAR-100分类与微调ResNet-18

这个脚本演示了如何使用PyTorch在CIFAR-100数据集上微调ResNet-18模型。模型最初在不同的数据集上进行预训练(如检查点加载所示)，然后在CIFAR-100上重新初始化并训练最终的全连接层。

### 特性

- **微调**:模型的权重从检查点加载，只有最后一层从头开始训练。
- **Data Loading**: CIFAR-100数据集装载了' DataLoader '，以方便批处理。
- **评估指标**:准确性计算前1和前5的预测。
- **可视化**:训练和测试的准确性绘制在不同的时代。

### 要求

- Python 3.x
- PyTorch
- torchvision
- matplotlib(用于绘图)
- CUDA(可选，用于GPU加速)

您可以使用pip安装所需的包:

```bash
pip install -r requirements.txt
```

### 使用

1. **准备CIFAR-100数据集**:如果尚未可用，脚本将自动下载并准备CIFAR-100数据集。

2. **加载预训练模型**:脚本假设在' ./checkpoint/checkpoint_0020.pth.tar '处有一个可用的检查点。根据需要修改路径。

3. **运行脚本**:使用Python执行脚本。该脚本将对模型进行微调，并绘制不同时期的准确性。

```bash
python prertain_res18.py
```

4. **绘图结果**:在训练之后，脚本将显示一个显示训练和测试准确性的绘图。

### 参数

- download:如果找不到数据集，是否下载数据集。
- shuffle:是否在每个epoch shuffle数据。
- batch_size:每批用于训练和测试的样本数量。

### 模型细节

- 模型是一个ResNet-18，最终的分类层被替换，以匹配CIFAR-100的类数量(100类)。
- 模型的权重从检查点加载，但最后一层被重新初始化和训练。

### 训练流程

- 脚本运行指定数量的epoch(默认为100)。
- Adam优化器只用于训练新的最终层，学习率为0.01，重量衰减为0.0008。

### 可视化

- 脚本使用matplotlib绘制top-1和top-5预测的训练和测试准确性。

