# 图像分析实验全集

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)  
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)  
[![OpenCV 4.6](https://img.shields.io/badge/OpenCV-4.6-green.svg)](https://opencv.org/)

---

## 项目概览

本仓库包含了 **6 个图像分析实验**，展示了在不同场景下如何运用深度学习（主要是卷积神经网络 CNN）进行图像分类。  
实验内容涵盖了数据预处理、模型构建、训练、评估以及最终预测的完整流程。  
其中，第一个实验为 **水质检测实验**；  
第二个实验为 **德国交通标志分类**；  
其余实验可视具体应用场景进行扩展与修改。

---

## 实验目录

1. [图像分析实验1 - 水质检测实验](#图像分析实验1---水质检测实验)
2. [图像分析实验2 - 德国交通标志分类](#图像分析实验2---德国交通标志分类)
3. [图像分析实验3](#图像分析实验3)
4. [图像分析实验4](#图像分析实验4)
5. [图像分析实验5](#图像分析实验5)
6. [图像分析实验6](#图像分析实验6)

---

## 图像分析实验1 - 水质检测实验

### 项目背景

数据集中包含一个文件夹存放所有的水质图片，每张图片的文件名以一个数字开头，该数字代表水质类别，其后随图片编号（每个类别约 50 张图像）。  
实验主要目标为：  
- 对图像进行预处理（统一调整为 128×128 的大小，并依据文件名确定类别标签）；  
- 构建并训练一个 CNN 模型对水质图片进行分类；  
- 按 80%/20% 划分数据集作为训练集和测试集，对模型性能进行评价；  
- 对新的测试图片进行预测，并直观展示预测结果与实际类别。

### 实验流程

1. **数据预处理**  
   - 利用 Python 的 `os` 模块遍历数据文件夹中的所有图片；  
   - 使用 OpenCV 读取图片并将其统一调整为 128×128；  
   - 根据文件名的首字符确定图片所属类别；  
   - 随机打乱数据后，按 80% 作为训练集、20% 作为测试集进行划分。

2. **模型构建与训练**  
   - 构建 CNN 模型，其主要结构包括：  
     - **输入层**：形状为 (128, 128, 3)（RGB 三通道）；  
     - **卷积层**：采用 32 个 3×3 卷积核，使用 ReLU 激活；  
     - **池化层**：使用 2×2 的最大池化，减少特征图尺寸；  
     - **展平层**：将二维特征图展平成一维向量；  
     - **全连接层**：首先一层包含 128 个神经元（ReLU 激活），接着输出层包含 5 个神经元（对应 5 个类别，采用 softmax 激活）。  
   - 编译模型时采用 `sparse_categorical_crossentropy` 作为损失函数、Adam 优化器以及 accuracy 指标；  
   - 采用 batch size 为 16，训练 10 个 Epoch。

3. **模型评估与预测**  
   - 在测试集上评估模型，并利用 Matplotlib 绘制训练与验证准确率的变化曲线；  
   - 保存训练好的模型，并加载后对新的测试图片进行预测；  
   - 最后展示测试图片及其预测结果（标题中标明实际类别与预测类别）。

### 实验结果

1. **模型训练结果**  
   
   ![训练结果](https://github.com/user-attachments/assets/94867d31-9369-4fd8-a486-c2352c5921bc)
   
2. **模型测试结果**

   ![测试结果1](https://github.com/user-attachments/assets/8967f24f-da60-4bd0-b631-5f86f741a0e2)  
   ![测试结果2](https://github.com/user-attachments/assets/931b96ed-a0aa-431f-970d-61a7f1dcd139)

---

## 图像分析实验2 - 德国交通标志分类

### 项目背景

该实验的数据集仅提供训练集，共划分为 43 个文件夹（编号从 0 到 42），每个文件夹内存放着对应类别的德国交通标志图片，且图片尺寸不一。  
鉴于数据集的特点，需要对数据进行划分和预处理，以便于后续模型的训练和评估。

### 实验流程

1. **数据预处理**  
   - 使用 Python 的 OS 库遍历每个文件夹中的所有图片；  
   - 利用 OpenCV 读取图片，并调整为统一的 32×32 大小；  
   - 利用 NumPy 对图片数据进行归一化处理，并将文件夹名称（代表类别编号）作为标签；  
   - 将数据集按照 80% 作为训练集、20% 作为测试集进行划分，并随机打乱数据顺序，提高模型训练的鲁棒性。

2. **模型构建与训练**  
   - 构建 CNN 模型，主要结构如下：  
     - **输入层**：设定输入尺寸为 32×32，具有 RGB 三个通道；  
     - **卷积层**：采用 32 个 3×3 卷积核，提取图像局部特征；  
     - **池化层**：采用大小为 2×2 的最大池化窗口，降低特征图维度；  
     - **展平层**：将二维特征图展平成一维向量；  
     - **全连接层**：第一层包含 128 个神经元，第二层包含 43 个神经元，输出 43 个类别的概率分布。  
   - 模型编译时使用交叉熵损失函数，Adam 优化器，并以 accuracy 作为模型评估指标；  
   - 设置 batch size 为 32，训练 10 个 Epoch。

3. **模型评估与预测**  
   - 在测试集上评估模型性能；  
   - 使用 PTL 库（或其他图像处理工具）读取并预处理测试图片，确保其大小为 32×32；  
   - 将预处理后的测试图片输入模型进行预测，并输出预测类别。

### 实验结果
1. **模型训练结果**  

<img width="683" alt="11" src="https://github.com/user-attachments/assets/d61aa9dd-6cae-46e0-b42f-eb4d9464b949" />


   
2. **模型测试结果**

<img width="427" alt="111" src="https://github.com/user-attachments/assets/3c356037-d848-4d40-9e3f-da36312f75c3" />


---

## 图像分析实验3 - 人脸识别实验

### 项目背景

本实验利用 Dlib 的预训练模型，实现对输入图像中人脸的检测与标注。  
主要流程包括读取彩色图像后转换为灰度图、检测人脸区域，并使用形状预测模型获取每张人脸的68个面部特征点，然后在图像上绘制这些特征点进行展示。

### 实验流程

1. **读取并转换图像**  
   - 使用 OpenCV 读取图像文件，并转换为灰度图像。

2. **创建检测器与形状预测器**  
   - 初始化 Dlib 的前脸检测器；  
   - 加载 Dlib 的预训练形状预测模型（用于检测人脸的68个特征点）。

3. **检测人脸并标注特征点**  
   - 在灰度图像中使用 `detector(gray, 1)` 检测所有人脸，返回每个人脸的矩形区域列表；  
   - 对于每个检测到的人脸区域，利用形状预测器获取68个特征点；  
   - 遍历每个特征点，打印其坐标，并在原图上绘制小圆圈标注特征点。

4. **显示结果**  
   - 使用 OpenCV 显示带有标注特征点的图像窗口；  
   - 通过 `cv2.waitKey(0)` 等待用户按键，随后调用 `cv2.destroyAllWindows()` 关闭



### 实验结果
![image](https://github.com/user-attachments/assets/786ed015-b9de-4740-ba5c-d47f29605138)
![image](https://github.com/user-attachments/assets/efa0394d-4776-4c85-9277-34662b6cb0d8)

## 图像分析实验4 - 番茄数据集实验

### 项目背景

本实验针对番茄叶病图像数据集展开。数据集共包含 8 个文件夹，每个文件夹存放一种番茄叶病对应的图像，图像大小各异。由于数据集特点，必须对所有图像进行统一预处理，以便后续训练能更好地捕获特征。值得注意的是，之前在使用 80% 作为训练集的方案下（并添加了图片翻转、调整学习率、更换优化器、损失函数以及调整模型结构等调参操作）时，实验准确率仅在 40%~50% 左右。

### 实验总体思路

整个流程包括以下三个步骤：
1. **数据预处理**  
2. **训练模型与评估**  
3. **读取测试图片并预测**

### 具体方法

1. **数据预处理**  
   - 使用 Python 的 OS 库遍历每个文件夹，打开其中的每张图片；  
   - 利用 OpenCV 将每张图片调整为统一尺寸 32×32；  
   - 使用 NumPy 对图片数据进行归一化处理；  
   - 以文件夹名称（代表类别编号）作为图片标签；  
   - 将整个数据集随机打乱后，按 80%（或其它比例，视具体调整情况）作为训练集、20% 作为测试集。  
     > 注：之前的多次调参尝试中采用了 80% 训练集方案，但效果不理想。

2. **训练模型与评估**  
   - 构建 CNN 模型，主要结构如下：  
     - **输入层**：尺寸设定为 32×32，RGB 三通道；  
     - **卷积层**：使用 32 个 3×3 的卷积核，提取局部特征；  
     - **池化层**：采用 2×2 的最大池化窗口，降低空间维度、减少参数；  
     - **展平层**：将二维特征图展平成一维向量；  
     - **全连接层**：第一层包含 128 个神经元；  
     - **输出层**：第二个全连接层设定为 8 个神经元，对应 8 个输出类别。  
   - 模型编译时使用交叉熵损失函数、Adam 优化器，并以 accuracy（准确率）作为评估指标；  
   - 训练时设定 batch size 为 32，并进行 10 个 Epoch 的迭代。

3. **读取测试图片，进行预测**  
   - 使用 PTL 或 OpenCV 库读取测试图片，并将其调整为 32×32 的尺寸；  
   - 对图像数据进行归一化处理后，输入训练好的模型进行类别预测；  
   - 最终输出预测结果，通过展示图像及其预测标签直观展示模型表现。

### 实验结果
1. **模型训练结果**

<img width="652" alt="11" src="https://github.com/user-attachments/assets/8b3afd33-cbc1-46f3-ae2f-0195de431521" />

2. **模型测试结果**

<img width="406" alt="111" src="https://github.com/user-attachments/assets/10a4dee7-8a14-4c22-afe1-b4edd79ce273" />

---


## 图像分析实验5 - 车辆数据集实验

### 项目背景

该实验针对车辆图像数据集展开。数据集共包含 10 个文件夹，每个文件夹存放着不同类别的车辆图像，且图像尺寸各异。实验的主要目标是对车辆图像数据进行统一预处理，同时利用 PyTorch 构建 CNN 模型对不同车辆类别进行分类。

### 实验总体思路

整个实验流程包括以下步骤：
1. **数据预处理**：统一调整图像尺寸、增加数据增强策略，并划分训练集与测试集；
2. **模型训练与评估**：构建基于 PyTorch 的 CNN 模型，完成模型训练和评估任务；
3. **读取测试图片并进行预测**：利用预处理后的模型对新的测试图像进行预测，并展示结果。

### 具体方法

1. **数据预处理**
   - 首先将所有图像统一调整为 72×72 的尺寸；
   - 对调整后的图像进行随机裁剪，将其裁剪为 64×64 的大小；
   - 同时添加水平翻转等数据增强操作，以增加训练数据的多样性；
   - 将预处理后的数据划分为训练集和测试集（例如，可采用 80%/20% 的比例），并随机打乱顺序，确保数据分布均匀。

2. **训练模型与评估**
   - **模型架构**：  
     - **第一卷积层**：输入通道数为 3（RGB 图像），输出通道数为 64，卷积核大小为 3×3；  
     - **第二卷积层**：输入通道数为 64，输出通道数为 128，卷积核大小为 3×3；  
     - **池化层**：应用大小为 2×2 的最大池化，降低特征图的尺寸；  
     - **Dropout 层**：在全连接层前后均添加 dropout 层以减少过拟合；  
     - **全连接层**：第一层全连接层的输入特征数根据预处理后图像与卷积操作结果确定（此处设定为 64×64×128），后接一个全连接层（输入特征数 512，输出 10 个神经元，对应 10 个分类）。
   - **模型编译**：  
     - 使用 CrossEntropyLoss 作为损失函数；  
     - 采用 Adam 优化器，设置学习率为 0.00018；  
     - 以 accuracy 作为模型在训练和测试阶段的评估指标。
   - **训练过程**：  
     - 使用 batch size 为 32，训练模型共迭代 30 个 Epoch。

3. **读取测试图片，进行预测**
   - 使用 Keras 的图像处理库（例如 `keras.preprocessing.image`）读取测试图片，并将其调整为 72×72 的尺寸，与训练过程预处理保持一致；
   - 对测试图像进行必要的归一化预处理后，输入预先训练好的模型进行类别预测；
   - 输出预测结果，并可将预测标签与原图一起展示，以直观比较分类效果。

### 实验结果
1. **模型训练结果**

<img width="237" alt="11" src="https://github.com/user-attachments/assets/a2ce9a89-be19-4f0f-b120-be29af837746" />


2. **模型测试结果**

<img width="406" alt="111" src="https://github.com/user-attachments/assets/e1e6ecdf-04a0-4756-9219-97d131a4027d" />



---

