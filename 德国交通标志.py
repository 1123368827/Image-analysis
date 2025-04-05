import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from PIL import Image
from keras.models import load_model

# 定义图片文件夹路径
data_dir = r'D:\PycharmProjects\deep_learning\Image analysis\GTSRB\Training'
num_classes = 43
img_size = (32, 32)

# 加载图片数据集
dataset = []
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)   #连接data_dir和folder成为一个路径
    if os.path.isdir(folder_path):
        images = os.listdir(folder_path)[:-1]  # 排除子文件夹下的最后一个文件 是readme.txt
        for image_name in images:
            image_path = os.path.join(folder_path, image_name)
            img = Image.open(image_path)
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # 归一化到 [0, 1]
            dataset.append((img_array, int(folder)))   #img_array是图像的数据，int(folder)是文件夹的名词-图像的类别，将数据集存储为包含图像数组和对应标签的元组列表。

# 打乱数据集顺序  训练效果更好
random.shuffle(dataset)

# 划分训练集和测试集   80训练  20测试
split_idx = int(0.8 * len(dataset))
train_data = dataset[:split_idx]
test_data = dataset[split_idx:]

# 准备训练数据
X_train, y_train = zip(*train_data)   #将图像数据和标签数据分开存储，以便后续的模型训练
X_train = np.array(X_train)
y_train = np.array(y_train)#将训练集和测试集中的图像数据和标签分离并准备为模型训练和评估做准备。

# 准备测试数据
X_test, y_test = zip(*test_data)
X_test = np.array(X_test)
y_test = np.array(y_test)

# # 构建模型
# model = keras.Sequential([
#     keras.Input(shape=(32, 32, 3)),   #大小32*32，3个通道
#     layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),       #特征提取，relu激活函数
#     layers.MaxPooling2D(pool_size=(2, 2)),                         #池化层，减少参数
#     layers.Flatten(),                                               #展平层，展平层一维向量，来连接 全连接层
#     layers.Dense(128, activation='relu'),                           #全连接层
#     layers.Dense(num_classes, activation='softmax')              #soft激活函数
# ])

# # 编译模型
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #计算模型损失的损失函数--交叉熵损失函数  adam-优化器  accuracy准确率来评估性能

# # 训练模型
# model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))  #同时处理32批样本，迭代10次，用测试集来评估性能



# model.save('my_model.keras')   #保存模型





# 使用加载的模型进行预测等操作
model = load_model('my_model.keras')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


predictions = model.predict(X_test)


# 假设 test_image 是一个新的图片数据，需要进行分类
# 先加载和预处理 test_image
test_image = Image.open('D:/PycharmProjects/deep_learning/Image analysis/GTSRB/Training/00014/00000_00013.ppm')
test_image = test_image.resize(img_size)   #调整图像大小为img_size
test_image = np.array(test_image) / 255.0  # 归一化到 [0, 1]
# 调整维度以匹配模型输入--增加一个维度
test_image = np.expand_dims(test_image, axis=0)
# 进行预测
predicted_class = np.argmax(model.predict(test_image), axis=-1)
print("Predicted class:", predicted_class)

# 评估模型在测试集上的准确率++
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')
