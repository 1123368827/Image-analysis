import json
import os
import cv2
import numpy as np
np.object = object
np.bool = bool
np.int=int
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
## 1.训练模型
# 读取annotations.json文件
def train():
    with open(r"D:\PycharmProjects\deep_learning\Image analysis\homework\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\annotations.json", 'r') as f:
        annotations = json.load(f)

    # 读取图像文件和标注信息
    images_path = r"D:\PycharmProjects\deep_learning\Image analysis\homework\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\images"
    image_files = os.listdir(images_path)

    # 定义类别和类别序号的对应关系
    class_names = {
        0: '背景',
        1: '瓶盖破损',
        2: '瓶盖变形',
        3: '瓶盖坏边',
        4: '瓶盖打旋',
        5: '瓶盖断点',
        6: '标贴歪斜',
        7: '标贴起皱',
        8: '标贴气泡',
        9: '喷码正常',
        10: '喷码异常'
    }

    # 加载图像和标签数据
    images = []
    labels = []

    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        image_file = [file for file in image_files if str(image_id) in file]
        if len(image_file) > 0:
            image_path = os.path.join(images_path, image_file[0])
            image = cv2.imread(image_path)

            # 将标注信息转换为矩形框坐标
            x, y, w, h = bbox
            x2, y2 = x + w, y + h

            # 在图像上绘制矩形框
            cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)

            images.append(cv2.resize(image, (224, 224)))  # 调整图像大小为224x224
            labels.append(category_id)

    # 将数据转换为NumPy数组
    images = np.array(images)
    labels = np.array(labels)

    # 划分训练集和测试集
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 构建卷积神经网络模型
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(11, activation='softmax')  # 11个类别，对应0-10
    ])

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # 训练模型
    history=model.fit(train_images,
                    train_labels,
                    epochs=30,
                    validation_data=(test_images, test_labels))

    # 测试模型Q
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    # 绘制损失函数与准确率曲线
    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    model.save("bottle.h5")


## 2.使用模型进行预测
# 加载模型
def predit():
    model = load_model("bottle.h5")

    def prepare_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # 调整图像大小为 224x224
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转换为RGB
        img = img / 255.0  # 归一化到0到1范围
        return img

    # 假设你有一个图像路径
    image_path = r"D:\PycharmProjects\deep_learning\Image analysis\homework\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\images\img_0001189.jpg"
    image = prepare_image(image_path)
    image = tf.expand_dims(image, axis=0)  # 增加批次维度
    predictions = model.predict(image)
    predicted_class = tf.argmax(predictions, axis=1)  # 获取最高概率的类别索引

    # 显示预测结果
    class_names = {
        0: '背景',
        1: '瓶盖破损',
        2: '瓶盖变形',
        3: '瓶盖坏边',
        4: '瓶盖打旋',
        5: '瓶盖断点',
        6: '标贴歪斜',
        7: '标贴起皱',
        8: '标贴气泡',
        9: '喷码正常',
        10: '喷码异常'
    }
    print("Predicted class:", class_names[predicted_class[0].numpy()])


if __name__=="__main__":
    train()
    predit()