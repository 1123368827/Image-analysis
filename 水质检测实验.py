import os
import random
import cv2
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
import cv2
from matplotlib import pyplot as plt
# 定义图片文件夹路径
path = 'D:\\PycharmProjects\\deep_learning\\Image analysis'
num_classes = 5
img_size = (128, 128)

dataset = []
for i in range(1, 6):
    dataset.append(([], i))

for file in os.listdir(path):
    label = int(file[0])
    img = cv2.imread(path + file)
    img_resized = cv2.resize(img, img_size)
    dataset[label - 1][0].append(img_resized)


X_train, Y_train, x_test, y_test = [], [], [], []


for i in range(len(dataset)):
    # 打乱数据集顺序
    data = dataset[i][0]
    # print(data)
    random.shuffle(data)
    # 划分训练集和测试集
    split_idx = int(0.8 * len(data))
    for j in data[:split_idx]:
        X_train.append(j)
        Y_train.append(i)
    for k in data[split_idx:]:
        x_test.append(k)
        y_test.append(i)


X_train = np.array(X_train)
x_test = np.array(x_test)
Y_train = np.array(Y_train)
y_test = np.array(y_test)


# 构建模型
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])

# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test))


# 评估训练模型
scores = model.evaluate(x_test, y_test, verbose=1)
print(scores)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# 保存模型和权重
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_model')
model_name = 'mood_classify.keras'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


model = load_model("D:\\PycharmProjects\\deep_learning\\Image analysis\\saved_model\\mood_classify.keras", compile=False)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(40,40))
img_size = (128, 128)
path = "4_3.jpg"
actual = path[0]
img = cv2.imread("D:\\PycharmProjects\\deep_learning\\Image analysis" + path)
img = cv2.resize(img, img_size)
test_img = np.expand_dims(img, axis=0)

prediction = model.predict(test_img)
max_index = np.argmax(prediction[0])
print(int(max_index) + 1)

plt.imshow(img)
plt.title("actual: " + str(actual) + " || " + "pre: " + str(int(max_index) + 1))
plt.axis("off")
plt.show()