import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import os            # os包集成了一些对文件路径和目录进行操作的类
import matplotlib.pyplot as plt
import time
from PIL import Image
""""实验记录
添加了水平和垂直翻转，没有什么用
修改了学习率，增大了一些（在0.0002-0.0003左右不错） 0.001就过大了
修改了输入维度，减少了模型的复杂度，没什么用

将批次从500调整到400，准确率从60多--70&
进行了裁剪之后，准确率突破70%，到达75&

"""
## 读取数据
data_dir = 'D:\PycharmProjects\deep_learning\Image analysis\car'
data_transform = {
    x: transforms.Compose([
        transforms.Resize([72, 72]),  # 将图像调整为 64x64 大小
        transforms.RandomCrop([64, 64]),  # 随机裁剪为 64x64 大小
        transforms.RandomHorizontalFlip(),  # 添加水平翻转


        transforms.ToTensor()
    ]) 
    for x in ['train', 'valid']
}

###  上面的代码和想的代码相同
# data_transform = {
#     'train': transforms.Compose([
#         transforms.Resize([64, 64]),  # 将图像调整为 64x64 大小
#         transforms.ToTensor(),  # 将图像转换为张量
#     ]),
#     'valid': transforms.Compose([
#         transforms.Resize([64, 64]),
#         transforms.ToTensor(),
#     ])
# }
image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                        transform = data_transform[x]) for x in ['train', 'valid']}  # 这一步相当于读取数据
dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                           batch_size = 4,
                                           shuffle = True) for x in ['train', 'valid']}  # 读取完数据后，对数据进行装载

class Models(torch.nn.Module):
    def __init__(self):
        super(Models, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

)
        self.dropout = torch.nn.Dropout(p=0.5)  # 添加 dropout 层
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(32*32 * 128, 512), 
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 10))

        
# ## forward 前向传播过程
# #得到输入，自动放到Conv部分
# #view函数展平（-1表示自动推断大小）
# #输入放入Classes部分
    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 32 * 32 * 128)
        x = self.Classes(x)
        return x

model = Models()
# print(model)




loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00018)

Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

epoch_n = 30
time_open = time.time()


## 外层循环，每个周期epoch
for epoch in range(epoch_n):
    print('epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)
    # 对于每个阶段phase，有两个可能的值：'train' 表示训练阶段，'valid' 表示验证阶段。
    for phase in ['train', 'valid']:
        if phase == 'train':
            # # 设置为True，会进行Dropout并使用batch mean和batch var
            print('training...')
            model.train(True)
        else:
            # # 设置为False，不会进行Dropout并使用running mean和running var
            print('validing...')
            model.train(False)

        running_loss = 0.0
        running_corrects = 0.0
        # 把每个批次的图片和标号（按顺序从1开始）对应   x为图片，y为标签
        for batch, data in enumerate(dataloader[phase], 1):
            X, Y = data
            # 将数据放在GPU上训练
            X, Y = Variable(X).cuda(), Variable(Y).cuda()
            # 模型预测概率
            y_pred = model(X)
            # pred，概率较大值对应的索引值，可看做预测结果，1表示行
            _, pred = torch.max(y_pred.data, 1)
            # 梯度归零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_f(y_pred, Y)
            # 训练 需要反向传播及梯度更新
            if phase == 'train':
                # 反向传播出现问题
                loss.backward()
                optimizer.step()
            # 损失和
            running_loss += loss.data.item()
            # 预测正确的图片个数
            running_corrects += torch.sum(pred == Y.data)
            # 训练时，每500个batch输出一次，训练loss和acc
            if batch % 400 == 0 and phase == 'train':
                print('batch{},trainLoss:{:.4f},trainAcc:{:.4f}'.format(batch, running_loss / batch,
                                                                        100 * running_corrects / (4 * batch)))
        # 输出每个epoch的loss和acc
        epoch_loss = running_loss * 4 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])
        print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
time_end = time.time() - time_open
print(time_end)

# torch.save(model, 'model_pytorch_car.pth')
# torch.save(model.state_dict(), 'model_pytorch_car.pth')

# #加载模型并将其移到 GPU 上
# # model = torch.load("D:/PycharmProjects/model_pytorch_car.pth")
# model.load_state_dict(torch.load('model_pytorch_car.pth'))
# model.to("cuda")
# model.eval()

# # # 假设输入数据为一张图片
# input_image_path = 'D:/PycharmProjects/deep_learning/Image analysis/car/test/57ee38ea85a63fb8766d5d9f74aafb71.jpg'
# input_image = Image.open(input_image_path)

# # # 定义数据预处理的转换
# transform = transforms.Compose([
#     transforms.Resize([64, 64]),  
#     transforms.ToTensor(),
# ])

# # 对输入图像进行预处理并移动到 GPU 上
# # input_tensor = transform(input_image).unsqueeze(0).to("cuda")  # 添加一个 batch 维度
# input_tensor=transform(input_image).unsqueeze(0).to("cuda")
# # 使用加载的模型进行预测
# with torch.no_grad():
#     output = model(input_tensor)

# # 获取预测结果的索引
# predicted_class_index = torch.argmax(output, dim=1).item()

# # 获取类别名
# class_names = os.listdir('D:/PycharmProjects/deep_learning/Image analysis/car/train')
# predicted_class_name = class_names[predicted_class_index]


# # 打印预测结果
# print("Predicted class name:", predicted_class_name)

# model = TheModelClass(*args, **kwargs) # type: ignore
# model.load_state_dict(torch.load("D:/PycharmProjects/model_pytorch_car.pth"))
# model.eval()

