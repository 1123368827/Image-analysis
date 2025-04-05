import torch.nn as nn
import torchvision.models as models
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as tnn
# 加载预训练的VGG16模型
model = models.vgg16(pretrained=True)

# 修改模型结构
classifier = list(model.classifier.children())
num_features = model.classifier[-1].in_features
classifier[-1] = nn.Linear(num_features, 200)

# 更新VGG16模型的分类器部分
model.classifier = nn.Sequential(*classifier)

# 打印修改后的VGG16模型结构
print(model)

BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 8
N_CLASSES = 200

transform = transforms.Compose([    # 数据转换
    transforms.Resize((224, 224)),  
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    #                      std  = [ 0.229, 0.224, 0.225 ]),
    ])

trainData = dsets.ImageFolder("D:\\PycharmProjects\\deep_learning\\Image analysis\\data_vgg\\train", transform)   # 自动加载数据并标注类别
testData = dsets.ImageFolder("D:\\PycharmProjects\\deep_learning\\Image analysis\\data_vgg\\valid", transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


model.to('cuda')
cost = tnn.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# for epoch in range(EPOCH):
#     for inputs, labels in trainLoader:
#         inputs, labels = inputs.to('cuda'), labels.to('cuda')
    
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#     print(epoch)

for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    for inputs, labels in trainLoader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss/cnt))
        loss.backward()
        optimizer.step()
    scheduler.step(avg_loss)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testLoader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on test set: {accuracy}')


## 对单张图片进行预测
from PIL import Image
transform = transforms.Compose([    # 数据转换
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

image_name = "D:\\PycharmProjects\\deep_learning\\Image analysis\\data_vgg\\valid\\048.European_Goldfinch\European_Goldfinch_0020_794644.jpg"
image = Image.open(image_name)
image = transform(image)
image = image.unsqueeze(0)
print(image.shape)
image = image.cuda()
outputs = model(image)
_, predicted = torch.max(outputs.data, 1)
print(predicted)

##保存模型
torch.save(model.state_dict(), 'vgg16.pth')
