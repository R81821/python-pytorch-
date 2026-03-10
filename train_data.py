import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

#训练数据集
train_dataset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data/cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                             download=True)

#数据集长度
train_dataset_len = len(train_dataset)
test_dataset_len = len(test_dataset)
print("训练集长度：{}".format(train_dataset_len))
print("测试集长度：{}".format(test_dataset_len))

#Dataloader 加载数据
train_dataloader = DataLoader(train_dataset, 64, shuffle=True)
test_dataloader = DataLoader(test_dataset, 64, shuffle=True)

#创建网络模型
tudui = Tudui()
tudui = tudui.cuda()

#损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

#激活函数
relu = nn.ReLU()

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 30

#添加Tensorboard
writer = SummaryWriter("./data_logs")

for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))

    #训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        #优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    #测试步骤开始
    total_test_loss = 0
    #整体正确的个数
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_dataset_len))
    total_test_step += 1
    writer.add_scalar("test_accuracy", total_accuracy / test_dataset_len, total_test_step)
    writer.add_scalar("test_loss", total_test_loss, total_test_step)

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close()