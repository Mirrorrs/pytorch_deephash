import os
import shutil
import argparse

import torch
import torch.nn as nn

from net import AlexNetPlusLatent

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler


# 添加命令行参数
parser = argparse.ArgumentParser(description='Deep Hashing')
# 学习率
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
# 随机梯度下降参数
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# epoch轮数
parser.add_argument('--epoch', type=int, default=128, metavar='epoch',
                    help='epoch')
# 加载训练过的模型
parser.add_argument('--pretrained', type=int, default=0, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
# 论文中的bits参数
parser.add_argument('--bits', type=int, default=48, metavar='bts',
                    help='binary bits')
# 模型路径
parser.add_argument('--path', type=str, default='model', metavar='P',
                    help='path directory')
args = parser.parse_args()


best_acc = 0
start_epoch = 1

# 使用torchvison中的transforms对图像进行变换
# 训练数据
transform_train = transforms.Compose(
    [transforms.Resize(256),
     transforms.RandomCrop(227),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 测试数据做同样变换
transform_test = transforms.Compose(
    [transforms.Resize(227),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
# 加载训练数据集
trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)
# 加载测试数据集
testset = datasets.CIFAR10(root='./data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True, num_workers=2)

net = AlexNetPlusLatent(args.bits)

# 使用cuda加速计算
use_cuda = torch.cuda.is_available()

if use_cuda:
    net.cuda()

# softmax层loss计算
softmaxloss = nn.CrossEntropyLoss().cuda()

# 设置优化器
optimizer4nn = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

# 更新学习率
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[64], gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    # 将module设置为training mode
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # 声明变量
        inputs, targets = Variable(inputs), Variable(targets)
        # 获取网络的输出结果
        _, outputs = net(inputs)
        # 计算loss
        loss = softmaxloss(outputs, targets)
        # 将module中的所有模型梯度参数设置为0
        optimizer4nn.zero_grad()

        
        loss.backward()

        optimizer4nn.step()

        train_loss += softmaxloss(outputs, targets).item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
    return train_loss/(batch_idx+1)

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
    acc = 100*int(correct) / int(total)
    if epoch == args.epoch:
        print('Saving')
        if not os.path.isdir('{}'.format(args.path)):
            os.mkdir('{}'.format(args.path))
        torch.save(net.state_dict(), './{}/{}'.format(args.path, acc))

if args.pretrained:
    net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
    test()
else:
    if os.path.isdir('{}'.format(args.path)):
        shutil.rmtree('{}'.format(args.path))
    for epoch in range(start_epoch, start_epoch+args.epoch):
        train(epoch)
        test()
        scheduler.step()
