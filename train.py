import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms

import csv
import os


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)


def get_train_dataloader(method='baseline'):
    mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    # 训练集预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    if method == 'cutout':
        train_transform.transforms.append(Cutout(args['n_holes'], args['length']))
    cifar100_train_dataset =datasets.CIFAR100(root='./',
                    train=True,
                    transform=train_transform,
                    download=True)
    cifar100_train_loader = DataLoader(dataset=cifar100_train_dataset,
                        batch_size=args['batch_size'],
                        shuffle=True,
                        pin_memory=True)
    return cifar100_train_loader

def get_test_dataloader():
    mean=[x/255.0 for x in [125.3, 123.0, 113.9]]
    std=[x/255.0 for x in [63.0, 62.1, 66.7]]
    # 测试集预处理
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
    cifar100_test_dataset =datasets.CIFAR100(root='./',
                    train=False,
                    transform=test_transform,
                    download=False)
    cifar100_test_loader = DataLoader(dataset=cifar100_test_dataset,
                        batch_size=args['batch_size'],
                        shuffle=False,
                        pin_memory=True)
    return cifar100_test_loader


class CSVLogger:
    def __init__(self, fieldnames, method ='baseline'):
        filename = './runs_2/CIFAR100_ResNet18_' + method + '.csv'
        self.csv_file = open(filename, 'a')
        writer = csv.writer(self.csv_file)
        writer.writerow([''])
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def train_cutout(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        model.zero_grad()
        pred = model(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # 计算训练过程中的准确率
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        # 打印训练过程中的loss和acc
        progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

    return (xentropy_loss_avg / (i + 1)), accuracy


def train_mixup(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        if args['cuda']:
            inputs, targets = inputs.cuda(), targets.cuda()

        # mixup数据处理
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args['alpha'], args['cuda'])
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        model.zero_grad()
        outputs = model(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(xentropy='%.3f' % (train_loss / (batch_idx + 1)), acc='%.3f' % (correct / total))

    return (train_loss / (batch_idx + 1)), (correct / total).item() / 100

def train_cutmix(epoch,model,criterion,optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader)
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))
        if args['cuda']:
            inputs, targets = inputs.cuda(), targets.cuda()

        r = np.random.rand(1)
        if args['alpha'] > 0 and r < args['cutmix_prob']: # 使用cutmix的概率
          # 生成mix的样本
            lam = np.random.beta(args['alpha'], args['alpha'])
            if args['cuda']:
                rand_index = torch.randperm(inputs.size()[0]).cuda()
            else:
                rand_index = torch.randperm(inputs.size()[0])
            target_a = targets
            target_b = targets[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # 调整lambda以与像素比匹配
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # 计算输出
            output = model(inputs)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            # 计算输出
            output = model(inputs)
            loss = criterion(output, targets)

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(xentropy='%.3f' % (train_loss / (batch_idx + 1)),
                acc='%.3f' % (correct / total))

    return (train_loss / (batch_idx + 1)), (correct / total).item()


# 测试函数
def test(model):
    model.eval()
    correct = 0.
    total = 0.
    for images, labels in test_loader:
        if args['cuda']:
            images = images.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    test_acc = correct / total
    model.train()
    return test_acc



args = {"dataset": 'cifar100', 
      "model": 'resnet18',  
      "batch_size" : 128,
      "epochs" : 100,  
      "learning_rate": 0.1, 
      "n_holes": 1,
      "length": 16,
      'alpha':0.2,
      "cutmix_prob": 0.1
}
args['cuda'] = torch.cuda.is_available()
cudnn.benchmark = True

torch.manual_seed(0)
if args['cuda']:
    torch.cuda.manual_seed(0)

train_loader = get_train_dataloader()
test_loader = get_test_dataloader()

def param_basedOn_method(method):
    if method == 'cutout':
        train = train_cutout
        writer = SummaryWriter('./runs_2/train_cutout') # 使用tensorboard进行可视化
    elif method == 'mixup':
        train = train_mixup
        writer = SummaryWriter('./runs_2/train_mixup')
    elif method == 'cutmix':
        train = train_cutmix
        writer = SummaryWriter('./runs_2/train_cutmix')
    elif method == 'baseline':
        train = train_cutout
        writer = SummaryWriter('./runs_2/train_baseline') 
    return train,writer


def train_save_model(method):
    # 模型
    model = ResNet18(num_classes=100)
    if args['cuda']:
        model = model.cuda()
    # 定义损失函数
    if args['cuda']:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'],
                    momentum=0.9, nesterov=True, weight_decay=5e-4)
    # 定义学习率优化
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.2)
    # 数据储存到csv文件
    try:
        os.makedirs('./runs_2')
    except:
        pass
    csv_logger = CSVLogger(fieldnames=['epoch', 'train_loss', 'train_acc', 'test_acc'],method=method)
    # 训练模型过程
    train,writer = param_basedOn_method(method)
    for epoch in range(1, args['epochs'] + 1):
        train_loss, train_acc = train(epoch,model,criterion,optimizer)
        test_acc = test(model)
        tqdm.write('test_acc: %.3f' % test_acc)
        scheduler.step()
        row = {'epoch': str(epoch), 'train_loss':str(train_loss), 'train_acc': str(train_acc), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        writer.add_scalar('test_acc', test_acc, global_step=epoch)
    writer.close()
    # 保存模型
    try:
        os.makedirs('./mycheckpoints')
    except:
        pass
    torch.save(model.state_dict(), './mycheckpoints/CIFAR100_ResNet18_' + method + '.pth')
    csv_logger.close()


method = 'cutmix'
train_save_model(method)