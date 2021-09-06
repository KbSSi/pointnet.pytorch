from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)  # 生成的随机数种子
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform) # 分类模型,num_classes是4

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0): # 这里的参数0指示i是从0开始还是1开始，若填1，则是从1开始，data中包括数据和标签，这里的dataloader size是84
        # points的维度：(32,2500,3)  target维度：(32,2500) 其中2500代表采样点的个数，32代表batch_size,确定参数时候确定的
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()

        # 两行固定写法
        optimizer.zero_grad()
        classifier = classifier.train()

        # 分类网络输出的pred维度是(32,2500,4),4代表该点所对应的4种分类类别的score
        pred, trans, trans_feat = classifier(points)  # 将数据点喂进分类网络

        # 调整pred和target的维度，方便求loss
        # 这里size变成（80000，4）,view的作用是调整维度，前面一个-1是让函数根据num_classes# 来调整维度，8000是由32 * 2500得来的
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1 # size为8000
        # print(pred.size(), target.size())

        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001

        # 两行固定写法
        loss.backward()
        optimizer.step()

        pred_choice = pred.data.max(1)[1] # 取出pre数据中score数据最大值的索引，相当于分类的标签
        correct = pred_choice.eq(target.data).cpu().sum() # 与target标签相同的个数
        # .item 将一个0维张量转换成浮点数
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

        # 这里对测试集做一次测试，以便于更细的观察，有些代码写法是每轮epoch做一次
        # if i % 10 == 0:
        #     j, data = next(enumerate(testdataloader, 0))
        #     points, target = data
        #     points = points.transpose(2, 1)
        #     points, target = points.cuda(), target.cuda()
        #     classifier = classifier.eval()
        #     pred, _, _ = classifier(points)
        #     pred = pred.view(-1, num_classes)
        #     target = target.view(-1, 1)[:, 0] - 1
        #     loss = F.nll_loss(pred, target)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(target.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))

    j, data = next(enumerate(testdataloader, 0))
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred = pred.view(-1, num_classes)
    target = target.view(-1, 1)[:, 0] - 1
    loss = F.nll_loss(pred, target)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    print('accuracy: %f' % (correct.item() / float(opt.batchSize * 2500)))

    # 保存训练好的模型
    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    # 每次epoch保留一个模型

## benchmark mIOU
shape_ious = []
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(2)[1]

    pred_np = pred_choice.cpu().data.numpy()
    target_np = target.cpu().data.numpy() - 1

    for shape_idx in range(target_np.shape[0]):
        parts = range(num_classes)#np.unique(target_np[shape_idx])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))  # 求交集
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))   # 求并集
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            print("I:{} U:{}".format(I, U))
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))

print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))