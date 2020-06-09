import torch
import torch.nn as nn
from models.net import SEMobileNetV1 as SEMobileNetV1
from models.net import MobileNetV1 as MobileNetV1
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision


val_dir = '/mnt/imagenet/raw_jpeg/2012/val/'
train_dir = '/mnt/imagenet/raw_jpeg/2012/train/'
input_size = 224
batch_size = 64
num_workers = 24
epochs = 200
lr = 1e-1
lr_step = 50


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
criterion = nn.CrossEntropyLoss()
def val(model):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batchid, (data, label) in enumerate(val_loader):
            # print(batchid)
            data, label = data.cuda(), label.cuda()
            out = model(data)
            loss = criterion(out, label)
            total_loss += loss.item()
            _, predicted = out.max(1)
            total += data.size(0)
            correct += predicted.eq(label).sum().item()
            # return correct / total
    print('Loss: ', total_loss/(batchid+1))
    print('Accuracy: ', correct / total)
    return correct / total


def train(model):
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
    
    model.train()
    for epochid in range(epochs):
        total_loss = 0
        total = 0
        correct = 0
        for batchid, (data, lable) in enumerate(train_loader):
            data, lable = data.cuda(), lable.cuda()
            optimizer.zero_grad()
            out = model(data)
            # print(out.size())
            loss = criterion(out, lable)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = out.max(1)
            correct += predicted.eq(lable).sum().item()
            total += lable.size(0)
            if batchid % 100 == 0:
                print('Epoch%d Batch %d Loss:%.3f Acc:%.3f LR:%f' %
                    (epochid, batchid, total_loss/(batchid+1), correct/total, lr_scheduler.get_last_lr()[0]))
        lr_scheduler.step()
        val(model)


if __name__ == '__main__':
    # net = SEMobileNetV1()
    # net = MobileNetV1()
    net = torchvision.models.resnet18()
    print(net)
    net.cuda()
    train(net)
    stat_dict = net.state_dict()
    torch.save(stat_dict, 'seretina.pth')