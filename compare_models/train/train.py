import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import model_dic


def train(model_name, use_cuda=True, epoch_size=10, pretrained=True):
    # download MNIST data
    transform   = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    trainset    = torchvision.datasets.CIFAR10('./image_data/CIFAR10', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True,num_workers=2)
    testset     = torchvision.datasets.CIFAR10('./image_data/CIFAR10', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, drop_last=False, num_workers=2)

    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    model_dic_data = model_dic.make_model_dic(pretrained)
    model = model_dic_data[model_name]
    new_model_dic = {}
    net = model.to(device) 
    
    # learn by nn
    criterion = nn.CrossEntropyLoss() # これがCost Function!!
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9) # これがminimumにいってくれるやつ
    highest_epoch = 0
    highest_accuracy = 0
    for epoch in range(epoch_size):
        tr_accuracy = 0
        accuracy = 0
        net = net.train()
        new_model_dic[epoch] = {'train':0, 'test':0}
        for (tr_x, tr_y) in tqdm(trainloader): # [:10500]:
            tr_x, tr_y = tr_x.to(device), tr_y.to(device)
            optimizer.zero_grad()
            # print(tr_x.size()) → torch.Size([4, 1, 28, 28])
            # print('tr_y:',tr_y)
            pred_y = net(tr_x)
            # print(pred_y)  #→ (4,10)
            loss = criterion(pred_y, tr_y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _, pred_y = torch.max(pred_y, dim=1)
                tr_accuracy += torch.sum(pred_y == tr_y,0)
                # print(tr_accuracy)
        print(f'Accuracy in {epoch} is : {tr_accuracy/60000:.4f}')
        print('finish training')
        new_model_dic[epoch]['train'] = f'{tr_accuracy/60000:.4f}'

        # validate
        net = net.eval() # これがあると、バッチサイズ1でも大丈夫：batch_norm
        with torch.no_grad(): # gradを計算し無くて済む→すぴーどが上がる
            for (test_x, test_y) in tqdm(testloader):
                test_x, test_y = test_x.to(device), test_y.to(device)
                pred_y = net(test_x)
                loss = criterion(pred_y, test_y)
                _, pred_y = torch.max(pred_y, dim=1)
                accuracy += torch.sum(pred_y == test_y,0)
                # print(accuracy)
            highest_epoch = epoch if accuracy > highest_accuracy else highest_epoch
            highest_net = net if accuracy > highest_accuracy else highest_net
            highest_accuracy = accuracy if accuracy > highest_accuracy else highest_accuracy
            print(f'Accuracy in {epoch} is : {accuracy/10000:.4f}')
            print('finish testing')
            new_model_dic[epoch]['test'] = f'{accuracy/10000:.4f}'
   
    # print accuracy
    print(f'Highest accuracy in {highest_epoch} is : {highest_accuracy/10000:.4f}')
    new_model_dic['highest_epoch'] = {highest_epoch:f'{highest_accuracy/10000:.4f}'}

    # # 指数平均とろう
    # PATH = './mnist_model_path.pth'
    # torch.save(highest_net.state_dict(), PATH)
    return new_model_dic

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    train(model)