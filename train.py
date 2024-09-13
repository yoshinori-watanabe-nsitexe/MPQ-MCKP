import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 
import mpq

def train(num_epochs,model,dataloader,criterion,optimizer,max_iteration=1000000):
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs,targets)
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#                fp.write(f'Epoch,{epoch+1}, {loss.item():.4f}')
                
            if(i>max_iteration):
                break
    return model

def main(args):
    num_epochs=args.num_epochs
    batchsize=args.batchsize
    lr=args.learningrate
    data_dir=args.data_dir
    max_iteration=args.max_iteration    
    pretrained=args.pretrained
    DEBUG=args.DEBUG

    targetbit=10
    Bset=[b for b in range(16)]
    
    tr=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if(args.dataset=="CIFAR10"):
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True,transform=tr)
        label_num=10
    elif(args.dataset=="CIFAR100"):
        dataset_train = datasets.CIFAR100(data_dir, train=True, download=True,transform=tr)
        label_num=100
    elif(args.dataset=="FashionMNIST"):
        dataset_train = datasets.FashionMNIST(data_dir, train=True, download=True,transform=tr)
        label_num=10
    else:
        tr=transforms.Compose([
            transforms.Resize((7,7)), #resnet18
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST(data_dir, train=True, download=True,transform=tr)
        label_num=10

    dataloader = DataLoader(dataset_train, batchsize, shuffle=True)

    #criterion = LabelSmoothingCrossEntropy(args.smoothing) #.cuda()
    criterion = nn.CrossEntropyLoss()
#
#    model= torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)
    model= torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)

    if(args.opt=="Adam"):
        optimizer = optim.Adam(list(model.parameters()),  lr = lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    if(not pretrained):
        model=train(num_epochs,model,dataloader,criterion,optimizer,max_iteration)

    bitassignment=mpq.getBitAssignment(dataloader,model,criterion,Bset,targetbit,batchsize,DEBUG=DEBUG,itenum=max_iteration)    
    print("assgined bit for layers ",bitassignment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,default='data', help='Directory for storing input data')
    parser.add_argument('-n', '--num_epochs',default=10,type=int)
    parser.add_argument('-bs', '--batchsize',default=4,type=int)
    parser.add_argument('-ds', '--dataset',default="mnist")    
    parser.add_argument('-l', '--learningrate',default=1e-4,type=float)
    parser.add_argument('-opt', '--opt',default="SGD")    
    parser.add_argument('-m', '--max_iteration',default=1000000,type=int)
    parser.add_argument('-p', '--pretrained',default=True)
    parser.add_argument('-D', '--DEBUG',default=False)

    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args)