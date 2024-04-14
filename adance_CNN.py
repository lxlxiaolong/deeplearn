import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307, ), (0.3081, ))
])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
train=True,
download=True,
transform=transform)
train_loader = DataLoader(train_dataset,
shuffle=True,
batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
train=False,
download=True,
transform=transform)
test_loader = DataLoader(test_dataset,
shuffle=False,
batch_size=batch_size)
class InceptionA(torch.nn.Module):
    def __init__(self,in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        brach1x1 = self.branch1x1(x)

        brach5x5 = self.branch5x5_1(x)
        brach5x5 = self.branch5x5_2(brach5x5)

        brach3x3 = self.branch3x3_1(x)
        brach3x3 = self.branch3x3_2(brach3x3)
        brach3x3 = self.branch3x3_3(brach3x3)

        brach_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding = 1)
        brach_pool = self.branch_pool(brach_pool)

        outputs = [brach1x1,brach5x5,brach3x3,brach_pool]
        return torch.cat(outputs,dim=1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.pooling =torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(1408,10)
    def forward(self,x):
        in_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.pooling(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size,-1)
        x = self.fc(x)
        return x

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)
def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss =criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss+= loss.item()
        if batch_idx%300==299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss=0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,label =data
            images, label = images.to(device), label.to(device)
            outputs =model(images)
            _,predicted = torch.max(outputs.data,dim=1)
            total += label.size(0)
            correct += (predicted==label).sum().item()
    print('Accuracy on test set: %d %%[ %d/%d ]' % (100 * correct / total,correct,total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()