import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
# 加载 .mat 文件
data1 = loadmat('train.mat')
label1 = loadmat('label.mat')
all_data = data1['train']
all_label = label1['train_label']
# all_data = torch.from_numpy(all_data)
# all_label = torch.from_numpy(all_label)
all_data = torch.tensor(all_data,dtype=torch.float32)
all_label = torch.tensor(all_label,dtype=torch.int64)
Loss_list = []
Accuracy_list = []


data_train, data_test, label_train, label_test = train_test_split(all_data, all_label, test_size = 0.2, random_state = 42)
train_dataset = TensorDataset(data_train,label_train)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=64)
test_dataset = TensorDataset(data_test,label_test)
test_loader = DataLoader(test_dataset,shuffle=True,batch_size=64)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3,padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3,padding=1)
        self.mp = torch.nn.MaxPool2d(2)
        self.conv4 = torch.nn.Conv2d(128, 64, kernel_size=3,padding=1)
        # self.rblock1 = ResidualBlock(16)
        # self.rblock2 = ResidualBlock(32)
        self.fc1 = torch.nn.Linear(576, 256)
        self.fc2 = torch.nn.Linear(256, 121)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.mp(x)
        x = F.relu(self.conv4(x))
        x = self.mp(x)
        x = x.view(in_size, -1)
        # x = self.rblock1(x)
        # x = self.mp(F.relu(self.conv2(x)))
        # x = self.rblock2(x)
        # x = x.view(in_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        target = target.squeeze()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))

            Loss_list.append(running_loss / 300)
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, label = data
            label = label.squeeze()
            images, label = images.to(device), label.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    print('Accuracy on test set: %d %%[ %d/%d ]' % (100 * correct / total, correct, total))
    Accuracy_list.append(100 * correct / total)

if __name__ == '__main__':
    number = 10
    for epoch in range(number):
        train(epoch)
        test()
#     print(all_label.size)
    x1 = range(0, number)
    x2 = range(0, 2*number)
    y1 = Accuracy_list
    y2 = Loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.ylim(0, 100)
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')

    plt.show()


# 定义两个数组





# 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上

