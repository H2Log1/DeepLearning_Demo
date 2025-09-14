## print("Hello, World!")

import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

torch.manual_seed(1)  # 设置随机种子, 用于复现

# 超参数
EPOCH = 1       # 前向后向传播迭代次数
LR = 0.001      # 学习率 learning rate 
BATCH_SIZE = 50 # 批量训练时候一次送入数据的size
DOWNLOAD_MNIST = True 

# 设备设置（自动使用 GPU 如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载mnist手写数据集
# 训练集
train_data = torchvision.datasets.MNIST(  
    root = './MNIST/',                      
    train = True,                            
    transform = torchvision.transforms.ToTensor(),                                                
    download=DOWNLOAD_MNIST 
 )
 
# 测试集
test_data = torchvision.datasets.MNIST(root='./MNIST/', train=False)  # train设置为False表示获取测试集
 
# 一个批训练 50个样本, 1 channel通道, 图片尺寸 28x28 size:(50, 1, 28, 28)
train_loader = Data.DataLoader(
    dataset = train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
) 
#  测试数据预处理；只测试前2000个
test_x = torch.unsqueeze(test_data.data,dim=1).float()[:2000] / 255.0
# shape from (2000, 28, 28) to (2000, 1, 28, 28)
test_y = test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(                      # 输入的图片 （1，28，28）
                in_channels=1,
                out_channels=16,            # 经过一个卷积层之后 （16,28,28）
                kernel_size=5,
                stride=1,                    # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)      # 经过池化层处理，维度为（16,14,14）
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(                         # 输入（16,14,14）
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),                                 # 输出（32,14,14）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)        # 输出（32,7,7）
        )

        self.out = nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)                     #（batch_size,16,14,14）
        x = self.conv2(x)                     # 输出（batch_size,32,7,7）
        x = x.view(x.size(0),-1)              # (batch_size,32*7*7)
        out = self.out(x)                     # (batch_size,10)
        return out

cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR) # 定义优化器
loss_func = nn.CrossEntropyLoss() # 定义损失函数

for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        # 将 batch 数据移动到选定设备
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        pred_y = cnn(batch_x)
        loss = loss_func(pred_y,batch_y)

        optimizer.zero_grad() # 清空上一层梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        if step % 50 == 0:
            # 评估时关闭梯度以节省内存
            with torch.no_grad():
                test_x_device = test_x.to(device)
                test_output = cnn(test_x_device)
                pred_y = torch.max(test_output, 1)[1].cpu().numpy()  # 确保在 CPU 上再转 numpy
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())

# 打印前十个测试结果和真实结果进行对比
with torch.no_grad():
    test_output = cnn(test_x[:10].to(device))
    pred_y = torch.max(test_output, 1)[1].cpu().numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')