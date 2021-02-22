---
title: "GAN Paper Review"
excerpt: "GAN Review"
date: '2021-02-21'
categories : paper-review
tags : [GAN,paper,review]
---

##### https://github.com/yunjey/pytorch-tutorial 님의 코드를 보고 코멘트를 추가했습니다.

Discriminator는 Generator가 생성하는 값이 가짜로 구별하기 위해서 노력하고

Generator는 Discriminator가 가짜로 판별할 수 있도록 결과를 생성한다

### 간단한 코드


```python
import torch
import torch.nn as nn
```


```python

D = nn.Sequential( #Discriminator 설정 
    nn.Linear(784 ,128),#레이어 설정 입력이 784차원(mnist 28*28) 출력이 128차원 
    nn.ReLU(), #activate function relu
    nn.Linear(128, 1),
    nn.Sigmoid())

G = nn.Sequential( #generator function 설정
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    nn.Tanh()) # 생성된 값이 -1 ~ 1

criterion = nn.BCELoss() # Binary Cross Entropy Loss(h(x), y), Sigmoid Cross Entropy Loss 함수라고도 불림. -ylogh(x)-(1-y)log(1-h(x))

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.01) #최적화 알고리즘은 adam 사용
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.01)
# 충돌하기에 2개의 optimizer를 설정

while True:
    # train D
    loss = criterion(D(x), 1) + criterion(D(G(z)), 0) #d(x)는 진짜 이미지라서 1이 나오도록 학습 #d(g(z))는 0이 나오도록 학습
    loss.backward() # 모든 weight에 대해 gradient값을 계산
    d_optimizer.step()

    # train G
    loss = criterion(D(G(z)), 1)
    loss.backward()
    g_optimizer.step() # generator의 파라미터를 학습 discriminator랑 다르게 학습
```


```python
pip install torchvision
```

    Requirement already satisfied: torchvision in c:\anaconda3\lib\site-packages (0.8.2)
    Requirement already satisfied: numpy in c:\anaconda3\lib\site-packages (from torchvision) (1.18.1)
    Requirement already satisfied: torch==1.7.1 in c:\anaconda3\lib\site-packages (from torchvision) (1.7.1)
    Requirement already satisfied: pillow>=4.1.1 in c:\anaconda3\lib\site-packages (from torchvision) (7.0.0)
    Requirement already satisfied: typing_extensions in c:\anaconda3\lib\site-packages (from torch==1.7.1->torchvision) (3.7.4.3)
    Note: you may need to restart the kernel to use updated packages.



```python
conda install torchvision -c pytorch
```

    Collecting package metadata (current_repodata.json): ...working... done
    Solving environment: ...working... done
    
    ## Package Plan ##
    
      environment location: C:\Anaconda3
    
      added / updated specs:
        - torchvision


​    
​    

```python
import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


```


```python

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

```


```python
# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

```


```python

# Image processing
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])
transform = transforms.Compose([ #transforms.Compose를 통해 이미지파일을 resize, totensor, normalize 시켜준다.
                transforms.ToTensor(), #이미지 데이터를 tensor로 바꿔준다. pytorch 이용을 위해서 필요
                transforms.Normalize(mean=[0.5],   # 1 for greyscale channels
                                     std=[0.5])])

```


```python

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to ../../data/MNIST\raw\train-images-idx3-ubyte.gz



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Extracting ../../data/MNIST\raw\train-images-idx3-ubyte.gz to ../../data/MNIST\raw
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz to ../../data/MNIST\raw\train-labels-idx1-ubyte.gz



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Extracting ../../data/MNIST\raw\train-labels-idx1-ubyte.gz to ../../data/MNIST\raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz to ../../data/MNIST\raw\t10k-images-idx3-ubyte.gz



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Extracting ../../data/MNIST\raw\t10k-images-idx3-ubyte.gz to ../../data/MNIST\raw
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to ../../data/MNIST\raw\t10k-labels-idx1-ubyte.gz



    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))


    Extracting ../../data/MNIST\raw\t10k-labels-idx1-ubyte.gz to ../../data/MNIST\raw
    Processing...


    C:\Anaconda3\lib\site-packages\torchvision\datasets\mnist.py:480: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  ..\torch\csrc\utils\tensor_numpy.cpp:141.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


    Done!



```python

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

```


```python
# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())
```


```python
# Device setting
D = D.to(device)
G = G.to(device)

```


```python
# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

```


```python
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

```


```python
# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
    
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
    
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

# Save the model checkpoints 
torch.save(G.state_dict(), 'G.ckpt')
torch.save(D.state_dict(), 'D.ckpt')
```



    ---------------------------------------------------------------------------
    
    KeyboardInterrupt                         Traceback (most recent call last)
    
    <ipython-input-21-74e7a8cf5473> in <module>
         23         z = torch.randn(batch_size, latent_size).to(device)
         24         fake_images = G(z)
    ---> 25         outputs = D(fake_images)
         26         d_loss_fake = criterion(outputs, fake_labels)
         27         fake_score = outputs


    C:\Anaconda3\lib\site-packages\torch\nn\modules\module.py in _call_impl(self, *input, **kwargs)
        725             result = self._slow_forward(*input, **kwargs)
        726         else:
    --> 727             result = self.forward(*input, **kwargs)
        728         for hook in itertools.chain(
        729                 _global_forward_hooks.values(),


    C:\Anaconda3\lib\site-packages\torch\nn\modules\container.py in forward(self, input)
        115     def forward(self, input):
        116         for module in self:
    --> 117             input = module(input)
        118         return input
        119 


    C:\Anaconda3\lib\site-packages\torch\nn\modules\module.py in _call_impl(self, *input, **kwargs)
        725             result = self._slow_forward(*input, **kwargs)
        726         else:
    --> 727             result = self.forward(*input, **kwargs)
        728         for hook in itertools.chain(
        729                 _global_forward_hooks.values(),


    C:\Anaconda3\lib\site-packages\torch\nn\modules\activation.py in forward(self, input)
        297 
        298     def forward(self, input: Tensor) -> Tensor:
    --> 299         return torch.sigmoid(input)
        300 
        301 


    KeyboardInterrupt: 



```python

```
