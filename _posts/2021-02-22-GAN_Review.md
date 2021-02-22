---
title: "GAN Paper Review"
excerpt: "GAN Review"
date: '2021-02-21'
categories : paper-review
tags : [GAN,paper,review]
---



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
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        cudatoolkit-11.0.221       |       h74a9793_0       627.0 MB
        libuv-1.40.0               |       he774522_0         255 KB
        ninja-1.10.2               |   py37h6d14046_0         246 KB
        pytorch-1.7.1              |py3.7_cuda110_cudnn8_0      1007.2 MB  pytorch
        torchvision-0.8.2          |       py37_cu110         7.2 MB  pytorch
        typing_extensions-3.7.4.3  |     pyha847dfd_0          25 KB
        ------------------------------------------------------------
                                               Total:        1.60 GB
    
    The following NEW packages will be INSTALLED:
    
      cudatoolkit        pkgs/main/win-64::cudatoolkit-11.0.221-h74a9793_0
      libuv              pkgs/main/win-64::libuv-1.40.0-he774522_0
      ninja              pkgs/main/win-64::ninja-1.10.2-py37h6d14046_0
      pytorch            pytorch/win-64::pytorch-1.7.1-py3.7_cuda110_cudnn8_0
      torchvision        pytorch/win-64::torchvision-0.8.2-py37_cu110
      typing_extensions  pkgs/main/noarch::typing_extensions-3.7.4.3-pyha847dfd_0


​    
​    
    Downloading and Extracting Packages
    
    torchvision-0.8.2    | 7.2 MB    |            |   0% 
    torchvision-0.8.2    | 7.2 MB    |            |   0% 
    torchvision-0.8.2    | 7.2 MB    |            |   1% 
    torchvision-0.8.2    | 7.2 MB    | 1          |   1% 
    torchvision-0.8.2    | 7.2 MB    | 2          |   2% 
    torchvision-0.8.2    | 7.2 MB    | 3          |   3% 
    torchvision-0.8.2    | 7.2 MB    | 3          |   4% 
    torchvision-0.8.2    | 7.2 MB    | 4          |   5% 
    torchvision-0.8.2    | 7.2 MB    | 5          |   6% 
    torchvision-0.8.2    | 7.2 MB    | 7          |   7% 
    torchvision-0.8.2    | 7.2 MB    | 7          |   8% 
    torchvision-0.8.2    | 7.2 MB    | 9          |   9% 
    torchvision-0.8.2    | 7.2 MB    | #          |  11% 
    torchvision-0.8.2    | 7.2 MB    | #1         |  12% 
    torchvision-0.8.2    | 7.2 MB    | #2         |  13% 
    torchvision-0.8.2    | 7.2 MB    | #4         |  15% 
    torchvision-0.8.2    | 7.2 MB    | #5         |  16% 
    torchvision-0.8.2    | 7.2 MB    | #6         |  17% 
    torchvision-0.8.2    | 7.2 MB    | #7         |  18% 
    torchvision-0.8.2    | 7.2 MB    | #8         |  19% 
    torchvision-0.8.2    | 7.2 MB    | ##         |  20% 
    torchvision-0.8.2    | 7.2 MB    | ##         |  21% 
    torchvision-0.8.2    | 7.2 MB    | ##1        |  22% 
    torchvision-0.8.2    | 7.2 MB    | ##2        |  23% 
    torchvision-0.8.2    | 7.2 MB    | ##3        |  24% 
    torchvision-0.8.2    | 7.2 MB    | ##5        |  25% 
    torchvision-0.8.2    | 7.2 MB    | ##6        |  27% 
    torchvision-0.8.2    | 7.2 MB    | ##8        |  28% 
    torchvision-0.8.2    | 7.2 MB    | ##9        |  30% 
    torchvision-0.8.2    | 7.2 MB    | ###        |  31% 
    torchvision-0.8.2    | 7.2 MB    | ###2       |  32% 
    torchvision-0.8.2    | 7.2 MB    | ###3       |  34% 
    torchvision-0.8.2    | 7.2 MB    | ###4       |  35% 
    torchvision-0.8.2    | 7.2 MB    | ###6       |  36% 
    torchvision-0.8.2    | 7.2 MB    | ###7       |  37% 
    torchvision-0.8.2    | 7.2 MB    | ###8       |  38% 
    torchvision-0.8.2    | 7.2 MB    | ###9       |  39% 
    torchvision-0.8.2    | 7.2 MB    | ####       |  41% 
    torchvision-0.8.2    | 7.2 MB    | ####2      |  42% 
    torchvision-0.8.2    | 7.2 MB    | ####3      |  44% 
    torchvision-0.8.2    | 7.2 MB    | ####4      |  45% 
    torchvision-0.8.2    | 7.2 MB    | ####5      |  46% 
    torchvision-0.8.2    | 7.2 MB    | ####6      |  47% 
    torchvision-0.8.2    | 7.2 MB    | ####7      |  48% 
    torchvision-0.8.2    | 7.2 MB    | ####9      |  49% 
    torchvision-0.8.2    | 7.2 MB    | #####      |  50% 
    torchvision-0.8.2    | 7.2 MB    | #####1     |  52% 
    torchvision-0.8.2    | 7.2 MB    | #####2     |  53% 
    torchvision-0.8.2    | 7.2 MB    | #####3     |  54% 
    torchvision-0.8.2    | 7.2 MB    | #####4     |  55% 
    torchvision-0.8.2    | 7.2 MB    | #####6     |  56% 
    torchvision-0.8.2    | 7.2 MB    | #####7     |  57% 
    torchvision-0.8.2    | 7.2 MB    | #####8     |  58% 
    torchvision-0.8.2    | 7.2 MB    | #####9     |  60% 
    torchvision-0.8.2    | 7.2 MB    | ######     |  61% 
    torchvision-0.8.2    | 7.2 MB    | ######1    |  62% 
    torchvision-0.8.2    | 7.2 MB    | ######2    |  63% 
    torchvision-0.8.2    | 7.2 MB    | ######4    |  64% 
    torchvision-0.8.2    | 7.2 MB    | ######5    |  65% 
    torchvision-0.8.2    | 7.2 MB    | ######6    |  66% 
    torchvision-0.8.2    | 7.2 MB    | ######7    |  68% 
    torchvision-0.8.2    | 7.2 MB    | ######8    |  69% 
    torchvision-0.8.2    | 7.2 MB    | ######9    |  70% 
    torchvision-0.8.2    | 7.2 MB    | #######1   |  71% 
    torchvision-0.8.2    | 7.2 MB    | #######2   |  72% 
    torchvision-0.8.2    | 7.2 MB    | #######3   |  73% 
    torchvision-0.8.2    | 7.2 MB    | #######5   |  75% 
    torchvision-0.8.2    | 7.2 MB    | #######6   |  76% 
    torchvision-0.8.2    | 7.2 MB    | #######7   |  77% 
    torchvision-0.8.2    | 7.2 MB    | #######9   |  79% 
    torchvision-0.8.2    | 7.2 MB    | ########   |  80% 
    torchvision-0.8.2    | 7.2 MB    | ########1  |  82% 
    torchvision-0.8.2    | 7.2 MB    | ########3  |  84% 
    torchvision-0.8.2    | 7.2 MB    | ########5  |  85% 
    torchvision-0.8.2    | 7.2 MB    | ########6  |  87% 
    torchvision-0.8.2    | 7.2 MB    | ########9  |  89% 
    torchvision-0.8.2    | 7.2 MB    | #########  |  91% 
    torchvision-0.8.2    | 7.2 MB    | #########2 |  92% 
    torchvision-0.8.2    | 7.2 MB    | #########4 |  94% 
    torchvision-0.8.2    | 7.2 MB    | #########6 |  96% 
    torchvision-0.8.2    | 7.2 MB    | #########7 |  98% 
    torchvision-0.8.2    | 7.2 MB    | ########## | 100% 
    torchvision-0.8.2    | 7.2 MB    | ########## | 100% 
    
    typing_extensions-3. | 25 KB     |            |   0% 
    typing_extensions-3. | 25 KB     | ########## | 100% 
    typing_extensions-3. | 25 KB     | ########## | 100% 
    
    cudatoolkit-11.0.221 | 627.0 MB  |            |   0% 
    cudatoolkit-11.0.221 | 627.0 MB  |            |   0% 
    cudatoolkit-11.0.221 | 627.0 MB  |            |   0% 
    cudatoolkit-11.0.221 | 627.0 MB  |            |   1% 
    cudatoolkit-11.0.221 | 627.0 MB  |            |   1% 
    cudatoolkit-11.0.221 | 627.0 MB  |            |   1% 
    cudatoolkit-11.0.221 | 627.0 MB  | 1          |   1% 
    cudatoolkit-11.0.221 | 627.0 MB  | 1          |   1% 
    cudatoolkit-11.0.221 | 627.0 MB  | 1          |   1% 
    cudatoolkit-11.0.221 | 627.0 MB  | 1          |   2% 
    cudatoolkit-11.0.221 | 627.0 MB  | 1          |   2% 
    cudatoolkit-11.0.221 | 627.0 MB  | 2          |   2% 
    cudatoolkit-11.0.221 | 627.0 MB  | 2          |   2% 
    cudatoolkit-11.0.221 | 627.0 MB  | 2          |   2% 
    cudatoolkit-11.0.221 | 627.0 MB  | 2          |   3% 
    cudatoolkit-11.0.221 | 627.0 MB  | 2          |   3% 
    cudatoolkit-11.0.221 | 627.0 MB  | 3          |   3% 
    cudatoolkit-11.0.221 | 627.0 MB  | 3          |   3% 
    cudatoolkit-11.0.221 | 627.0 MB  | 3          |   3% 
    cudatoolkit-11.0.221 | 627.0 MB  | 3          |   4% 
    cudatoolkit-11.0.221 | 627.0 MB  | 3          |   4% 
    cudatoolkit-11.0.221 | 627.0 MB  | 3          |   4% 
    cudatoolkit-11.0.221 | 627.0 MB  | 4          |   4% 
    cudatoolkit-11.0.221 | 627.0 MB  | 4          |   4% 
    cudatoolkit-11.0.221 | 627.0 MB  | 4          |   5% 
    cudatoolkit-11.0.221 | 627.0 MB  | 4          |   5% 
    cudatoolkit-11.0.221 | 627.0 MB  | 4          |   5% 
    cudatoolkit-11.0.221 | 627.0 MB  | 5          |   5% 
    cudatoolkit-11.0.221 | 627.0 MB  | 5          |   5% 
    cudatoolkit-11.0.221 | 627.0 MB  | 5          |   5% 
    cudatoolkit-11.0.221 | 627.0 MB  | 5          |   6% 
    cudatoolkit-11.0.221 | 627.0 MB  | 5          |   6% 
    cudatoolkit-11.0.221 | 627.0 MB  | 5          |   6% 
    cudatoolkit-11.0.221 | 627.0 MB  | 6          |   6% 
    cudatoolkit-11.0.221 | 627.0 MB  | 6          |   6% 
    cudatoolkit-11.0.221 | 627.0 MB  | 6          |   7% 
    cudatoolkit-11.0.221 | 627.0 MB  | 6          |   7% 
    cudatoolkit-11.0.221 | 627.0 MB  | 6          |   7% 
    cudatoolkit-11.0.221 | 627.0 MB  | 7          |   7% 
    cudatoolkit-11.0.221 | 627.0 MB  | 7          |   7% 
    cudatoolkit-11.0.221 | 627.0 MB  | 7          |   7% 
    cudatoolkit-11.0.221 | 627.0 MB  | 7          |   8% 
    cudatoolkit-11.0.221 | 627.0 MB  | 7          |   8% 
    cudatoolkit-11.0.221 | 627.0 MB  | 8          |   8% 
    cudatoolkit-11.0.221 | 627.0 MB  | 8          |   8% 
    cudatoolkit-11.0.221 | 627.0 MB  | 8          |   8% 
    cudatoolkit-11.0.221 | 627.0 MB  | 8          |   9% 
    cudatoolkit-11.0.221 | 627.0 MB  | 8          |   9% 
    cudatoolkit-11.0.221 | 627.0 MB  | 9          |   9% 
    cudatoolkit-11.0.221 | 627.0 MB  | 9          |   9% 
    cudatoolkit-11.0.221 | 627.0 MB  | 9          |   9% 
    cudatoolkit-11.0.221 | 627.0 MB  | 9          |  10% 
    cudatoolkit-11.0.221 | 627.0 MB  | 9          |  10% 
    cudatoolkit-11.0.221 | 627.0 MB  | 9          |  10% 
    cudatoolkit-11.0.221 | 627.0 MB  | #          |  10% 
    cudatoolkit-11.0.221 | 627.0 MB  | #          |  10% 
    cudatoolkit-11.0.221 | 627.0 MB  | #          |  11% 
    cudatoolkit-11.0.221 | 627.0 MB  | #          |  11% 
    cudatoolkit-11.0.221 | 627.0 MB  | #          |  11% 
    cudatoolkit-11.0.221 | 627.0 MB  | #1         |  11% 
    cudatoolkit-11.0.221 | 627.0 MB  | #1         |  11% 
    cudatoolkit-11.0.221 | 627.0 MB  | #1         |  11% 
    cudatoolkit-11.0.221 | 627.0 MB  | #1         |  12% 
    cudatoolkit-11.0.221 | 627.0 MB  | #1         |  12% 
    cudatoolkit-11.0.221 | 627.0 MB  | #2         |  12% 
    cudatoolkit-11.0.221 | 627.0 MB  | #2         |  12% 
    cudatoolkit-11.0.221 | 627.0 MB  | #2         |  12% 
    cudatoolkit-11.0.221 | 627.0 MB  | #2         |  13% 
    cudatoolkit-11.0.221 | 627.0 MB  | #2         |  13% 
    cudatoolkit-11.0.221 | 627.0 MB  | #3         |  13% 
    cudatoolkit-11.0.221 | 627.0 MB  | #3         |  13% 
    cudatoolkit-11.0.221 | 627.0 MB  | #3         |  13% 
    cudatoolkit-11.0.221 | 627.0 MB  | #3         |  14% 
    cudatoolkit-11.0.221 | 627.0 MB  | #3         |  14% 
    cudatoolkit-11.0.221 | 627.0 MB  | #3         |  14% 
    cudatoolkit-11.0.221 | 627.0 MB  | #4         |  14% 
    cudatoolkit-11.0.221 | 627.0 MB  | #4         |  14% 
    cudatoolkit-11.0.221 | 627.0 MB  | #4         |  15% 
    cudatoolkit-11.0.221 | 627.0 MB  | #4         |  15% 
    cudatoolkit-11.0.221 | 627.0 MB  | #4         |  15% 
    cudatoolkit-11.0.221 | 627.0 MB  | #5         |  15% 
    cudatoolkit-11.0.221 | 627.0 MB  | #5         |  15% 
    cudatoolkit-11.0.221 | 627.0 MB  | #5         |  15% 
    cudatoolkit-11.0.221 | 627.0 MB  | #5         |  16% 
    cudatoolkit-11.0.221 | 627.0 MB  | #5         |  16% 
    cudatoolkit-11.0.221 | 627.0 MB  | #6         |  16% 
    cudatoolkit-11.0.221 | 627.0 MB  | #6         |  16% 
    cudatoolkit-11.0.221 | 627.0 MB  | #6         |  16% 
    cudatoolkit-11.0.221 | 627.0 MB  | #6         |  17% 
    cudatoolkit-11.0.221 | 627.0 MB  | #6         |  17% 
    cudatoolkit-11.0.221 | 627.0 MB  | #6         |  17% 
    cudatoolkit-11.0.221 | 627.0 MB  | #7         |  17% 
    cudatoolkit-11.0.221 | 627.0 MB  | #7         |  17% 
    cudatoolkit-11.0.221 | 627.0 MB  | #7         |  18% 
    cudatoolkit-11.0.221 | 627.0 MB  | #7         |  18% 
    cudatoolkit-11.0.221 | 627.0 MB  | #7         |  18% 
    cudatoolkit-11.0.221 | 627.0 MB  | #8         |  18% 
    cudatoolkit-11.0.221 | 627.0 MB  | #8         |  18% 
    cudatoolkit-11.0.221 | 627.0 MB  | #8         |  18% 
    cudatoolkit-11.0.221 | 627.0 MB  | #8         |  19% 
    cudatoolkit-11.0.221 | 627.0 MB  | #8         |  19% 
    cudatoolkit-11.0.221 | 627.0 MB  | #9         |  19% 
    cudatoolkit-11.0.221 | 627.0 MB  | #9         |  19% 
    cudatoolkit-11.0.221 | 627.0 MB  | #9         |  19% 
    cudatoolkit-11.0.221 | 627.0 MB  | #9         |  20% 
    cudatoolkit-11.0.221 | 627.0 MB  | #9         |  20% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##         |  20% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##         |  20% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##         |  20% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##         |  21% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##         |  21% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##         |  21% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##1        |  21% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##1        |  21% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##1        |  21% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##1        |  22% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##1        |  22% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##1        |  22% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##2        |  22% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##2        |  22% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##2        |  23% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##2        |  23% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##2        |  23% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##3        |  23% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##3        |  23% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##3        |  23% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##3        |  24% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##3        |  24% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##4        |  24% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##4        |  24% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##4        |  24% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##4        |  25% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##4        |  25% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##4        |  25% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##5        |  25% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##5        |  25% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##5        |  26% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##5        |  26% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##5        |  26% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##6        |  26% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##6        |  26% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##6        |  26% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##6        |  27% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##6        |  27% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##7        |  27% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##7        |  27% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##7        |  27% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##7        |  28% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##7        |  28% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##8        |  28% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##8        |  28% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##8        |  28% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##8        |  29% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##8        |  29% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##8        |  29% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##9        |  29% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##9        |  29% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##9        |  30% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##9        |  30% 
    cudatoolkit-11.0.221 | 627.0 MB  | ##9        |  30% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###        |  30% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###        |  30% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###        |  30% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###        |  31% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###        |  31% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###1       |  31% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###1       |  31% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###1       |  31% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###1       |  32% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###1       |  32% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###1       |  32% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###2       |  32% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###2       |  32% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###2       |  33% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###2       |  33% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###2       |  33% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###3       |  33% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###3       |  33% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###3       |  34% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###3       |  34% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###3       |  34% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###4       |  34% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###4       |  34% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###4       |  34% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###4       |  35% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###4       |  35% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###5       |  35% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###5       |  35% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###5       |  35% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###5       |  36% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###5       |  36% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###5       |  36% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###6       |  36% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###6       |  36% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###6       |  37% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###6       |  37% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###6       |  37% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###7       |  37% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###7       |  37% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###7       |  37% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###7       |  38% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###7       |  38% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###8       |  38% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###8       |  38% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###8       |  38% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###8       |  39% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###8       |  39% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###9       |  39% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###9       |  39% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###9       |  39% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###9       |  40% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###9       |  40% 
    cudatoolkit-11.0.221 | 627.0 MB  | ###9       |  40% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####       |  40% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####       |  40% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####       |  40% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####       |  41% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####       |  41% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####1      |  41% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####1      |  41% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####1      |  41% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####1      |  42% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####1      |  42% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####1      |  42% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####2      |  42% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####2      |  42% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####2      |  43% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####2      |  43% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####2      |  43% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####3      |  43% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####3      |  43% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####3      |  43% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####3      |  44% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####3      |  44% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####4      |  44% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####4      |  44% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####4      |  44% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####4      |  45% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####4      |  45% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####5      |  45% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####5      |  45% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####5      |  45% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####5      |  46% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####5      |  46% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####5      |  46% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####6      |  46% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####6      |  46% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####6      |  47% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####6      |  47% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####6      |  47% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####7      |  47% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####7      |  47% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####7      |  47% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####7      |  48% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####7      |  48% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####8      |  48% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####8      |  48% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####8      |  48% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####8      |  49% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####8      |  49% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####8      |  49% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####9      |  49% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####9      |  49% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####9      |  49% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####9      |  50% 
    cudatoolkit-11.0.221 | 627.0 MB  | ####9      |  50% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####      |  50% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####      |  50% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####      |  50% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####      |  51% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####      |  51% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####      |  51% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####1     |  51% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####1     |  51% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####1     |  52% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####1     |  52% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####1     |  52% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####2     |  52% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####2     |  52% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####2     |  52% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####2     |  53% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####2     |  53% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####3     |  53% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####3     |  53% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####3     |  53% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####3     |  54% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####3     |  54% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####4     |  54% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####4     |  54% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####4     |  54% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####4     |  55% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####4     |  55% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####4     |  55% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####5     |  55% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####5     |  55% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####5     |  56% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####5     |  56% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####5     |  56% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####6     |  56% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####6     |  56% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####6     |  57% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####6     |  57% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####6     |  57% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####7     |  57% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####7     |  57% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####7     |  57% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####7     |  58% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####7     |  58% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####8     |  58% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####8     |  58% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####8     |  58% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####8     |  59% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####8     |  59% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####8     |  59% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####9     |  59% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####9     |  59% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####9     |  60% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####9     |  60% 
    cudatoolkit-11.0.221 | 627.0 MB  | #####9     |  60% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######     |  60% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######     |  60% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######     |  60% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######     |  61% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######     |  61% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######1    |  61% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######1    |  61% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######1    |  61% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######1    |  62% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######1    |  62% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######2    |  62% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######2    |  62% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######2    |  62% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######2    |  63% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######2    |  63% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######2    |  63% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######3    |  63% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######3    |  63% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######3    |  64% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######3    |  64% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######3    |  64% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######4    |  64% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######4    |  64% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######4    |  64% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######4    |  65% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######4    |  65% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######5    |  65% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######5    |  65% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######5    |  65% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######5    |  66% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######5    |  66% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######5    |  66% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######6    |  66% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######6    |  66% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######6    |  67% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######6    |  67% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######6    |  67% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######7    |  67% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######7    |  67% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######7    |  67% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######7    |  68% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######7    |  68% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######8    |  68% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######8    |  68% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######8    |  68% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######8    |  69% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######8    |  69% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######8    |  69% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######9    |  69% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######9    |  69% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######9    |  70% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######9    |  70% 
    cudatoolkit-11.0.221 | 627.0 MB  | ######9    |  70% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######    |  70% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######    |  70% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######    |  71% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######    |  71% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######    |  71% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######1   |  71% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######1   |  71% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######1   |  71% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######1   |  72% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######1   |  72% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######2   |  72% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######2   |  72% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######2   |  72% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######2   |  73% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######2   |  73% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######3   |  73% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######3   |  73% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######3   |  73% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######3   |  74% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######3   |  74% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######3   |  74% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######4   |  74% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######4   |  74% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######4   |  75% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######4   |  75% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######4   |  75% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######5   |  75% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######5   |  75% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######5   |  75% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######5   |  76% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######5   |  76% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######6   |  76% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######6   |  76% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######6   |  76% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######6   |  77% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######6   |  77% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######6   |  77% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######7   |  77% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######7   |  77% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######7   |  78% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######7   |  78% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######7   |  78% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######8   |  78% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######8   |  78% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######8   |  78% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######8   |  79% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######8   |  79% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######9   |  79% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######9   |  79% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######9   |  79% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######9   |  80% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######9   |  80% 
    cudatoolkit-11.0.221 | 627.0 MB  | #######9   |  80% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########   |  80% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########   |  80% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########   |  81% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########   |  81% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########   |  81% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########1  |  81% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########1  |  81% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########1  |  81% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########1  |  82% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########1  |  82% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########2  |  82% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########2  |  82% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########2  |  82% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########2  |  83% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########2  |  83% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  83% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  83% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  83% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########3  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########4  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########4  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########4  |  84% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########4  |  85% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########4  |  85% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########4  |  85% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########5  |  85% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########5  |  85% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########5  |  85% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########5  |  86% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########5  |  86% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########6  |  86% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########6  |  86% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########6  |  86% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########6  |  87% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########6  |  87% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########6  |  87% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########7  |  87% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########7  |  87% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########7  |  88% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########7  |  88% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########7  |  88% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########8  |  88% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########8  |  88% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########8  |  89% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########8  |  89% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########8  |  89% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########9  |  89% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########9  |  89% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########9  |  89% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########9  |  90% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########9  |  90% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########  |  90% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########  |  90% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########  |  90% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########  |  91% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########  |  91% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########  |  91% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########1 |  91% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########1 |  91% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########1 |  92% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########1 |  92% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########1 |  92% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########2 |  92% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########2 |  92% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########2 |  93% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########2 |  93% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########2 |  93% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########3 |  93% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########3 |  93% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########3 |  94% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########3 |  94% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########3 |  94% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########4 |  94% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########4 |  94% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########4 |  94% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########4 |  95% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########4 |  95% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########4 |  95% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########5 |  95% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########5 |  95% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########5 |  96% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########5 |  96% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########5 |  96% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########6 |  96% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########6 |  96% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########6 |  96% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########6 |  97% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########6 |  97% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########7 |  97% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########7 |  97% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########7 |  97% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########7 |  98% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########7 |  98% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########8 |  98% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########8 |  98% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########8 |  98% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########8 |  99% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########8 |  99% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########8 |  99% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########9 |  99% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########9 |  99% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########9 | 100% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########9 | 100% 
    cudatoolkit-11.0.221 | 627.0 MB  | #########9 | 100% 
    cudatoolkit-11.0.221 | 627.0 MB  | ########## | 100% 
    
    ninja-1.10.2         | 246 KB    |            |   0% 
    ninja-1.10.2         | 246 KB    | ########## | 100% 
    ninja-1.10.2         | 246 KB    | ########## | 100% 
    
    libuv-1.40.0         | 255 KB    |            |   0% 
    libuv-1.40.0         | 255 KB    | ########## | 100% 
    libuv-1.40.0         | 255 KB    | ########## | 100% 
    
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   0% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB |            |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   1% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 1          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   2% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 2          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   3% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 3          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   4% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 4          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   5% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 5          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   6% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 6          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   7% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 7          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   8% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 8          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |   9% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | 9          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  10% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #          |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  11% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #1         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  12% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #2         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  13% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #3         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  14% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #4         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  15% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #5         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  16% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #6         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  17% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #7         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  18% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #8         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  19% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  20% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  20% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  20% 
    pytorch-1.7.1        | 1007.2 MB | #9         |  20% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  20% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  20% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  20% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  20% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  20% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##         |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  21% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##1        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  22% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##2        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  23% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##3        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  24% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##4        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  25% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##5        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  26% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##6        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  27% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##7        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  28% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##8        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  29% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ##9        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  30% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###        |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  31% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###1       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  32% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###2       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  33% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###3       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  34% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###4       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  35% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###5       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  36% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###6       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  37% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###7       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  38% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###8       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  39% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ###9       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  40% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####       |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  41% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####1      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  42% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####2      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  43% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####3      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  44% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####4      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  45% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####5      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  46% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####6      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  47% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####7      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  48% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####8      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  49% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  50% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  50% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  50% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  50% 
    pytorch-1.7.1        | 1007.2 MB | ####9      |  50% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  50% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  50% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  50% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  50% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  50% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####      |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  51% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####1     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  52% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####2     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  53% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####3     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  54% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####4     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  55% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####5     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  56% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####6     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  57% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####7     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  58% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####8     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  59% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | #####9     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  60% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######     |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  61% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######1    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  62% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######2    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  63% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######3    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  64% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######4    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  65% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######5    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  66% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######6    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  67% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######7    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  68% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######8    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  69% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | ######9    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  70% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######    |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  71% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######1   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  72% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######2   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  73% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######3   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  74% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######4   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  75% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######5   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  76% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######6   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  77% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######7   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  78% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######8   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  79% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | #######9   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  80% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########   |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  81% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########1  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  82% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########2  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  83% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########3  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  84% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########4  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  85% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########5  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  86% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########6  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  87% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########7  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  88% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########8  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  89% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  90% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  90% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  90% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  90% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  90% 
    pytorch-1.7.1        | 1007.2 MB | ########9  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  90% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########  |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  91% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########1 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  92% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########2 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########3 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########3 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########3 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########3 |  93% 
    pytorch-1.7.1        | 1007.2 MB | #########3 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########3 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  94% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########4 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  95% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########5 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  96% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########6 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  97% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########7 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  98% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########8 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 |  99% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | #########9 | 100% 
    pytorch-1.7.1        | 1007.2 MB | ########## | 100% 
    pytorch-1.7.1        | 1007.2 MB | ########## | 100% 
    Preparing transaction: ...working... done
    Verifying transaction: ...working... done
    Executing transaction: ...working... done
    
    Note: you may need to restart the kernel to use updated packages.



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

    Epoch [0/200], Step [200/600], d_loss: 0.0406, g_loss: 4.2033, D(x): 0.99, D(G(z)): 0.03
    Epoch [0/200], Step [400/600], d_loss: 0.0862, g_loss: 6.5490, D(x): 0.96, D(G(z)): 0.03
    Epoch [0/200], Step [600/600], d_loss: 0.0175, g_loss: 5.7655, D(x): 0.99, D(G(z)): 0.01
    Epoch [1/200], Step [200/600], d_loss: 0.0514, g_loss: 5.6775, D(x): 0.97, D(G(z)): 0.01
    Epoch [1/200], Step [400/600], d_loss: 0.5289, g_loss: 2.8135, D(x): 0.94, D(G(z)): 0.32
    Epoch [1/200], Step [600/600], d_loss: 0.5269, g_loss: 3.8699, D(x): 0.85, D(G(z)): 0.10
    Epoch [2/200], Step [200/600], d_loss: 0.1466, g_loss: 4.4655, D(x): 0.96, D(G(z)): 0.07
    Epoch [2/200], Step [400/600], d_loss: 0.2015, g_loss: 3.4515, D(x): 0.97, D(G(z)): 0.15
    Epoch [2/200], Step [600/600], d_loss: 0.3296, g_loss: 2.6831, D(x): 0.93, D(G(z)): 0.19
    Epoch [3/200], Step [200/600], d_loss: 0.3843, g_loss: 3.3577, D(x): 0.84, D(G(z)): 0.11
    Epoch [3/200], Step [400/600], d_loss: 0.1215, g_loss: 4.6607, D(x): 0.95, D(G(z)): 0.06
    Epoch [3/200], Step [600/600], d_loss: 0.7507, g_loss: 2.4923, D(x): 0.74, D(G(z)): 0.12
    Epoch [4/200], Step [200/600], d_loss: 0.8635, g_loss: 1.9649, D(x): 0.75, D(G(z)): 0.24
    Epoch [4/200], Step [400/600], d_loss: 0.3533, g_loss: 3.4091, D(x): 0.86, D(G(z)): 0.09
    Epoch [4/200], Step [600/600], d_loss: 0.2123, g_loss: 4.0489, D(x): 0.93, D(G(z)): 0.06
    Epoch [5/200], Step [200/600], d_loss: 0.3329, g_loss: 3.6035, D(x): 0.86, D(G(z)): 0.04
    Epoch [5/200], Step [400/600], d_loss: 0.1624, g_loss: 4.0930, D(x): 0.97, D(G(z)): 0.07
    Epoch [5/200], Step [600/600], d_loss: 0.5851, g_loss: 3.9824, D(x): 0.90, D(G(z)): 0.21
    Epoch [6/200], Step [200/600], d_loss: 0.3065, g_loss: 5.3071, D(x): 0.93, D(G(z)): 0.11
    Epoch [6/200], Step [400/600], d_loss: 0.0518, g_loss: 5.0439, D(x): 0.98, D(G(z)): 0.03
    Epoch [6/200], Step [600/600], d_loss: 0.2572, g_loss: 5.8598, D(x): 0.89, D(G(z)): 0.02
    Epoch [7/200], Step [200/600], d_loss: 0.2495, g_loss: 5.9823, D(x): 0.90, D(G(z)): 0.03
    Epoch [7/200], Step [400/600], d_loss: 0.1449, g_loss: 3.6583, D(x): 0.95, D(G(z)): 0.03
    Epoch [7/200], Step [600/600], d_loss: 0.1989, g_loss: 4.4194, D(x): 0.94, D(G(z)): 0.08



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
