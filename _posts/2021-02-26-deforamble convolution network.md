---
title: "Deformable Convolutional Networks Paper Review"
excerpt: "DCN Review"
date: '2021-02-26'
categories : paper-review
tags : [DCN,paper,review,deformable]
---



#### Deformable Convolutional Networks


#### Convolutional(합성곱 연산)

* 두 함수를 합성하는 합성곱 연산

* 한 함수를 뒤집고 이동하면서, 두 함수의 곱을 적분하여 계산

#### Pooling Layer

* 여러 화소를 종합하여 하나의 화소로 변환하는 계층이다.

* 풀링 계층을 통과하면 영상의 크기가 줄어들고, 정보가 종합된다.

    * 최댓값과 평균값을 사용한 풀링이 가장 많이 사용된다.
    



#### kernel=filter

* 커널이 커지면 연산량이 늘어나다.
* 이미지가 금방 축소되어서 네트워크의 깊이가 충분하지 않다.

#### strid 

* stride가 2이면 filter가 2칸씩 이동한다.

* stride 2로해서 pooling을 하면 영상이 작아진다.
  * 작아진 영상에서 3x3 feature를 뽑아서 원래 영상을 보면 6x6을 보는 효과가 있다. 즉, pooling에 의해서 하나의 커널이 넓은 영역을 커버

#### dilation 

* 커널사의의 간격 

* receptive field를 효율적으로 넓힐 수 있다.

* 공간 해상도 보전.

  <img src="https://raw.githubusercontent.com/landgm/image/image/img/image-20210226153732363.png" alt="image-20210226153732363" style="zoom: 33%;" />
  
    
#### Convolutional neural networks

   * CNN(Convolutional Neural Network)은 이미지의 공간 정보를 유지하면서 인접 이미지와의 특징을 효과적으로 인식하고 강조하는 방식으    로 **이미지의 특징을 추출**하는 부분과 **이미지를 분류**하는 부분으로 구성됩니다. 특징 추출 영역은 **Filter**를 사용하여 공유 파라미터 수를       최소화하면서 이미지의 특징을 찾는 Convolution 레이어와 특징을 **강화하고 모으**는 Pooling 레이어로 구성됩니다.

   * CNN은 Filter의 크기, Stride, Padding과 Pooling 크기로 출력 데이터 크기를 조절하고, 필터의 개수로 출력 데이터의 채널을 결정합니    다.

   * CNN는 같은 레이어 크기의 Fully Connected Neural Network와 비교해 볼 때, 학습 파라미터양은 20% 규모입니다. 은닉층이 깊어질 수록       학습 파라미터의 차이는 더 벌어집니다. CNN은 Fully Connected Neural Network와 비교하여 더 작은 학습 파라미터로 더 높은 인식률을       제공합니다. 
   
   * [출처](http://taewan.kim/post/cnn/)


##### Abstract

* 고정된 convolutional filter (ex 3 x 3, 5 x 5)대신 알고리즘이 배워서 다르게 하자

* convolution과 ROI pooling에 learnable offset을 줘서 정사각형이 아닌 약간 틀어진 형태로 만들자 



##### Introduction
* augmenting을 사용(고양이 사진에서 뒤집은 사진, 조금 잘려진 사진 등을 학습시킨다.)

    * lable preserving을 하면서 data augmentaion을 알고 있어야한다(고양이라는 사실은 변하면 안된다).

    * 암세포를 찾을 때 사이즈를 scailing을 하면 암인지 아닌지에 대한 label을 바꿀 수 있다.(사람이 하기에는 한계)


* SIFT 사용 
    * SIFT (Scale-Invariant Feature Transform)은 이미지의 크기와 회전에 불변하는 특징을 추출하는 알고리즘입니다. 서로 다른 두 이미지       에서 SIFT 특징을 각각 추출한 다음에 서로 가장 비슷한 특징끼리 매칭해주면 두 이미지에서 대응되는 부분을 찾을 수 있다는 것이 기본       원리입니다. 즉, 그림 1과 같이 크기와 회전은 다르지만 일치하는 내용을 갖고 이미지에서 동일한 물체를 찾아서 매칭해줄 수 있는 알고리즘입니다.
    [출처](https://bskyvision.com/21) 
    
    * 이상적이지만 쉽지않다.

* 현존하는 방법은 일반화하기 어렵고 복잡할 때 사용할 수 없다.
  
* 이미지 내에 있는 물체의 크기에 따라 다른 크기의 filter를 사용하는게 좋다

#### 극복하기 위한 방안

* deformable convolution

* deformable ROI pooling (ROI pooling은 다양한 사이즈에서 fixed 사이즈를 얻는다.) 

    * 이러한 방법은 spatial transform networks(가장자리에 있는 숫자를 가운데로 옮겨주는 역할)와 비슷하다.


#### Deformable Convolution

* 고정된 필터 R ={(-1,-1),(-1,0),...,(0,1),(1,1)} 3*3 filter dilation 1 에 offset을 주자.

* offset을 분수로 준다. ex) (1, 1.5) ,(1.3, 0.7)

* offset과 w를 따로 학습해서 convolution




#### Deformable ROI Pooling

* fc(fully connected layer를 통해서 offset 학습)


#### Deformable ConvNets

* 기존의 convolution layer를 살짝만 바꿔도 사용가능.
* 구조만 바꿔주면 자동으로 spatial transform을 해준다
* Deformable ConvNets은 두가지 stage가 있다
    * 전체 입력 이미지에 대한 특징 맵을 생성합니다.
    * a shallow task specific network generates results from the feature maps. 

* 마지막 3개 layer에 deformable convolution을 적용하는게 가장 좋았다.



---

#### 결론
* deformable filter는 background,small object, large object를 detection할 때 높은 성능을 보인다.
    * 물체의 모양에 따라 Receptive Field가 유기적으로 변형된다.

* deformable roi pooling은 예를들어 강아지 얼굴 손 발 별로 크기가 조절되면서 판별한다.
    * 필터의 모양을 기반으로 물체의 위치와 종류를 파악
* regular mnist에 대해서 regular cnn과 deformable cnn이 차이는 없다

* scaled mnist에 대해서 regular cnn보다 deformable cnn이 성능이 매우 높다. 

* 데이터의 특징을 필터를 통해서 찾는 것이 아니라 입력 데이터 x에서 직접 찾으려는 시도를 했다.


---

#### References
* [PR-002: Deformable Convolutional Networks (2017) ](https://www.youtube.com/watch?v=RRwaz0fBQ0Y&list=PLXiK3f5MOQ760xYLb2eWbtOKOwUC-bByj&index=3)
* [CNN, Convolutional Neural Network 요약](http://taewan.kim/post/cnn/)
* [ SIFT (Scale Invariant Feature Transform)의 원리.txt](https://bskyvision.com/21)
* [Deformable Convolution Nertworks 분석](https://ys-cs17.tistory.com/33)
* [논문 Deformable Convolutional Network](https://m.blog.naver.com/PostView.nhn?blogId=leesoo9297&logNo=221165325526&proxyReferer=https:%2F%2Fwww.google.com%2F)



```python

```
