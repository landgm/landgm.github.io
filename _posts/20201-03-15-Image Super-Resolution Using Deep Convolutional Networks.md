---
title: "Image Super Resolution Using Deep Convolutional Networks"
excerpt: "SRCNN"
date: '2021-03-15'
categories : paper-review
tags : [SRCNN,paper,review]
use_math : true
---



#### 사전지식

* dictionary : 사전 학습은 즉 우리가 영어사전에서 모르는 단어의 뜻을 찾는 것과 같이 어떤 주어진 간단한 단서(e.g. 스펠링)를 이용해서 필요한 정보(e.g. 단어의 뜻과 용법)를 찾아내는 방식의 알고리즘이다. 
[출처](https://bskyvision.com/177)

* manifold : Manifold란 고차원 데이터(e.g Image의 경우 (256, 256, 3) or...)가 있을 때 고차원 데이터를 데이터 공간에 뿌리면 sample들을 잘 아우르는 subspace가 있을 것이라는 가정에서 학습을 진행하는 방법입니다.
[출처](https://deepinsight.tistory.com/124 [Steve-Lee's Deep Insight])

#### Abstract

* low resolution images를 high resolution image로 (저해상도 -> 고해상도)
* 각각의 요소들을 다루는 전통적인 방법과는 달리 모든 layer를 최적화한다.
* RGB를 동시에 다룰 수 있다.

#### Introdcutin

* 하나의 솔루션이 존재하는게 아니라 다양한 솔루션이 존재하는 심각한 문제

* 저해상도 이미지로부터 고해상도 이미지를 만들 때 다양한 고해상도 이미지가 있어서 어렵다.

* 이러한 문제는 강력한 사전정보로 솔루션 space를 제약함으로써 완하한다.

* 현재까지 external example based method와 sparse coding based를 활용하여 super resolution 이용

* 기존의 방법들은 pipeline의 모든 단계들에 대해서 최적화 시키지는 않는다.

---

* SRCNN이 기존의 방법과 차이점
    * 저해상도 이미지와 고해상도 이미지를 end to end mapping을 직접적으로 학습시킨다.
    * hidden layer를 이용해서 기존의 external example based approaches와 다르게 dictionary와 manifolds를 explicitly하게 학습하지 않음.
    * convolutional layer로 patch의 extraction 과 aggregation이 이루어진다.

---

* Convolutional Neural Networks For Super-Resolution

    * Patch extraction and representation
        * 저해상도 이미지 Y로 부터 patch를 추출하고 각각의 patch를 고차원벡터로 표현해준다.
        * 고차원 벡터는 feature-map의 집합으로 구성된다.
        * $F_1(Y) = \max(0,W_1*Y + B_1)$
        * $W_1은 filters B_1은 baises$
        * Relu 적용
        * kernal size c x f1 x f1 (c는 채널 사이즈 rgb면 3)
        * n1 dimensional vector를 output으로 보냄
    * Non-linear mapping
        * $F_2(Y) = \max(0,W_2*F_1(Y) + B_2)$
        *  n1 dimensional vector를 filter(n1xf2xf2)를 통해서 n2 dimensional vector로 만든다.
        * n2 dimensional vector는 high resolution patch이다.
    * Reconstuction 
        * highresolution patch를 종합하여 final high resolution patch를 만든다.
        * $F(Y) = W_3*F_2(Y) + B_3$
        * final high resolution이미지와 실제 higt resoultuon image를 비교해야한다.
    * 모든 convolution layer에 대해 padding을 하지 않는다.왜냐하면 training동안 border effects를 피하기 위해서.

---

* 다양한 평가요소

    * psnr : 최대 신호 대 잡음비 영상 화질 손실정보에 대한 평가

    * ssim : 밝기 명암 구조를 조합하여 두 영상의 유사도 평가

    * ms ssim : 다양한 배율에 대해 ssim평가

    * ifc : 서로 다른 두 확률분포에 대한 의존도 평가





#### Conclusion

    * SRCNN의 성능이 좋다.
    * 2014년 이후 거의 모든 suer resolution 문제에 대한 연구들이 srcnn을 기반으로 함

* References
    * [현아의 일희일비 블로그](https://hyuna-tech.tistory.com/entry/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Image-Super-Resolution-Using-Deep-Convolutional-Networks)
    * [딥러닝 논문 읽기 모임](https://www.youtube.com/watch?v=1jGr_OFyfa0&list=PLlMkM4tgfjnJhhd4wn5aj8fVTYJwIpWkS&index=5)


```python

```
