---
layout: post
title:  "How to make github blog!"
summary: make blog
author: gang min
date: '2021-02-19 '
category: ETC

---

# 깃허브 블로그 만드는 방법

## 수만은 시행착오 끝에 성공해서 다른 분들은 쉽게 했으면 좋겠다는 마음으로 업로드합니다.

[깃허브 테마](https://github.com/topics/jekyll-theme)

* 위에 링크를 타고 들어가면 많은 테마가 있습니다. 거기서 하나를 골라주세요

![참고](https://github.com/landgm/image/blob/master/img/20210219_210313.png?raw=true)



## 우축위에 Fork를 눌러줍니다

![참고2](https://github.com/landgm/image/blob/master/img/20210219_210603.png?raw=true)



# **Setting**에 들어가서 Repostiory name을 만들어줍니다

* **본인**의 **아이디**가 무조건들어가야합니다

    * **본인아이디.github.io**

    * 이후 **rename**을 눌러줍니다

# rename을 해준 다음에 Code에 들어가서 _config.yml에 들어가 줍니다

* 들어가서 url: https://landgm.github.io (자신의 url이 들어가도록 만들어 줍니다)

* _config.yml에서 테마와 이름등을 변경할 수 있습니다.

 이렇게하면 블로그에 글을 쓸 수 있습니다


### 이제 어떻게 글을 쓰는지에 대해서 소개하겠습니다

* Add file
    * create new file
        * (자신의 깃허브 레퍼지토리)/_posts/2021-02-19-파일이름.md  #무조건 연도 월 일 이름.md해주셔야합니다


```python

---
#layout: page또는 single
#title: "파일이름"  (해쉬태그는 빼주세요!)
---
#이렇게 설정하면 파일이름을 설정할 수 있습니다.
```


      File "<ipython-input-6-9b91c31a6fd4>", line 1
        ---
           ^
    SyntaxError: invalid syntax



# 주피터 노트북을 이용해서 파일 올리기 전에 설정할 것들
 * typora 설치
  * typora 자동 업로드 하기

[typora 설정법](https://taeuk-gang.github.io/wiki/Typora%20%EC%8B%A0%EA%B8%B0%EB%8A%A5%20-%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%9E%90%EB%8F%99%20%EC%97%85%EB%A1%9C%EB%93%9C/)

설정할 때 repository를 따로 생성해주는게 편합니다!

* 주피터 노트북을 가시면 
   * File
       * Download as
           * markdown을 이용합니다

다운받은 파일을 **_posts**에 들어가서 Add file -> uploadfile을 이용해서 포스팅해줍니다 (1분 소요)

### 저는 이미지 올리는데 오류가 많이나서 고생을 했는데요 그걸 해결한 방법을 알려드리겠습니다

### 이미지를 올릴 때는 typora에 먼저 image upload를 해야합니다 

   * github에 지정해준 image repository를 가면 image가 있습니다
   * 그 이미지 url을 가져와서 typora에 붙여줍니다.

이제 cmd를 열어서

cd C:\자신의 폴더위치

git add .현재 디렉토리에 있는 업데이트 된 파일을 전부 스테이징 영역으로 추가합니다.

git commit -m "image update"

그 전에 원격 저장소와 내 로컬을 연결해야 합니다.

git remote add origin(https://github.com/깃허브 아이디/블로그 주소)

git push origin master 안되면  git push -f origin master


### 일분 정도 기다리면 블로그에 올라가는걸 보실 수 있습니다!


```python

```
