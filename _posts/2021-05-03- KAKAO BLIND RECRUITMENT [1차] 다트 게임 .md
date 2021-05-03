---
title: "KAKAO BLIND RECRUITMENT[1차] 다트 게임_프로그래머스_파이썬"
excerpt: "find all, compile"
date: '2021-05-03'
categories : programmers
tags : [programmers, python]
use_math : true
---



* 사전지식
    * findall
    * compile
* findall : 정규식과 매치되는 모든 문자열(substring)을 리스트로 돌려준다.
* 문자만 찾고 싶을 때


```python
import re
string = '1D2S#10S'
```


```python
re.findall("[^0-9']+", string)
```




    ['D', 'S#', 'S']



* 숫자만 찾고 싶을 때


```python
re.findall("\d+", string) 
```




    ['1', '2', '10']



* re.compile의 결과로 돌려주는 객체 p(컴파일된 패턴 객체)를 사용하여 그 이후의 작업을 수행


```python
dartResult = '1D2S#10S'
p = re.compile('(\d+)([SDT])([*#]?)') 
dart = p.findall(dartResult)
dart
```




    [('1', 'D', ''), ('2', 'S', '#'), ('10', 'S', '')]



* 숫자, S,D,T, *또는 # 마지막에 ?가 없으면 *과 #이없는 1번과 3번이 출력이 되지 않는다.


```python
dartResult = '1D2S#10S'
p = re.compile('(\d+)([SDT])([*#])') 
dart = p.findall(dartResult)
dart
```




    [('2', 'S', '#')]



* 나의 풀이


```python
import re
def solution(dartResult):
    answer = [0,0,0]
    string = re.findall("[^0-9']+", dartResult)
    count = re.findall("\d+", dartResult) 
    for i in range(len(string)):
        par = string[i]
        for j in range(len(par)):
            num = 0
            if j ==0:
                if par[j] == 'S':
                    num += int(count[i])*1
                    answer[i] += num
                elif par[j] == 'D':
                    num += int(count[i])**2
                    answer[i] += num
                elif par[j] == 'T':
                    num += int(count[i])**3
                    answer[i] += num
            else :
                if par[j] == '*':
                    if  i ==0:
                        answer[i] =  answer[i]*2
                    else :
                        answer[i-1] =  answer[i-1]*2
                        answer[i] =  answer[i]*2        
                elif par[j] == '#':
                    answer[i] =  answer[i]*-1
    return sum(answer)
print(solution('1D2S#10S'))
```

    9


* 다른 사람 풀이


```python
import re


def solution(dartResult):
    bonus = {'S' : 1, 'D' : 2, 'T' : 3}
    option = {'' : 1, '*' : 2, '#' : -1}
    p = re.compile('(\d+)([SDT])([*#]?)')
    dart = p.findall(dartResult)
    for i in range(len(dart)):
        if dart[i][2] == '*' and i > 0:
            dart[i-1] *= 2
        dart[i] = int(dart[i][0]) ** bonus[dart[i][1]] * option[dart[i][2]]

    answer = sum(dart)
    return answer
print(solution('1D2S#10S'))
```

    9


* References
   * https://wikidocs.net/4308
