# 1. Why Computer Vision?

# 2. Image Classification

### Image Classification이란?

사진을 컴퓨터에게 입력했을 때 사진에 나오는 객체(사람, 사물, 장소)를 판별하는 것

![gdscai3-1.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/kL8QK5XizG.png)

### semantic gap이란?

인간이 보는 이미지와 컴퓨터 보는 이미지 간의 차이

예를 들어 인간은 위에 있는 고양이 사진을 보면 바로 고양이라고 인식을 하지만 컴퓨터는 이미지를 보면 숫자 집합으로 본다. 이렇게 컴퓨터가 이미지를 보는 방식 때문에 image classification에서 여러 문제가 발생하게 된다.

![gdscai3-2.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/ykF4miRxvb.png)

카메라의 각도, 조명, 이미지의 크기에 따라 pixel이 달라짐, 물체가 가려지거나 배경색과 비슷해지면 인식이 어려움 등등의 이유로 컴퓨터가 이미지를 분류하는데 어려움을 갖는다

→ 좋은 image classfication model은 이러한 variation에 흔들리지 않아야한다.

# 3. KNN

KNN 알고리즘은 분류(classfication)알고리즘으로 유사한 특성을 가진 데이터는 유사한 범주에 속하는 경향이 있다는 가정하에 사용한다.

![gdscai3-3.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/E6S9Z8ME9r.png)

예를 들어 여기서 데이터가 주어져 있으면 빨간색인 세모 모양의 데이터는 초록색 그룹과 노란색 그룹 중에 노란색 그룹에 속한다. 

이렇게 주변의 가장 가까운 K개의 데이터를 보고 데이터가 속할 그룹을 판단하는 알고리즘이다. 

---

그 이후로 KNN 알고리즘으로 분류하는 거 정리해서 쓰고 싶었는데.. 

KNN 과제 리포트 제출이 좀 많이 급해가지고 

나중에 내용 보충해서 올리겠습니다..흑흑

여기가 그나마 가장 이해한 부분인데 말이죠..

나 진짜 이번 주  KNN 공부 짱열심히 했는데..

대충억울어쩌구

---

# 4. Linear Classification

Neural Network를 구성하는 가장 기본적인 요소

![gdscai3-5.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/thaj1IwuJS.png)

입력 이미지가 (가로, 세로, 채널)이 (32, 32, 3)이고 10개의 카테고리 중 하나를 분류하는 문제가 있다고 할 때 입력 이미지 (32*32*3)을 하나의 열벡터로 피면 (3072*1)이 된다. 이 x 와 W를 곱했을 때 10개의 스코어가 나와야 하니 W는 10*3072 → 결론적으로 10*1의 스코어 가져다 줌

![gdscai3-6.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/GciR2vrIki.png)

여기 ‘W’에 train data의 요약된 정보가 들어있다. → test 할 때 시간 단축

![gdscai3-4.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/VzwBSOyWay.png)

KNN 알고리즘은 모든 이미지를 pixel-wise 비교할 Training Set의 모든 image들을 기억하고 있고 또 연산을 해야 한다는 비효율성을 가지고 있었는데 이 단점을 해결해줄 model이 Linear Classification. 위의 그림처럼 벡터로 변환한 image에 weight를 곱해서 score가 가장 높은 카테고리로 image classification하는 방식이다. 

근데 솔직히 무슨 뜻인지 잘 모르겠다ㅎ

어려워요

아니 알겠는데 막상 하자니 어려워요..

# 5. Loss Function

Loss Function?

- 모델의 Output이 얼마나 틀렸는지 나타내는 척도 (loss가 작을 수록 좋음)
- 최종 목표는 오차를 최소로 만드는 Parameter를 구하는 것임(성능이 좋은 모델을 만들자!)

1주차에도 있었던 내용이지만

쨌든 다시 보자면 

Classifier가 이미지를 잘 분류하는지 정량적으로 평가하는 함수로 이미지 데이터인 x와 라벨 데이터인 y가 한 쌍인 데이터셋을 가지고 Classifier의 예측값과 라벨 데이터 y를 비교하여 모두 더한 뒤, 데이터의 개수로 나눈 값이 loss가 된다.

# 6. Multiclass SVM Loss

![gdscai3-7.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/MxL5qY8PXK.png)

sj : 정답이 아닌 클래스의 score

syi: 정답인 클래스의 score

상수 → safety margin

safety margin을 더해서 분류의 최소값을 올려서 분류 성능을 끌어올림(?)

만약 정답인 클래스의 score가 정답이 아닌 클래스의 score+safety margin보다 크다면 loss는 

아니라면 sj -syi + 1의 loss값을 가짐

이 값들을 모두 합한 값이 최종 loss

위의 방식으로 각 class 별 loss를 구하면 고양이는 2.9, 자동차는 0, 개구리는 12.9 값을 가지고 최종 L은 (2.9 + 0 + 12.9)/3

# 7. Regularization : Beyond Training Error

regularization?

overfitting을 방지하는 중요한 기법 중 하나

→ 모델을 구성하는 계수들이 학습 데이터에 너무 완벽하게 overfitting되지 않도록 정규화 요소(regularization term)을 더해주는 것

**L1 regularization**

- 매끄러운 그래프를 원할 때 쓰는 정규화
- 특정 요소만의 의존보다는 모든 요소의 전체적인 영향을 원하는 정규화
- 기존의 cost function 에 가중치의 크기가 포함되면서 가중치가 너무 크지 않은 방향으로 학습되도록 함
- weight의 수를 줄여 중요한 feature를 쉽게 선택(feature selection) 할 수 있음

![gdscai3-8.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/6Zpodcb9P6.png)

**L2 regularization**

- 분류기가 복잡하다고 느껴질 때 쓰는 정규화
- 기존의 cost function 에 가중치의 제곱을 포함하여 더함으로써 L1 Regularization 과 마찬가지로 가중치가 너무 크지 않은 방향으로 학습되게 함
- 가중치값에 0이 많도록 하여 보다 더 단순한 식 만들어줌

![gdscai3-9.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/gEJCB1Bjyv.png)

잉 어렵..

---

**참고 레퍼런스**

[cs231n 2강 Image classfication pipeline (tistory.com)](https://machine-learning-engineer.tistory.com/3#:~:text=Image%20Classification%EC%9D%B4%EB%9E%80%20%3A%C2%A0%20%EC%82%AC%EC%A7%84%EC%9D%84%20%EC%BB%B4%ED%93%A8%ED%84%B0%EC%97%90%EA%B2%8C%20%EC%9E%85%EB%A0%A5%ED%96%88%EC%9D%84%20%EB%95%8C%20%EC%82%AC%EC%A7%84%EC%97%90,%EC%A7%91%ED%95%A9%EC%97%90%20%EB%B6%88%EA%B3%BC%ED%95%98%EB%A9%B0%20%EA%B7%B8%20%EC%88%AB%EC%9E%90%EB%93%A4%EC%9D%98%20%EC%A7%91%ED%95%A9%EC%9D%84%20array%20%28%EB%B0%B0%EC%97%B4%29%EB%9D%BC%EA%B3%A0%20%EB%B6%80%EB%A5%B8%EB%8B%A4)

[CS231n study_lect2 (velog.io)](https://velog.io/@simon5287/CS231n-studylect2#:~:text=semantic%20gap%EC%9D%80%20image%20classification%EC%9D%B4%20%ED%95%B4%EA%B2%B0%ED%95%B4%EC%95%BC%20%ED%95%A0%20%EB%AC%B8%EC%A0%9C%EC%9D%98%20%EC%9B%90%EC%9D%B8%EC%9D%B4%EB%8B%A4.,%EC%BB%B4%ED%93%A8%ED%84%B0%EB%8A%94%20%EC%9D%B4%EB%AF%B8%EC%A7%80%EB%A5%BC%20%EB%B3%BC%20%EB%95%8C%20%EC%88%AB%EC%9E%90%20%EC%A7%91%ED%95%A9%EC%9C%BC%EB%A1%9C%20%EB%B3%B4%EA%B2%8C%20%EB%90%9C%EB%8B%A4.)

[2강. Image Classification (tistory.com)](https://hul980.tistory.com/20)

[이미지 분류 (Image Classification) | Reinventing the Wheel (heekangpark.github.io)](https://heekangpark.github.io/Stanford_cs231n/02-image-classification)

[K-최근접 이웃(K-Nearest Neighbor) 쉽게 이해하기 - 아무튼 워라밸 (hleecaster.com)](https://hleecaster.com/ml-knn-concept/)

[KNN Classifier와 Linear Classifier, 그리고 Loss Function | Sehyun Ryu](https://sehyunryu.github.io/posts/KNN_Classifier%EC%99%80_Linear_Classifier,_%EA%B7%B8%EB%A6%AC%EA%B3%A0_Loss_Function/)

[[CS231n 2강 정리] NN,K-NN, Linear Classification (tistory.com)](https://oculus.tistory.com/7#google_vignette)

[[CS231n] 2강 정리 (tistory.com)](https://moding.tistory.com/entry/CS231n-2%EA%B0%95-%EC%A0%95%EB%A6%AC)

[[CS231n] Lecture 3 : Loss Functions and Optimization (velog.io)](https://velog.io/@jeong_jaeyoon/CS231n-Lecture-3-Loss-Functions-and-Optimization)

[L1 & L2 loss/regularization · Seongkyun Han's blog](https://seongkyun.github.io/study/2019/04/18/l1_l2/)

[L1 Regularization & L2 Regularization (tistory.com)](https://hyebiness.tistory.com/11)

[딥러닝 용어 정리, L1 Regularization, L2 Regularization 의 이해, 용도와 차이 설명 (tistory.com)](https://light-tree.tistory.com/125)