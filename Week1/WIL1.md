# 1. AI, ML, DL ?

**DL < ML < AI**

- AI 인공지능
    - 인간의 능력을 인공적으로 구현
- ML 머신러닝
    - 데이터로부터 규칙을 학습하는 AI의 하위 분야
- DL 딥러닝
    - Neural network를 기반으로 한 ML의 하위 분야

# 2. Deep Learning Component

### Data

### Model

### Loss function

- 실제 값과 예측 값의 차이를 측정하는 측도
- 만약 실제 값과 예측 값이 차이가 크면 오차가 크다
    - 이 오차가 커지면 손실함수 값이 커짐
    - ⇒ 손실함수를 최소화하는 적절한 모델을 찾는 것이 머신러닝/딥러닝의 목표
    
1. 회귀(Regression) 손실함수
    1. 연속형 target 값을 예측하는 분석하는 기법
    2. MSE : 실제값과 예측값의 차이인 오차의 제곱합의 평균
2. Classification
3. Probabilicstic

### Optimization (최적화)

- 손실 함수 (Loss Function) 값을 최소화하는 파라미터를 구하는 과정
    
    ![gdscai1-3.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/xDaCqAbr6I.png)
    
- Gradient Descent Method (경사하강법)
    - 손실 함수 그래프에서 값이 가장 낮은 지점으로( = 손실 함수의 최솟값) 걍사를 타고 하강하는 기법
- 최적화 용어 정리 (아래에서 자세히 살펴볼 예정..)
    - Generalization
        - Train data에 대한 성능과 Test data에 대한 성능 차이
        
        ![gdscai1-4.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/D2ikd3BFCd.png)
        
    - Under-fitting vs. over-fitting
        - 학습 데이터에 대해서는 성능이 나오지만, Test dadta에 대해서는 성능이 떨어질 때 모델이 overfitting이 되었다고 표현
        - 모델의 복잡도가 떨어지거나 학습이 잘 되지 못해서 학습 데이터에 대해서도 성능이 떨어지는 경우 underfitting되었다고 표현
        
        ![gdscai1-5.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/VWMTBgHMFt.png)
        
    - Cross validation (교차 검증)
        - 데이터 셋을 여러 조각으로 나누어 학습과 validation에 번갈아 사용하는 기법
        
        ![gdscai1-6.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/sApyn6k6h0.png)
        
    - Bias-variance tradeoff
        
        ![gdscai1-7.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/FuJqgNfhUU.png)
        
    - Bootstrapping
        - 가진 데이터의 일부만 사용해서 모델을 생성하는 것을 여러 번 반복
        
        ![gdscai1-8.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/MTWETHU5Mk.png)
        

# 3. Neural Network

인공 신경망은 선형함수와 비선형 함수로 층을 쌓고 있다

**신경망을 쌓을 때 선형함수만 쌓는다면?**

3개를 다음과 같이 쌓는다고 가정하면 (x는 입력값)

`f1 = W1*x + b1`

`f2 = W2*f1 + b2`

`f3 = W3*f2 + b3`

f1과 f2를 f3에 대입해보면

`f3 = W3*(W2*(W1*x + b1) + b2) + b3`

`f3 = W3*((W2*W1*x + W2*b1) + b2) + b3`

`f3 = W3*W2*W1*x + W3*W2*b1 + W3*b2 + b3`

W와 b는 상수이기 때문에 거대한 선형 함수처럼 보인다 → 3개의 선형함수의 층은 아래와 같은 한 개의 선형함수로

`f3 = W4*x + b4`

`(W4=W1*W2*W3, b4=(W3*W2*b1)+(W3*b2)+b3)`

⇒ 신경망에서는 선형함수 다음에 꼭 비선형 함수를 넣어준다

# 4. Nonlinear Function

딥러닝 네트워크에서 노드에 들어오는 값들에 대해 바로 다음 레이어로 전달하지 않고 비선형 함수를 주로 통과시킨 후 전달

### 1. Sigmoid 함수

- 0 ~ 1 사이의 값으로 제한되며,
- 값이 무한히 커지면 1에 수렴하고
- 값이 무한히 작아지면 0에 수렴한다
- 경사 하강법을 적용하여 가중치를 업데이트할 때 기울기가 0에 가까워져 소실되는 기울기 소실 현상이 발생할 수 있음

![gdscai1-9.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/eI5mLuPC4B.png)

### 2. ReLU 함수

- CNN 등 딥러닝 모델에 자주 쓰이는 비선형함수
- f(x) = max(0,x)
- x > 0이면 기울기가 1인 직선
- x < 0이면 0이 된다
- Sigmoid, tanh에 비해 연산이 빠르다
- Sigmoid, tanh에 비해 Gradient가 출력층과 멀리 있는 Layer까지 전달 ㄱㄴ

![gdscai1-11.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/3Qlzra4se2.png)

### 3. Hyperbolic Tangent 함수

- 탄젠트 함수를 -90도 회전시킨 모양의 그래프
- -1 ~ 1 사이의 값으로 제한되며, 음수의 무한대로 가면 -1에 수렴하고
- 양수의 무한대로 가면 1에 수렴

![gdscai1-10.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/freU4X1Ur8.png)

# 5. Multi-Layer Perceptron (다층 퍼셉트론)

다층 퍼셉트론

![gdscai1-12.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/AGPrezuGCN.png)

- 층이 2개 이상 존재하는 신경망
- 은닉층
    - 입력층과 출력층을 제외한 층
    - 은닉층이 1개만 있는 다층 퍼셉트론 → 얇은 신경망
    - 은닉층이 2개 이상 → 깊은 신경망
        - 이 깊은 신경망을 학습시키는 것이 딥러닝

# 6. Generalization

### Generalization

![gdscai1-4.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/D2ikd3BFCd.png)

일반화(Generalization)

- 학습을 시키면 Iteration(반복)이 지나감에 따라 Training data에 대한 Training error가 줄어든다
- training error가 0에 가까워져도 항상 원하는 최적값에 도달했다는 보장은 없다

### Under-fitting vs Over-fitting

![gdscai1-5.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/VWMTBgHMFt.png)

- Overfitting은 학습 데이터(Training Set)에 대해 과하게 학습된 상황
    - → 학습 데이터 이외의 데이터에 대해서는 모델이 잘 동작하지 않는다
    - Overfitting이 발생했을 때 사용할 수 있는 해결책
        - Model Capacity 낮추기
        - Dropout (학습을 할 때 일부 뉴런을 끄고 학습)
        - L1/L2 정규화
        - 학습 데이터 늘리기 (data augmentation)
- Underfitting은 이미 있는 Train set도 학습을 하지 못한 상태
    - 발생 이유
        - 학습 반복 횟수가 너무 적음
        - 데이터의 특성에 비해 모델이 간단
        - 데이터의 양이 너무 적음

### Cross Validation

![gdscai1-6.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/sApyn6k6h0.png)

- Trainind data와 Test data를 나누어 주어지며
- Training data와 Validation data를 나누어서 Training data를 주고 학습시키려는 것이 목적이다

### Ensemble

- 하나의 모델만을 학습시키는 것이 아닌 여러 모델을 학습시켜 결합하는 방식으로 문제 처리
- 장점
    - 과적합(Overfitting) 개선
    - 예측 정확도 향상

**Bagging**

- 여러 모델에서 나온 결과값을 평균값 또는 중간값을 활용하는 방식
- variance를 감소시키는 역할
- 병렬적으로 학습

**Boosting**

- 예측이 틀린 데이터에 대해 올바르게 예측하도록 다음 분류기에 가중치 부여하며 학습과 예측 진행
- 각각의 모델이 연속적으로 학습
- bias를 감소시키는 역할
- 순차적으로 학습

### Regularization

generalization을 잘 되게 하고 싶음 → 학습에 반대되도록 어떤 규제를 걸음 (학습을 방해)

- Early stopping
    - training에 활용하지 않은 데이터셋에 지금까지 학습된 모델을 평가해보고 loss를 보고 loss가 어느 순간부터 커지기 시작하면 그때 멈춤
    
    ![gdscai1-13.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/RotaClpNFY.png)
    
- Parameter norm permalty
    - neural network 파라미터가 너무 커지지 않게 하는 것
    - 네트워크를 학습할 때 네트워크 weight 숫자들이 작으면 작을 수록 좋음
    - → 부드러운 함수일수록 generalization performance가 좋을 것임
    
    ![gdscai1-14.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/pNCnFpe7gF.png)
    
- Data augmentation
    - 데이터셋이 어느 정도 이상 커지게 되면 기존의 머신러닝에서 활용하는 방법론들이 많은 수의 데이터를 표현할만한 표현력이 부족
    - but 데이터가 한정적 → Data Augmentation함
    
    ![gdscai1-15.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/lmfWhSEiTa.png)
    
- Noise robustness
    - 입력데이터에 노이즈 집어넣기
    
    ![gdscai1-16.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/z801UBJxLq.png)
    
- Dropout
    - neural network의  weight를 일반적으로 0으로 바꾸는 것
    - 특정 확률로 뉴런의 출력을 0으로 만듦
    - 이때 train 단계에서만 dropout이 적용되고, test 단계에서는 사용x
- Label smoothing

# 7. Convolutional Neural Networks(CNN)

CNN은 image를 분류하기 위해 개발된 Network → 이미지에 최적화되어 있음

![gdscai1-17.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/b18tM4nD3x.png)

**Convolutional Layer (합성곱계층)**

- 입력된 이미지의 특징(화소, 구성)을 반영해서 새로 처리된 이미지를 생성하는 계층
- 이미지에 대해 필터연산 수행

**Pooling Layer (풀링 계층)**

- 합성곱 계층을 통과한 이미지의 대표적인 픽셀만 뽑는 역할을 함 (가장 강한 신호만 전달..?)

**Fully connected layer**

---

**참고 레퍼런스**

[손실함수(Loss function)의 통계적 분석 (tistory.com)](https://hyewon328.tistory.com/entry/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98Loss-function-%ED%8C%8C%ED%97%A4%EC%B9%98%EA%B8%B0)

[[Deep Learning] 최적화 개념과 경사 하강법(Gradient Descent) (tistory.com)](https://heytech.tistory.com/380#:~:text=%EC%B5%9C%EC%A0%81%ED%99%94%20%EA%B0%9C%EB%85%90%20%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EB%B6%84%EC%95%BC%EC%97%90%EC%84%9C%20%EC%B5%9C%EC%A0%81%ED%99%94%28Optimization%29%EB%9E%80%20%EC%86%90%EC%8B%A4%20%ED%95%A8%EC%88%98%28Loss%20Function%29,%ED%95%99%EC%8A%B5%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A5%BC%20%EC%9E%85%EB%A0%A5%ED%95%98%EC%97%AC%20%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%20%EA%B5%AC%EC%A1%B0%EB%A5%BC%20%EA%B1%B0%EC%B3%90%20%EC%98%88%EC%B8%A1%EA%B0%92%28%28hat%7By%7D%29%29%EC%9D%84%20%EC%96%BB%EC%8A%B5%EB%8B%88%EB%8B%A4.)

[week2 - Optimization 최적화 기법 (tistory.com)](https://my-coding-footprints.tistory.com/101)

[[부스트캠프 AI Tech / Day12] 딥러닝 기초 Optimization | Always Awake Sally (bsm8734.github.io)](https://bsm8734.github.io/posts/bc-d012-1-dlbasic-optimization/)

[활성화 함수(activation function) (velog.io)](https://velog.io/@xdfc1745/%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98activation-function#:~:text=%ED%99%9C%EC%84%B1%ED%99%94%20%ED%95%A8%EC%88%98%20%EB%94%A5%EB%9F%AC%EB%8B%9D%20%EB%84%A4%ED%8A%B8%EC%9B%8C%ED%81%AC%EC%97%90%EC%84%9C%20%EB%85%B8%EB%93%9C%EC%97%90%20%EB%93%A4%EC%96%B4%EC%98%A4%EB%8A%94%20%EA%B0%92%EB%93%A4%EC%97%90%20%EB%8C%80%ED%95%B4,%EA%B0%92%EC%9D%84%20%EA%B3%84%EC%86%8D%20%EC%A0%84%EB%8B%AC%ED%95%98%EB%A9%B4%20%EA%B3%84%EC%86%8D%20%EB%98%91%EA%B0%99%EC%9D%80%20%EC%84%A0%ED%98%95%ED%95%A8%EC%88%98%20%EC%8B%9D%EC%9D%B4%20%EB%90%9C%EB%8B%A4.)

[활성화 함수 (Activation Functions) (velog.io)](https://velog.io/@yoonene/%ED%99%9C%EC%84%B1%ED%99%94-%ED%95%A8%EC%88%98-Activation-Functions)

[신경망 (2) - 다층 퍼셉트론(Multi Layer Perceptron)과 활성화 함수(Activation function) (tistory.com)](https://yhyun225.tistory.com/21)

[4. Optimization의 주요 용어 이해 (velog.io)](https://velog.io/@leejy1373/Deep-learning-%EA%B8%B0%EC%B4%88-4.-%EC%B5%9C%EC%A0%81%ED%99%94Optimization%EC%9D%98-%EC%A3%BC%EC%9A%94-%EC%9A%A9%EC%96%B4-%EC%9D%B4%ED%95%B4)

[Overfitting과 Underfitting 정의 및 해결 방법 (tistory.com)](https://22-22.tistory.com/35)

[[Deep Learning] 앙상블 학습(Ensemble Learning) (tistory.com)](https://xangmin.tistory.com/137#:~:text=%EC%95%99%EC%83%81%EB%B8%94%20%ED%95%99%EC%8A%B5%20%28Ensemble%20Learning%29%EC%9D%B4%EB%9E%80%20%ED%95%98%EB%82%98%EC%9D%98%20%EB%AA%A8%EB%8D%B8%EB%A7%8C%EC%9D%84%20%ED%95%99%EC%8A%B5%EC%8B%9C%EC%BC%9C%20%EC%82%AC%EC%9A%A9%ED%95%98%EC%A7%80,%EB%AA%A8%EB%8D%B8%EC%9D%84%20%EC%A1%B0%ED%95%A9%ED%95%98%EC%97%AC%20%EC%9D%BC%EB%B0%98%ED%99%94%20%28generalization%29%20%EC%84%B1%EB%8A%A5%EC%9D%84%20%ED%96%A5%EC%83%81%ED%95%A0%20%EC%88%98%20%EC%9E%88%EB%8B%A4.)

[[데이터분석] 머신러닝 앙상블기법 개념 및 Bagging vs Boosting 차이 : 네이버 블로그 (naver.com)](https://m.blog.naver.com/yjhead/222116788833)

[Convolutional Neural Network(CNN) _기초 개념 (tistory.com)](https://han-py.tistory.com/230)