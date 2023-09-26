# Semantic Segmentation

: 이미지 내에서 픽셀 단위로 객체를 분류해내는 작업

ex) 이미지에서 픽셀을 사람, 자동차, 비행기 등의 물리적 단위로 분류하는 방법

- Classification (분류)
    - 인풋에 대해 하나의 레이블을 예측하는 작업
    - AlexNet, ResNet, Xception
- Localization/Detection (모델)
    - 물체의 레이블을 예측하면서 그 물체가 어디에 있는지 정보를 제공 (ex 물체가 있는 곳에 네모 그리기)
    - YOLO, R-CNN
- Segmentation (분할)
    - 모든 픽셀의 레이블을 예측
    - FCN, SegNet, DeepLab

![gdscai2-1.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/ATa4657PsK.png)

### CNN

- 일반적으로 Convolution layer와 fully connected layer들로 이루어짐
- 항상 입력 이미지를 네트워크에 맞는 고정된 사이즈로 작게 만들어서 입력

### FCN

- Convolution layer만을 사용
- CNN에서는 모델 뒤쪽에서 Fully Connected layer가 나오는데 FCN에서는 FC Layer 대신 1*1 Convolutional layer 사용

![gdscai2-2.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/RJi9Dia966.png)

1. Feature을 추출하는 Convolution 단계
2. 뽑아낸 future에 대해 pixelwise prediction 단계
3. classification을 한 뒤 각 원래의 크기로 만들기 위한 Unsampling 단계
    - 여러 단계의 (convolution + pooling)을 거치게 되면, feature-map의 크기가 줄어든다.
    - 픽셀 단위로 예측을 하려면 줄어든 feature-map의 결과를 다시 키워야한다.
    
    ![gdscai2-3.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/osMBG8AkCv.png)
    
    - 이전의 Pooling Layer의 값을 가져와 더한다
    - 아래를 보면 ground truth와 비교해 FCN-32s로 얻은 segmentation map은 많이 뭉뚱그려져 있고 디테일하지 못하다
    - 더 디테일한 segmentation map을 얻기 위해, skip combining
        
        ![gdscai2-4.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/lQqjiPFzGL.png)
        
    - FCN-16s는 앞쪽 블록에서 얻은 예측 결과 맵과, 2배로 unsampling한 맵을 더한 후, 한 번에 16배로 unsampling을 해주어 얻는다
    - 여기서 한 번 더 앞쪽 블록을 사용하면 FCN-8s
    - FCN-32s > FCN-16s > FCN-8s 순으로 결과가 좋아지는 것을 볼 수 있다
        
        ![gdscai2-5.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/zGKSVJjbhW.png)
        
        ![gdscai2-6.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/apWxNwZYUm.png)
        

# Object Detection

이미지 내에서 물체의 위치와 그 종류를 찾아내는 것

Object Detection은 크게 1-stage Detector와 2-stage Detector로 구분할 수 있다

- 1-stage Detector
    - 두 가지 task를 동시에 행하는 방법
    - ex) R-CNN, Fast R-CNN, Faster R-CNN
- 2-stage Detector
    - 두 문제를 순차적으로 행하는 방법
    - ex) YOLO 계열, SSD 계열

![gdscai2-7.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/XN5oxZTwMD.png)

### R-CNN

![gdscai2-8.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/umeCtdOJ4o.png)

![gdscai2-9.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/H8mk8L0Qr4.png)

1. 이미지를 입력하고
2. 이미지에서 selective search를 통해 2000개의 bounding box 후보를 생성
3. bounding box 후보 영역의 이미지 크기를 resize하여 CNN 모델에 넣고 feature를 추출
4. CNN 모델에서 얻은 feature를 기반으로 SVM을 통해 해당 영역에 대한 예측 수행

전체 task를 두 단계로 나눌 수 있는데, 위의 1,2번은 물체의 위치를 찾는 Region Proposal, 3,4번은 물체를 분류하는 Region Classification이다. (2-stage Detector)

(사실 너무 어려워 이렇게 이어지는 것이 맞는지 확신은 없다..)

- **Region Proposal**
    - 주어진 이미지에서 물체가 있을법한 위치를 찾는 것
    - Selective Search라는 알고리즘 적용해 물체가 있을법한 박스를 찾는다
        - (Selective Search → 주변 픽셀 간의 유사도를 기준으로 Segmentation 만들고 이를 기준으로 물체가 있을 법한 박스를 추론)
- **Feature Extraction**
    - 미리 이미지 넷으로 학습된 CNN을 가져와서, Object Detection용 데이터 셋으로 fine tuning 한 뒤, selective search 결과로 뽑힌 이미지들로부터 특징 벡터 추출
- **Classification**
- **Bounding Box Regression**

### SPPNet

R-CNN의 단점인 고정된 입력 이미지 사이즈, 중복되는 CNN 계산을 개선한 네트워크 

(기존의 CNN 아키텍쳐들은 모두 입력 이미지가 고정되어야 했기 때문에 신경망을 통과시키기 위해서는 이미지를 고정된 크기로 크롭하거나 비율을 조정해야 함 → 물체의 일부분이 잘리거나 본래의 생김새가 달라짐 → 상관없이 Conv layer 통과시키고, FC layer 통과 전에 피쳐 맵 들을 동일한 크기로 조절해주는 pooling 적용하자 → 입력 이미지의 크기를 조절하지 않은 채로 컨볼루션 진행하면 원본 이미지의 특징을 고스란히 간직한 피쳐 맵 얻을 수 있음)

![gdscai2-10.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/OLQTEn3o6P.png)

1. 전체 이미지를 pre-train한 네트워크에 주입
2. Selective search를 통해 찾은 크기와 비율이 다른 Rol에 SPP 적용하여 고정된 사이즈의 feature 추출
3. FC layer 통과
4. SVM classifier 학습 & Bounding box regression

![gdscai2-11.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/7jtyHrtlCf.png)

### Fast R-CNN

CNN 특징 추출부터 classification, bounding box regression까지 모두 하나의 모델에서 학습

![gdscai2-12.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/heCqjUo0FH.png)

1. 전체 이미지를 미리 학습된 CNN을 통과시켜 피쳐맵 추출
2. Selective Search를 통해 찾은 각각의 Rol에 대해 Rol Pooling 진행 → 고정된 크기의 feature vector 얻음
3. 추출된 feature vector는 fully connected layer들을 통과한 뒤, 두 개의 브랜치로 나눔 (softmax, bbox regressor)
4. Softmax branch는 softmax를 통과시켜 객체의 class를 분류(SVM 사용 x), bbox regressor branch는 bounding box regression을 통해 selective search로 찾은 박스의 위치 조정

### YOLO

- 한번만 보고 처리를 해주는 Object Detection 모델
- 입력된 이미지를 일정 분할로 그리드한 다음, 신경망을 통과해서 바운딩 박스와 클래스 예측을 생헝해서 최종 감지 출력을 결정
- 기존에 region proposal, classification 이렇게 두 단계로 나누어서 진행하던 방식에서 region proposal 단계를 제거하고 한번에 Object Detection을 수행하는 구조를 가짐

![gdscai2-13.PNG](https://i.esdrop.com/d/f/AfOYjCl4ON/4zfEWDdcgc.png)

1. 가로 세로 동일한 그리드 영역으로 나누고
2. 각 그리드 영역에 대해서 어디에 사물이 존재하는지 바운딩 박스와 박스에 대한 신뢰도 점수 예측(신뢰도 높을 수록 박스 굵음) & 어떤 사물읹이에 대한 classification 작업 동시 진행
3. 굵은 박스들만 남김 (NMS 알고리즘)

# AutoGrad & Optimizer

### nn.Module

- 딥러닝을 구성하는 layer의 기초가 되는 classs
- Input, Output, Forward, Backward를 정의
- 학습의 대상이 되는 parameter(tensor)를 정의

### nn.Parameter

- Tensor 객체의 상속 객체
- nn.Module 내에 attribute의 경우 required_grad=True로 지정되어 학습 대상이 된다.

```python
class MyLinear(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
    	super().__init__()
        # 예를 들어, input이 (5,7)이고 output이 (5,12)인 경우 weight는 (7,12)인데,
        # 이 때 in_features가 7, out_features가 12
        self.in_features = in_features
        self.out_features = out_features
        
		# weights를 지정해주는 부분, 실제로 직접 입력할 일은 x
    	self.weights = nn.Parameter(torch.randn(in_featrues, out_features))
        # Parameter가 아닌 Tensor로 지정해주어도 방식은 동일
        # 그러나 autograd 지정이 어렵다
        
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x:Tensor): # y_hat 부분 (xw+b)
    	return x @ self.weights + self.bias
```

### Backward

- Layer에 있는 Parameter들의 미분을 수행
- Forward의 결과값과 실제값의 차이에 대해 미분 수행해서
- 해당 값을 기준으로 Parameter을 업데이트

# Pytorch Dataset, Model Save

아직 공부를 안했습니다..

너무 어려워요..

# Transfer Learning

- 이미 잘 훈련된 모델이 있고, 해당 모델과 유사한 문제를 해결할 때 사용한다

여기 이 [블로그](https://www.notion.so/b5cfe12a77c949ebbeb22f59a64ef788?pvs=21)가 되게 재밌게 설명이 되어있어서 링크 첨부하는 것을 마지막으로 마무리해야겠다..ㅎ

---

**참고 레퍼런스**

[1편: Semantic Segmentation 첫걸음!. Semantic Segmentation이란? 기본적인 접근 방법은? | by 심현주 | Hyunjulie | Medium](https://medium.com/hyunjulie/1%ED%8E%B8-semantic-segmentation-%EC%B2%AB%EA%B1%B8%EC%9D%8C-4180367ec9cb)

[[딥러닝] 이미지 세그멘테이션(Image Segmentation) (velog.io)](https://velog.io/@dongho5041/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%84%B8%EA%B7%B8%EB%A9%98%ED%85%8C%EC%9D%B4%EC%85%98Image-Segmentation)

[CNN을 활용한 주요 Model - (4) : Semantic Segmentation (reniew.github.io)](https://reniew.github.io/18/)

[딥러닝 Segmentation(3) - FCN(Fully Convolution Network) (velog.io)](https://velog.io/@cha-suyeon/%EB%94%A5%EB%9F%AC%EB%8B%9D-Segmentation3-FCNFully-Convolution-Network)

[semantic segmentation의 목적과 대표 알고리즘 FCN의 원리 by bskyvision.com](https://bskyvision.com/entry/semantic-segmentation%EC%9D%98-%EC%9D%98%EB%AF%B8%EC%99%80-%EB%AA%A9%EC%A0%81)

[딥러닝 Object Detection 모델 살펴보기(1) : R-CNN (RCNN) 논문 리뷰 : 네이버 블로그 (naver.com)](https://m.blog.naver.com/baek2sm/222782537693)

[R-CNN 을 알아보자 (velog.io)](https://velog.io/@whiteamericano/R-CNN-%EC%9D%84-%EC%95%8C%EC%95%84%EB%B3%B4%EC%9E%90)

[갈아먹는 Object Detection [1] R-CNN (tistory.com)](https://yeomko.tistory.com/13)

[[간단 설명] Object Detection 기초 : RCNN, SPPNet, Fast RCNN, Faster RCNN, YOLO / 딥러닝을 이용한 객체 검출의 기초 — CV DOODLE (tistory.com)](https://mvje.tistory.com/85#google_vignette)

[[CS231N] Object Detection의 종류 (R-CNN, SPPnet, Fast R-CNN, Faster R-CNN, YOLO까지) (tistory.com)](https://ok-lab.tistory.com/41)

[갈아먹는 Object Detection [2] Spatial Pyramid Pooling Network (tistory.com)](https://yeomko.tistory.com/14)

[갈아먹는 Object Detection [3] Fast R-CNN (tistory.com)](https://yeomko.tistory.com/15)

[갈아먹는 Object Detection [5] Yolo: You Only Look Once (tistory.com)](https://yeomko.tistory.com/19)

[YOLO Object Detection, 객체인식 - 개념, 원리, 주목해야 할 이유, Use Case - 데이터헌트 (thedatahunt.com)](https://www.thedatahunt.com/trend-insight/guide-for-yolo-object-detection)

[[Pytorch] Autograd and Optimizer (velog.io)](https://velog.io/@khs0415p/Pytorch-Autograd-and-Optimizer)

[[인공지능] PyTorch AutoGrad & Optimizer (velog.io)](https://velog.io/@ausrn731/%EC%9D%B8%EA%B3%B5%EC%A7%80%EB%8A%A5-PyTorch-AutoGrad-Optimizer)

[[인턴일지] Transfer Learning (전이 학습) 이란? : 네이버 블로그 (naver.com)](https://blog.naver.com/PostView.nhn?blogId=flowerdances&logNo=221189533377)

[[Deep Learning]Transfer Learning(전이학습)이란? - Meaningful AI (meaningful96.github.io)](https://meaningful96.github.io/deeplearning/Transferlearning/)