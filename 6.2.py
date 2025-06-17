# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:17:58 2025

@author: User
"""
import numpy as np
"""
신경망 학습의 목적 : 손실 함수의 값을 최대한 낮추는 매개변수를 찾는 것 - 최적화optimization
최적의 매개변수 값을 찾는 단서로 매개변수의 기울기(미분)을 이용함 - 확률적 경사 하강법(SGD)
SGD의 단점과 다른 최적화 기법을 소개
"""

# 6.1.1 모험가 이야기
# 6.1.2 확률적 경사 하강법(SGD)
"""
W ← W - η * ∂L/∂W

W : 갱신할 가중치 매개변수
∂L/∂W : W에 대한 손실 함수의 기울기
η : 학습률(정해진 상수값. 0.01, 0.001 등)
"""


# 최적화를 담당하는 클래스를 분리해 구현하면 기능을 모듈화하기 좋다.
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# 6.1.3 SGD의 단점
"""
SGD는 단순하고구현이 쉽지만, 문제에 따라 비효율적일 때가 있다.
다음 함수의 최솟값을 구해보자
f(x, y) = 1/20 * x² + y²
각 점에서 함수의 기울기는 (x/10, 2y)로 y축 방향은 가파른데 x축 방향은 완만하다.
또 최솟값은 (0, 0)이지만 기울기 대부분은 그 방향을 가리키지 않는다.
따라서 SGD를 적용하면 y축으로 지그재그로 수렴한다.
SGD는 비등방성anisotropy 함수(방향에 따라 성질, 여기서는 기울기가 달라지는 함수)에서는
탐색 경로가 비효율적이다.
이러한 단점을 개선해주는 모멘텀, AdaGrad, Adam이라는 방법을 소개한다.
"""

# 6.1.4 모멘텀
"""
모멘텀Momentum : 물리에서의 운동량
v ← αv - η * ∂L/∂W
W ← W + v

W : 갱신할 가중치 매개변수
∂L/∂W : W에 대한 손실 함수의 기울기
η : 학습률
v : 속도. 기울기 방향으로 힘을 받아 물체가 가속되는 것을 나타냄
α : 마찰/저항에 해당(0.9)

마치 공이 바닥을 구르는 듯한 움직임을 보여준다.
"""


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# 6.1.5 AdaGrad
"""
학습률이 너무 작으면 학습 시간이 길어지고 너무 크면 발산한다.다
학습률을 정하는 효과적인 기술로 학습률 감소learning rate decay가 있다.
학습을 진행하면서 학습률을 점차 줄여나간다.

AdaGrad 방식은 각각의 매개변수에 맞춰 적응적으로 학습률을 조정하며 학습을 진행한다.
h ← h + ∂L/∂W ⊙ ∂L/∂W
W ← W - η *1/√h * ∂L/∂W

⊙ : 행렬의 원소별 곱셈
h는 기존 기울기를 제곱해서 누적하며, 매개변수 갱신에 1/√h를 곱해준다.
매개변수가 크게 갠신된 원소는 학습률이 낮아진다.

NOTE : AdaGrad는 과거의 기울기를 제곱하여 계속 더하기 때문에 학습을 진행할 수록
갱신 강도가 약해진다. 이 문제를 개선한 기법으로 RMSProp이 있다.
RMSProp에서는 먼 과거의 기울기는 서서히 잊고 새로운 기울기 정보를 크게 반영한다.
이를 지수이동평균Exponential Moving Average이라 하며 과거 기울기의 반영 규모를
기하급수적으로 감소시킨다.
"""


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}

            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            # 1e-7 : h[key]에 0이 있는 경우 0으로 나누는 것을 방지. 이 값도 설정 가능


# 6.1.6 Adam
"""
모멘텀과 AdaGrad의 두 기법을 융합한 기법. 2015년에 제안되었음.
또 하이퍼파라미터의 '편향 보정'이 진행됨

RMSprop과 Adam은 common/optizizer.py에서 구현해둠
"""

# 6.1.7 어느 갱신 방법을 이용할 것인가?
"""
각 방법의 그래프는 optimizer_compare_naive.py를 참고

모든 문제에서 항상 뛰어난 기법은 없다. 하이퍼파라미터를 어떻게 설정하느냐에 따라서도
결과가 달라진다.
"""

# 6.1.8 MNIST 데이터셋으로 본 갱신 방법 비교
"""
숫자 인식을 대상으로 네 기법을 비교한 그래프는 optimizer_compare_mnist.py 참고
각 층이 100개의 뉴런으로 구성된 5층 신경망에서 ReLU를 활성화 함수로 사용
인식률은 AdaGrad > Adam > Momentum >> SGD 순서였음
"""
"""
신경망 학습에서 특히 중요한 것이 가중치의 초깃값이다.
권장 초깃값에 대해 설명한다.
"""

# 6.2.1 초깃값을 0으로 하면?
"""
가중치 감소weight decay : 오버피팅을 억제해 범용 성능을 높이는 테크닉
가중치 매개변수의 값이 작아지도록 학습한다.
가중치를 작게 만들고 싶으면 초깃값도 최대한 작게 시작하는 것이 정석.
현재 예제에선 표준편차가 0.01인 정규분포를 사용했음

하지만 가중치를 0으로 설정하면 학습이 올바르게 이뤄지지 않는다.
(정확히는 가중치를 균일한 값으로 설정해서는 안된다.)
오차역전파법에서 모든 가중치의 값이 똑같이 갱신되기 때문.
가중치가 고르게 되는 것을 막기 위해서는 초깃값을 무작위로 설정해야 한다.
"""

# 6.2.2 은닉층의 활성화값 분포
"""
은닉층의 활성화값(활성화 함수의 출력 데이터)를 분석하면 중요한 정보를 얻을 수 있다.
활성화 함수로 시그모이드 함수를 사용하는 5층 신경망에 무작위로 생성한 입력 데이터를 흘리며
각 층의 활성화값 분포를 히스토그램으로 그려 가중치의 초깃값에 따라 은닉층 활성화값이
어떻게 변화하는지 확인한다. weight_init_activation_histogram.py참고
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 활성화값을 저장

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]
    w = np.random.randn(node_num, node_num) * 1
    a = np.dot(x, w)
    z = sigmoid(a)
    activations[i] = z

"""
각 층의 활성화값들이 0과 1에 치어쳐 분포되어 있다.
시그모이드는 출력이 0, 1에 가까워지면 미분값은 0에 가까워지기 때문에 역전파의 기욱기 값이
점점 작아지다가 사라진다. 이를 기울기 소실gradient vanishing이라고 한다.
층을 깊게 하는 딥러닝에서는 기울기 소실이 더 큰 문제가 될 수 있다.


가중치의 표준편차를 0.01로 바꾸면 활성화값은 0.5에 치우쳐져 있다.
활성화값이 치우치면 기울기 소실은 일어나지 않지만 표현력 관점에서 문제가 있다.
즉 다수의 뉴런이 거의 같은 값을 출력하고 있으니 뉴런을 여러 개 둔 의미가 없어진다.
따라서 활성화 값은 골고루 분포되어야 한다.

사비에르 글로로트와 요슈아 벤지오는 논문에서 Xavier 초깃값을 권장했으며 일반적인
딥러닝 프레임워크들이 표준적으로 사용하고 있다.
Xavier 초깃값 : 초깃값의 표준편차가 1/√n이 되도록 설정
n : 앞 층의 노드 수

Xavier 초깃값의 분포를 보면 층이 깊어지면 형태가 일그러지지만 앞선 방식들보다 넓게 분포한다.

NOTE : 층이 깊어지면 일그러지는 현상은 sigmoid 함수 대신 tanh(쌍곡선 함수)를 이용하면
개선된다. tanh 함수도 S자 곡선이지만 (0, 0.5)에서 대칭인 시그모이드와는 다르게 원점 대칭이다.
활성화 함수용으로는 원점에서 대칭인 함수가 바람직하다고 알려져 있다.
"""

# 6.2.3 ReLU를 사용할 때의 가중치 초깃값
"""
Xavier 초깃값은 활성화 함수가 선형인 것을 전제로 한다. 좌우 대칭인 함수는 중앙 부근이 선형이라
볼 수 있어서 Xavier 초깃값이 적당하다. 반면 ReLU를 이용할 때는 이에 특화된 초깃값을 권장한다.
이 초깃값을 발견자 카이밍 히의 이름을 따 He 초깃값이라고 한다.

He 초깃값 : 초깃값의 표준편차가 √(2/n)이 되도록 설정(Xavier의 2배)
n : 앞 층의 노드 수

ReLU는 음수가 0이므로 더 넓게 퍼트리기 위해 계수가 2배여야 한다고 해석할 수 있다.

ReLU를 사용한 경우, 0.01, xavier, He를 사용했을 때
0.01은 활성화 값이 아주 작아 기울기 소실이 발생하며
Xavier는 층이 깊어질 수록 치우침이 커진다.
He는 모든 층에서 균일하게 분포되었다.
"""

# 6.2.4 MNIST 데이터셋으로 본 가중치 초깃값 비교
"""
weight_init_compare.pt 참고.
층별 뉴런 수가 100개인 5층 신경망에서 ReLU함수를 사용.
std=0.01일 경우 학습이 전혀 이뤄지지 않는다.(순전파 때 너무 0으로 밀집한 작은 값이 흐르기 때문.
그로인해 역전파 때의 기울기도 작아져 가중치가 거의 갱신되지 않는다.)
Xavier와 He의 경우는 순조롭게 학습되고 있다.(진도는 He가 더 빠르다.)
"""