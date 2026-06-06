import random
import math

random.seed(42)


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # 초기 가중치 세팅 (He 초기화)
        std = math.sqrt(2.0 / in_features)
        self.weights = []
        for _ in range(out_features):
            row = [random.gauss(0.0, std) for _ in range(in_features)]
            self.weights.append(row)
        self.bias = [0.0 for _ in range(out_features)]

        self.forward_value = None  # 순전파 밸류 기록
        self.d_weight = []
        self.d_bias = []

    def forward(self, x):
        self.forward_value = x

        outputs = []
        for i in range(self.out_features):
            # 가중치 곱하기
            node_sum = 0
            for j in range(self.in_features):
                node_sum += x[j] * self.weights[i][j]

            # bias 더하기
            node_sum += self.bias[i]
            outputs.append(node_sum)

        return outputs

    def backward(self, backward_value):
        self.d_bias = [gradient for gradient in backward_value]
        self.d_weight = []
        for i in range(self.out_features):
            row_gradient = []
            for j in range(self.in_features):
                # 가중치를 수정하기 위한 미분
                row_gradient.append(backward_value[i] * self.forward_value[j])
            self.d_weight.append(row_gradient)

        next_backward_value = [0.0] * self.in_features
        for i in range(self.out_features):
            for j in range(self.in_features):
                next_backward_value[j] += backward_value[i] * self.weights[i][j]

        return next_backward_value