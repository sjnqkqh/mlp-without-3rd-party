import random

random.seed(42)


class LinearLayer:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # 초기 가중치 세팅
        self.weights = []
        for _ in range(out_features):
            row = [random.uniform(-0.1, 0.1) for _ in range(in_features)]
            self.weights.append(row)
        self.bias = [random.uniform(-0.1, 0.1) for _ in range(out_features)]

    def forward(self, x):
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