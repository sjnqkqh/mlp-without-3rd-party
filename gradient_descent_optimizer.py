class GradientDescentOptimizer:
    def __init__(self, layer_list, lr=0.1):
        self.layer_list = layer_list
        self.lr = lr

    def step(self):
        for layer in self.layer_list:
            for i in range(layer.out_features):
                for j in range(layer.in_features):
                    # 가중치 조절
                    layer.weights[i][j] -= self.lr * layer.d_weight[i][j]

            for i in range(layer.out_features):
                layer.bias[i] -= self.lr * layer.d_bias[i]
