from linear_layer import LinearLayer
from relu import ReLuActivation


# Linear Layer와 ReLu를 역어주는 레이어
class MlpModel:
    def __init__(self):
        self.linear_1 = LinearLayer(in_features=2, out_features=32)
        self.activation = ReLuActivation()
        self.linear_2 = LinearLayer(in_features=32, out_features=1)

    def forward(self, x: list):
        x = self.linear_1.forward(x)
        x = self.activation.forward(x)
        x = self.linear_2.forward(x)
        return x