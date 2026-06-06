from linear_layer import LinearLayer
from relu import ReLuActivation


# Linear Layer와 ReLu를 역어주는 레이어
class MlpModel:
    def __init__(self):
        self.linear_1 = LinearLayer(in_features=2, out_features=32)
        self.activation = ReLuActivation()
        self.linear_2 = LinearLayer(in_features=32, out_features=1)

    def forward(self, forward_value: list):
        forward_value = self.linear_1.forward(forward_value)
        forward_value = self.activation.forward(forward_value)
        forward_value = self.linear_2.forward(forward_value)
        return forward_value
    
    def backward(self, root_backward_value):
        backward_value = self.linear_2.backward(root_backward_value)
        backward_value = self.activation.backward(backward_value)
        backward_value = self.linear_1.backward(backward_value)
        return backward_value