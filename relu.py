class ReLuActivation:
    def __init__(self):
        self.forward_value = None # 기존에 들어온 값 저장, 역전파 시 미분값 계산을 위해 사용함

    def forward(self, forward_value: list):
        self.forward_value = forward_value
        outputs = [max(0.0, val) for val in forward_value]
        return outputs

    def backward(self, backward_value: list):
        next_backward_value = []
        for i in range(len(self.forward_value)):
            if self.forward_value[i] > 0:
                next_backward_value.append(backward_value[i])
            else:
                next_backward_value.append(0.0)

        return next_backward_value