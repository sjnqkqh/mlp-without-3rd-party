class ReLuActivation:
    def __init__(self):
        self.x = None # 기존에 들어온 값 저장, 역전파 시 미분값 계산을 위해 사용함

    def forward(self, x: list):
        self.x = x
        outputs = [max(0.0, val) for val in x]
        return outputs