import math


class BceLossWithLogits:
    def __init__(self):
        self.logit = None
        self.y = None
        self.prob = None

    def forward(self, logit_list, y_list):
        self.logit = logit_list[0]
        self.y = y_list[0]

        # 시그모이드 연산
        if self.logit >= 0:
            self.prob = 1.0 / (1.0 + math.exp(-self.logit))
        else:
            exp_t = math.exp(self.logit)
            self.prob = exp_t / (1.0 + exp_t)

        # 손실값 계산
        eps = 1e-15
        p = max(eps, min(1.0 - eps, self.prob))

        loss_val = -(self.y * math.log(p) + (1.0 - self.y) * math.log(1.0 - p))
        return loss_val

    def backward(self):
        d_logit = self.prob - self.y
        return [d_logit] # 모델이 예상한 답 - 실제 답
