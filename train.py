import random
from bce_loss_with_logits import BceLossWithLogits
from train_validation_data_set import get_data_set
from gradient_descent_optimizer import GradientDescentOptimizer
from mlp_model import MlpModel


def train_one_epoch(model, loss_function, optimizer, x_train, y_train):
    """1 에폭 동안 학습을 수행합니다."""
    train_loss = 0.0
    train_correct = 0

    for i in range(len(x_train)):
        sample_x = x_train[i]
        sample_y = y_train[i]

        # 순전파 -> 채점
        predicted_logit = model.forward(sample_x)
        loss_val = loss_function.forward(predicted_logit, sample_y)
        train_loss += loss_val

        # 정확도 누적
        pred_label = 1.0 if loss_function.prob >= 0.5 else 0.0
        if pred_label == sample_y[0]:
            train_correct += 1

        # 역전파 및 최적화 진행
        root_backward_value = loss_function.backward()
        model.backward(root_backward_value)
        optimizer.step()

    avg_train_loss = train_loss / len(x_train)
    train_accuracy = (train_correct / len(x_train)) * 100
    return avg_train_loss, train_accuracy


def evaluate(model, loss_function, x_val, y_val):
    """검증 데이터셋으로 모델을 평가합니다."""
    val_loss = 0.0
    val_correct = 0

    for i in range(len(x_val)):
        sample_x = x_val[i]
        sample_y = y_val[i]

        # 순전파 -> 채점
        predicted_logit = model.forward(sample_x)
        loss_val = loss_function.forward(predicted_logit, sample_y)
        val_loss += loss_val

        # 정확도 누적
        val_pred_label = 1.0 if loss_function.prob >= 0.5 else 0.0
        if val_pred_label == sample_y[0]:
            val_correct += 1

    avg_val_loss = val_loss / len(x_val)
    val_accuracy = (val_correct / len(x_val)) * 100
    return avg_val_loss, val_accuracy


def main():
    random.seed(42)  # 시드 고정

    # 데이터셋 로드
    x_train, y_train, x_val, y_val = get_data_set()
    print(f"-> 데이터 로드 완료! (학습셋: {len(x_train)}건 / 검증셋: {len(x_val)}건)")
    print("-" * 60)

    # 하이퍼파라미터 및 컴포넌트 인스턴스화
    epochs = 30
    learning_rate = 0.1

    model = MlpModel()
    loss_function = BceLossWithLogits()
    optimizer = GradientDescentOptimizer(layer_list=[model.linear_1, model.linear_2], lr=learning_rate)

    print("🚀 태극 모양 점 데이터를 이용한 순수 파이썬 MLP 학습을 가동합니다!")
    print("-" * 60)

    # 메인 학습 및 검증 루프
    for epoch in range(epochs):
        # 학습 단계
        avg_train_loss, train_accuracy = train_one_epoch(
            model, loss_function, optimizer, x_train, y_train
        )

        # 검증 단계
        avg_val_loss, val_accuracy = evaluate(
            model, loss_function, x_val, y_val
        )

        # 에폭마다 상태 출력
        print(f"Epoch [{epoch + 1:2d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} (Acc: {train_accuracy:.2f}%) | "
              f"Val Loss: {avg_val_loss:.4f} (Acc: {val_accuracy:.2f}%)")

    print("-" * 60)
    print("🎉 최종 완료! 분할된 태극 데이터를 바탕으로 검증 검사까지 수행하는 완벽한 파이프라인이 완성되었습니다.")


if __name__ == "__main__":
    main()