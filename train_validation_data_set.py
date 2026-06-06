import json
import random

random.seed(42)


def get_data_set():
    # JSON 데이터 로드
    # 해당 데이터는 태극 모양 점 그래프 형태
    with open("./data.json", "r") as f:
        data = json.load(f)

    x_raw = data["x"]  # [[x1, y1], [x2, y2], ...] 구조의 2차원 리스트, 점의 위치를 의미
    y_raw = data["y"]  # [0, 1, 0, 1, ...] 구조의 1차원 리스트, 점의 성질(*색상)을 의미

    # 클래스 0과 클래스 1의 인덱스를 각각 분리합니다. (데이터 상에서 붉은 점과 푸른 점 분리)
    class_0_indices = [i for i, label in enumerate(y_raw) if label == 0]
    class_1_indices = [i for i, label in enumerate(y_raw) if label == 1]

    # 각각 무작위로 섞기 (셔플)
    random.shuffle(class_0_indices)
    random.shuffle(class_1_indices)

    # 8:2 비율로 쪼갤 기준점 계산
    split_point = int(len(class_0_indices) * 0.8)

    # 학습용/검증용 인덱스 취합
    train_indices = class_0_indices[:split_point] + class_1_indices[:split_point]
    val_indices = class_0_indices[split_point:] + class_1_indices[split_point:]

    # 다시 한번 전체적으로 섞어줍니다.
    random.shuffle(train_indices)
    random.shuffle(val_indices)

    # 3. 최종 학습(Train) / 검증(Val) 데이터셋 생성
    x_train = [x_raw[i] for i in train_indices]
    x_val = [x_raw[i] for i in val_indices]

    # y 데이터는 앞서 배운 unsqueeze(1) 규칙을 반영하여 [[0], [1], [1]] 형태로 저장합니다.
    y_train = [[y_raw[i]] for i in train_indices]
    y_val = [[y_raw[i]] for i in val_indices]

    return x_train, y_train, x_val, y_val
