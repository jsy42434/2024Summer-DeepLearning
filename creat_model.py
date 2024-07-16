import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

#작업 영역 수정 후 실행 cd "C:\Users\AI06\Desktop\박스 각도 계산 프로그램"
#혹은 데이터 셋 경로를 절대결로로 지정

# YOLOv8 모델 로드
model = YOLO('yolov8s-seg.pt')

# 데이터셋 경로 설정(yaml 경로)
data_path = '/box_total/data.yaml'

# 모델 학습
model.train(data=data_path,
                      epochs=30,
                      imgsz=640,
                      weight_decay=0.0005,
                      patience=5)

# 학습 완료된 모델로 바로 시작 할 때 주석 해제
# 학습된 모델의 pt 경로를 불러와야 함
#model_path = '../best.pt'
#model = YOLO(model_path)

# 이미지 경로 설정(테스트 이미지 경로 설정)
image_path = "/box_total_data_set/test/images/box_5_png_jpg.rf.4da2f11509dde0a73e2e39982b30516b"
# 이미지 로드
image = cv2.imread(image_path)

# 객체 탐지 및 세그멘테이션 수행
results = model(image_path)

# 결과 시각화
for result in results:
    masks = result.masks.data  # 세그멘테이션 마스크
    boxes = result.boxes.xyxy  # 바운딩 박스 좌표

    for i, mask in enumerate(masks):
        # 텐서를 넘파이 배열로 변환
        mask_np = mask.cpu().numpy()

        # 폴리곤으로 변환
        contours, _ = cv2.findContours(mask_np.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # 근사화하여 4개의 꼭지점으로 제한
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # 근사된 꼭지점이 4개일 때만 출력 및 그리기
                # 폴리곤 좌표 출력
                approx = approx.reshape(-1, 2)  # 2차원 배열로 변환
                print(f"Contour {i} points:")
                for point in approx:
                    print(f"({point[0]}, {point[1]})")

                # 폴리곤 그리기
                cv2.polylines(image, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

        # 바운딩 박스도 그리기 (원하는 경우)
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

# 결과 출력
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()