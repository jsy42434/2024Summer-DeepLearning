import cv2
import numpy as np
from ultralytics import YOLO
import math
import time     # sleep 함수 사용시 필요
import math
import csv
import os

#작업 영역 수정 후 실행 cd "C:\Users\AI06\Desktop\박스 각도 계산 프로그램"
#혹은 모델 경로를 절대결로로 지정
# 48.19
# 10도에서 45도 사이일때 인식 잘됨

def append_angles_to_csv(filename, angles, average_angles):
    """
    주어진 angle과 average_angle 값을 배열 형태로 받은 후, 기존 CSV 파일에 추가합니다.
    파일이 존재하지 않는 경우, 새로 생성합니다.
    
    :param filename: 생성할 또는 추가할 CSV 파일 이름
    :param angles: 각도 배열
    :param average_angles: 평균 각도 배열
    """
    # 새 파일인지 확인
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 파일이 새로 생성되는 경우 헤더 작성
        if not file_exists:
            writer.writerow(['angle', 'average_angle'])
        
        # 배열의 각 요소를 한 줄씩 추가
        for angle, average_angle in zip(angles, average_angles):
            writer.writerow([angle, average_angle])
        
        
def trimmed_mean(arr, lower_trim_ratio, upper_trim_ratio):
    """
    배열에서 양 끝값들의 일정 비율을 제외한 평균을 계산합니다.
    
    :param arr: 리스트 또는 배열 형태의 숫자 데이터
    :param lower_trim_ratio: 제외할 작은 값의 비율 (0과 0.5 사이의 값)
    :param upper_trim_ratio: 제외할 큰 값의 비율 (0과 0.5 사이의 값)
    :return: 양 끝값들의 일정 비율을 제외한 평균 값
    """
    if not 0 <= lower_trim_ratio <= 0.5:
        raise ValueError("lower_trim_ratio는 0과 0.5 사이의 값이어야 합니다.")
    if not 0 <= upper_trim_ratio <= 0.5:
        raise ValueError("upper_trim_ratio는 0과 0.5 사이의 값이어야 합니다.")
    if lower_trim_ratio + upper_trim_ratio >= 1:
        raise ValueError("lower_trim_ratio와 upper_trim_ratio의 합은 1보다 작아야 합니다.")
    
    n = len(arr)
    if n == 0:
        raise ValueError("입력 배열은 비어 있을 수 없습니다.")
    
    arr_sorted = sorted(arr)
    lower_trim_count = int(n * lower_trim_ratio)
    upper_trim_count = int(n * upper_trim_ratio)
    
    trimmed_arr = arr_sorted[lower_trim_count:n - upper_trim_count]

    # 평균 계산
    trimmed_mean = sum(trimmed_arr) / len(trimmed_arr)
    
    return trimmed_mean


def calculate_angle(real_length, image_length):
    """
    상자의 실제 길이비와 이미지에서 추출된 길이비를 사용하여 각도를 계산합니다.
    
    Parameters:
    real_length (float): 실제 길이
    image_length (float): 이미지에서 추출된 길이
    
    Returns:
    float: 계산된 각도 (도 단위)
    """
    
    
    # 길이 비율을 계산합니다.
    length_ratio = image_length/real_length
    
    # 길이 비율이 -1에서 1 사이에 있는지 확인합니다.
    
    if not -1 <= length_ratio <= 1:
        print(f"real_length가 image_length보다 커야 합니다")
        return 0.0

    
    # 각도를 계산합니다 (라디안 단위)
    angle_rad = math.acos(length_ratio)
    
    # 각도를 도 단위로 변환합니다.
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def calculate_average_ratio(lengths):
    """
    상자의 네 변의 길이가 들어있는 배열을 사용하여 긴 변과 짧은 변의 길이비를 구합니다.
    
    short_lengths (float): 짧은 변 길이
    long_lengths (float): 긴 변 길이
    
    Returns:
    float: 계산된 길이 비
    """
    # 입력된 길이가 4개인지 확인
    if len(lengths) != 4:
        raise ValueError("길이는 반드시 4개의 값을 가져야 합니다.")

    # 길이를 정렬하여 짧은 길이와 긴 길이를 구분
    sorted_lengths = sorted(lengths)
    short_lengths = sorted_lengths[1]  # 두번째로 짮은 길이사용
    long_lengths = sorted_lengths[2:]   # 가장 긴 두 길이

    # 긴 길이 평균을 계산
    average_long = sum(long_lengths) / 2

    # 평균 비율을 계산
    ratio = average_long / short_lengths

    return ratio

def calculate_distance(point1, point2):
    """두 점 사이의 유클리드 거리 계산"""
    return np.sqrt(np.sum((point1 - point2) ** 2))

# 메인 시작
# 학습 완료된 모델의 pt 경로를 불러오기
model_path = r"..\best.pt" #상대 결로로 설정
model = YOLO(model_path)

# 0도 일때 상자의 (긴 변/짧은 변)의 값
real_ratio=1.50

# 계산된 각도들을 저장할 배열
angle_list=[]
angle_average_list=[]
ratio_averages=[]
# 웹캠 초기화
cap = cv2.VideoCapture(0)

# 웹캠이 열려 있지 않은 경우 예외 처리
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

name =0


while True:
    
    filename = f'angles{name}.csv'

    # 프레임 캡처
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델에 프레임 전달하여 객체 탐지 및 세그멘테이션 수행
    results = model(frame)

    # 결과 시각화
    for result in results:
        if result.masks is not None:  # 세그멘테이션 결과가 있는 경우에만 처리
            masks = result.masks.data  # 세그멘테이션 마스크

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
                        for point in approx:
                            # 이미지에 좌표값 출력
                            cv2.putText(frame, f"({point[0]}, {point[1]})", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        distances = []
                        num_points = len(approx)
                        for j in range(num_points):
                            p1 = approx[j]
                            p2 = approx[(j + 1) % num_points]
                            distance = calculate_distance(p1, p2)
                            distances.append(distance)
                        
                        # 긴 변과 짧은변의 비율 계산
                        ratio=calculate_average_ratio(distances)
                        ratio_averages.append(ratio)
                        ratio_average=trimmed_mean(ratio_averages,0.3,0.1)
                        # 0도일때 비율과 측정된 비율을 사용하여 각도 계산
                        angle = calculate_angle(real_ratio,ratio)
                        # 계산된 각도를 배열에 저장
                        angle_list.append(angle)
                        angle_average=trimmed_mean(angle_list,0.3,0.1)
                        angle_average_list.append(angle_average)
                        print(f"측정 각도 :{angle:.2f}도")
                        # 측정된 각도들을 넣은 배열에서 하위 30%, 상위 10%의 값을 제외한 값들의 평균 출력
                        print(f"평균 각도 :{angle_average:.2f}도")

                        # 폴리곤 그리기
                        cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
                        
                        # 각도 화면에 출력
                        cv2.putText(frame, f"ratio :{ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"ratio_average :{ratio_average:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"angle :{angle:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"angle_average :{angle_average:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Crosshair parameters
    length = 20  # Length of the crosshair lines
    color = (0, 255, 0)  # Color of the crosshair (Green)
    thickness = 2  # Thickness of the crosshair lines

    # Draw horizontal line
    cv2.line(frame, (center_x - length, center_y), (center_x + length, center_y), color, thickness)
    # Draw vertical line
    cv2.line(frame, (center_x, center_y - length), (center_x, center_y + length), color, thickness)
    # 결과 프레임 출력
    cv2.imshow('YOLOv8 Detection', frame)


    key = cv2.waitKey(1) & 0xFF
    # 'r' 키를 누르면 평균 배열(angle_average) 초기화
    if key == ord('r'):
        angle_list = []
        angle_average_list = []
    # 'c' 키를 누르면 angle_list와 angle_average_list를 csv파일로 저장
    if key == ord('c'):
        append_angles_to_csv(filename, angle_list, angle_average_list)
        name+=1
    # 's' 키를 누르는동안 프로그램 정지
    elif key == ord('s'):
        while True:  # 다시 키 입력을 확인
            key = cv2.waitKey(1) & 0xFF
            if key != ord('s'):  # 's' 키에서 손을 뗐을 때 루프 탈출
                break
    
    # 'q' 키를 누르면 종료
    elif key == ord('q'):
        break

# 웹캠 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()