import cv2
import numpy as np

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    
    # Bounding Box 좌표를 float으로 변환 (필수)
    boxes = np.array(boxes, dtype="float")
    
    # 박스의 좌표값들 추출
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 박스 면적 계산
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 박스의 점수를 내림차순으로 정렬
    idxs = np.argsort(areas)[::-1]
    
    pick = []
    
    while len(idxs) > 0:
        last = idxs[0]
        pick.append(last)
        suppress = [0]
        
        for i in range(1, len(idxs)):
            # 현재 박스와 비교할 박스 좌표
            xx1 = max(x1[last], x1[idxs[i]])
            yy1 = max(y1[last], y1[idxs[i]])
            xx2 = min(x2[last], x2[idxs[i]])
            yy2 = min(y2[last], y2[idxs[i]])
            
            # 교차 영역의 너비와 높이 계산
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            
            # IoU 계산
            overlap = float(w * h) / areas[idxs[i]]
            
            # 지정한 임계값보다 겹친다면 제거
            if overlap > overlapThresh:
                suppress.append(i)
        
        idxs = np.delete(idxs, suppress)
    
    return boxes[pick].astype("int")

image = cv2.imread("/home/user/test_canny/test_canny/test.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

template = cv2.imread("/home/user/test_canny/test_canny/template/template_test.jpeg")
template= cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
w ,h = template.shape[::-1]

template1 = cv2.imread("/home/user/test_canny/test_canny/template/template.jpeg")
template1= cv2.cvtColor(template1, cv2.COLOR_RGB2GRAY)
w1 ,h1 = template.shape[::-1]

res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where(res >= threshold)
res1 = cv2.matchTemplate(image_gray, template1, cv2.TM_CCOEFF_NORMED)
threshold1 = 0.7
loc1 = np.where(res1 >= threshold1)

# 일치한 좌표들을 Bounding Box로 변환
boxes = []
for pt in zip(*loc[::-1]):
    # 각 좌표에 대한 사각형 영역 (x1, y1, x2, y2)
    boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])

# 큰 면적만 선택하기 위해 NMS 적용
picked_boxes = non_max_suppression(boxes, overlapThresh=0.3)

# 선택된 큰 영역들에 사각형 그리기
for (x1, y1, x2, y2) in picked_boxes:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    print(f"좌표: ({x1}, {y1}) ~ ({x2}, {y2}), 면적: {(x2 - x1) * (y2 - y1)}")
    conter_x = (x1 + x2) // 2
    conter_y = (y1 + y2) // 2
    robot_x = 320
    robot_y = 400
    cv2.circle(image, (conter_x, conter_y), 1, (0, 200, 200), 2) # pomboard center
    cv2.circle(image, (robot_x,robot_y), 1, (0, 0, 255), 2)# robot center
    distance = (((conter_x - robot_x)**2 + (conter_y - robot_y)**2)**0.5)*0.258 #cm xd


boxes1 = []
for pt1 in zip(*loc1[::-1]):
    boxes1.append([pt1[0], pt1[1], pt1[0] + w, pt1[1] + h])
picked_boxes1 = non_max_suppression(boxes1, overlapThresh=0.3)
for (x3, y3, x4, y4) in picked_boxes1:
    cv2.rectangle(image, (x3, y3), (x4, y4), (0, 0, 255), 2)
    print(f"좌표: ({x3}, {y3}) ~ ({x4}, {y4}), 면적: {(x4 - x3) * (y4 - y3)}")
    conter_x1 = (x3 + x4) // 2
    conter_y1 = (y3 + y4) // 2
    cv2.circle(image, (conter_x1, conter_y1), 1, (0, 200, 200), 2) # pomboard center
    cv2.circle(image, (robot_x,robot_y), 1, (0, 0, 255), 2)# robot center
    distance1 = (((conter_x1 - robot_x)**2 + (conter_y1 - robot_y)**2)**0.5)*0.258 #cm 단위

    print(f"distance: {distance}")


cv2.imshow('test',image)

cv2.waitKey()
cv2.destroyAllWindows()