import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]  # 사용할 모델 리스트
model_name = models[2]  # Facenet 모델 사용

frame_interval = 30
frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        try:
            dfs = DeepFace.find(img_path=frame, db_path="/home/zenbook/Deepface/test_db", model_name=model_name, enforce_detection=False)
            if dfs:  # 리스트가 비어있지 않은지 확인
                first_data = dfs[0]
                print("First data:")
                print(first_data)
            else:
                print("No face found in the frame.")
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
    cv2.imshow('Webcam', frame)
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()