import cv2

# 웹캠에서 영상을 캡처하기 위해 VideoCapture 객체를 생성합니다.
cap = cv2.VideoCapture(0)

while True:
    # 웹캠에서 프레임을 읽어옵니다.
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 프레임을 그레이스케일로 변환합니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러를 적용하여 노이즈를 줄입니다.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 윤곽선을 검출합니다.
    edges = cv2.Canny(blurred, 50, 150)
    
    # 윤곽선을 그리기 위해 contours를 찾습니다.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 윤곽선을 원본 프레임에 그립니다.
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    # 결과 이미지를 표시합니다.
    cv2.imshow('Contours', frame)
    
    # 'q' 키를 누르면 프로그램을 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 자원을 해제하고 모든 창을 닫습니다.
cap.release()
cv2.destroyAllWindows()