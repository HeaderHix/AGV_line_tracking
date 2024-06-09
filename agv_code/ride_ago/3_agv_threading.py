import cv2
import numpy as np
from pymycobot.myagv import MyAgv

# AGV 연결
agv = MyAgv("/dev/ttyAMA2", 115200)

def detect_yellow_line(frame):
    # 이미지의 하단 4분의 1만 추출하여 ROI 설정
    height = frame.shape[0]
    roi = frame[(3 * height // 4):, :]
    
    # Convert the ROI from BGR to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # Threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    yellow_line = cv2.bitwise_and(roi, roi, mask=yellow_mask)
    return yellow_line, roi.shape[1]  # Return the detected line and width of the ROI

def process_frame(frame):
    # Detect yellow line in the frame
    yellow_line, roi_width = detect_yellow_line(frame)
    # Convert the image to grayscale
    gray = cv2.cvtColor(yellow_line, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour based on area
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            left_bound = roi_width // 4
            right_bound = 3 * roi_width // 4
            if cx < left_bound:
                return "LEFT"
            elif cx > right_bound:
                return "RIGHT"
            else:
                return "GO_AHEAD"
    return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame from camera")
            break

        result = process_frame(frame)
        if result:
            print(result)
            if result == "LEFT":
                agv.counterclockwise_rotation(1)
            elif result == "RIGHT":
                agv.clockwise_rotation(1)
            elif result == "GO_AHEAD":
                agv.go_ahead(15)

        # Draw dividing lines on the frame
        height = frame.shape[0]
        division_line_y = 3 * height // 4
        roi_width = frame.shape[1]
        left_bound = roi_width // 4
        right_bound = 3 * roi_width // 4
        center_bound = roi_width // 2

        # Draw horizontal line to show the ROI area
        cv2.line(frame, (0, division_line_y), (roi_width, division_line_y), (255, 0, 0), 2)
        # Draw vertical lines to divide the bottom quarter into three parts
        cv2.line(frame, (left_bound, division_line_y), (left_bound, height), (255, 0, 0), 2)
        cv2.line(frame, (right_bound, division_line_y), (right_bound, height), (255, 0, 0), 2)

        # Enlarge the center frame
        frame[division_line_y:, left_bound:right_bound] = cv2.resize(frame[division_line_y:, left_bound:right_bound], (roi_width // 2, height // 4))

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
