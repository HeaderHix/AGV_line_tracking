#!/usr/bin/env python

from __future__ import print_function
import signal
import threading

import roslib; roslib.load_manifest('myagv_teleop')
import rospy

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

import sys
from select import select

if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

import cv2
import numpy as np
import time

TwistMsg = Twist

msg = """
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
"""

moveBindings = {
        'i':(1,0,0,0),
        'o':(0,0,0,-1),
        'j':(1,0,0,1),
        'l':(1,0,0,-1),
        'u':(0,0,0,1),
        ',':(-1,0,0,0),
        '.':(-1,0,0,-1),
        'm':(-1,0,0,1),
        'O':(0,0,0,-1),
        'I':(1,0,0,0),
        'r':(0,1,0,0),
        'L':(0,-1,0,0),
        'U':(0,0,0,1),
        '<':(-1,0,0,0),
        '>':(-1,0,0,-1),
        'M':(-1,0,0,1),
        's':(0,0,0,0),
        
    }

speedBindings={
        'q':(1.1,1.1),
        'z':(.9,.9),
        'w':(1.1,1),
        'x':(.9,1),
        'e':(1,1.1),
        'c':(1,.9),
    }

class PublishThread(threading.Thread):
    def __init__(self, rate):
        super(PublishThread, self).__init__()
        self.publisher = rospy.Publisher('cmd_vel', TwistMsg, queue_size = 1)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0
        self.condition = threading.Condition()
        self.done = False

        # Set timeout to None if rate is 0 (causes new_message to wait forever
        # for new data to publish)
        if rate != 0.0:
            self.timeout = 1.0 / rate
        else:
            self.timeout = None

        self.start()

    def wait_for_subscribers(self):
        i = 0
        while not rospy.is_shutdown() and self.publisher.get_num_connections() == 0:
            if i == 4:
                print("Waiting for subscriber to connect to {}".format(self.publisher.name))
            rospy.sleep(0.5)
            i += 1
            i = i % 5
        if rospy.is_shutdown():
            raise Exception("Got shutdown request before subscribers connected")

    def update(self, x, y, z, th, speed, turn):
        self.condition.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn
        # Notify publish thread that we have a new message.
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

    def run(self):
        twist_msg = TwistMsg()

        if stamped:
            twist = twist_msg.twist
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.header.frame_id = twist_frame
        else:
            twist = twist_msg
        while not self.done:
            if stamped:
                twist_msg.header.stamp = rospy.Time.now()
            self.condition.acquire()
            # Wait for a new message or timeout.
            self.condition.wait(self.timeout)

            # Copy state into twist message.
            twist.linear.x = self.x * self.speed
            twist.linear.y = self.y * self.speed
            twist.linear.z = self.z * self.speed
            twist.angular.x = 0
            twist.angular.y = 0
            twist.angular.z = self.th * self.turn

            self.condition.release()

            # Publish.
            self.publisher.publish(twist_msg)

        # Publish stop message when thread exits.
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.publisher.publish(twist_msg)


# def detect_yellow_buleNredObject_line(frame):
#     height = frame.shape[0]

#     roi = frame[(14 * height //20):(16 * height // 20), :]
#     roi_BR = frame[(19 * height //20):(20 * height // 20), :]
#     hsv_BR = cv2.cvtColor(roi_BR, cv2.COLOR_BGR2HSV)
    
#     # Define the lower and upper bounds for the red color in HSV
#     lower_red = np.array([0, 120, 70])
#     upper_red = np.array([10, 255, 255])
#     # Convert the ROI from BGR to HSV color space
#     hsv_Y = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#     # Define the range of yellow color in HSV
#     lower_yellow = np.array([20, 100, 100])
#     upper_yellow = np.array([30, 200, 200])

#     lower_blue = np.array([110, 50, 50])
#     upper_blue = np.array([130, 255, 255])
#     # Threshold the HSV image to get only yellow colors
#     blue_mask = cv2.inRange(hsv_BR, lower_blue, upper_blue)
#     yellow_mask = cv2.inRange(hsv_Y, lower_yellow, upper_yellow)
#     red_mask = cv2.inRange(hsv_BR, lower_red, upper_red)

#     # Bitwise-AND mask and original image
#     blue_result = cv2.bitwise_and(roi_BR, roi_BR, mask=blue_mask)
#     red_result = cv2.bitwise_and(roi_BR, roi_BR, mask=red_mask)
#     # Bitwise-AND mask and original image
#     yellow_line = cv2.bitwise_and(roi, roi, mask=yellow_mask)
#     return yellow_line,red_result,blue_result, roi.shape[1]  # Return the detected line and width of the ROI

# last_direction = None  # 이전에 반환한 값을 저장하는 변수 초기화

# def process_frame(frame):
#     global last_direction  # 전역 변수로 선언하여 함수 내에서 사용

#     yellow_line, roi_width = detect_yellow_buleNredObject_line(frame)
#     gray = cv2.cvtColor(yellow_line, cv2.COLOR_BGR2GRAY)
#     _, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         max_contour = max(contours, key=cv2.contourArea)
#         M = cv2.moments(max_contour)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"])
#         section_width = roi_width // 35  # ROI 너비를 25개의 섹션으로 나눕니다.
#         left_section = cx // section_width
#         right_section = cx // section_width

#         #빨간색 검출시 's' return


        
#         #파란색 검출시 'r' return

#                 # 왼쪽 끝 섹션에 위치하는지 확인
#         if left_section < 8:
#             last_direction = "u"  # 좌회전
#         elif 8 < left_section < 14:
#             last_direction = "j"  # 좌측+전진
#         elif 27 > right_section > 21:
#             last_direction = "l"  # 우측+전진
#         elif right_section > 27:
#             last_direction = "o"  # 우회전
#         else:
#             last_direction = "i"  # 직진
#     rospy.sleep(0.5)
#     return last_direction

def detect_colors(frame):
    # �̹����� �ϴ� 4���� 1�� �����Ͽ� ROI ����
    height = frame.shape[0]
    roi = frame[(3 * height // 4):, :]
    
    # ROI�� BGR���� HSV ���������� ��ȯ
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # �������� ���� HSV ���� ����
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    # �Ķ����� ���� HSV ���� ����
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])
    
    # �������� �����ϱ� ���� ����ũ ����
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    # �Ķ����� �����ϱ� ���� ����ũ ����
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    return red_mask, blue_mask

# ����� ���� ���� �Լ�
def detect_yellow_line(frame):
    # �̹����� �ϴ� 4���� 1�� �����Ͽ� ROI ����
    height = frame.shape[0]
    roi = frame[(3 * height // 4):, :]
    
    # ROI�� BGR���� HSV ���������� ��ȯ
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # ������� ���� HSV ���� ����
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    # ������� �����ϱ� ���� ����ũ ����
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # ����ũ�� ���� �̹����� ��Ʈ������ �����Ͽ� ����� ���� ����
    yellow_line = cv2.bitwise_and(roi, roi, mask=yellow_mask)
    return yellow_line, roi.shape[1]  # ����� ���ΰ� ROI�� �ʺ� ��ȯ
last_direction = None
# ������ ó�� �Լ�
def process_frame(frame):
    global last_direction 
    # ������, �Ķ��� ����
    red_mask, blue_mask = detect_colors(frame)

    # ������ ���� �� ����
    if np.any(red_mask):
        last_direction ="s"
    # �Ķ��� ���� �� ȸ�� ���� ����
    elif np.any(blue_mask):
        # last_direction ="J"
        # time.sleep(0.2)
        # last_direction ="i"
        # time.sleep(0.2)
        # last_direction ="L"
        last_direction = "r"
    # ����� ���� ����
    else:
        # if last_direction == "r":
        #     rospy.sleep(0.5)
        #     last_direction = None
        # ����� ������ �����Ͽ� ���� ����
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
                left_bound = 2*roi_width // 7
                # left_cross = 2 * roi_width // 5
                # right_cross =   5 * roi_width // 5
                right_bound = 5 * roi_width // 7
                if cx < left_bound:
                    last_direction = "u"
                elif cx > right_bound:
                    last_direction = "o"
                # elif right_cross < cx <right_bound:
                #     last_direction = "l"
                # elif left_cross < cx <right_cross:
                #     last_direction = "i"
                # elif left_bound < cx <left_cross:
                #     last_direction = "j"
                else:
                    last_direction = "i"    
    return last_direction
    
# def getKey(settings, timeout):
#     if sys.platform == 'win32':
#         # getwch() returns a string on Windows
#         key = msvcrt.getwch()
#     else:
#         tty.setraw(sys.stdin.fileno())
#         # sys.stdin.read() returns a string on Linux
#         rlist, _, _ = select([sys.stdin], [], [], timeout)
#         if rlist:
#             key = sys.stdin.read(1)
#         else:
#             key = ''
#         termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
#     return key
def blue():
    for i in range(0,3):
                    if i == 0:
                        pub_thread.update(0, 1, 0, 0, speed, turn)
                        # rospy.sleep(0.3)
                    elif i == 1:
                        pub_thread.update(1, 0, 0, 0, speed, turn)
                        # rospy.sleep(0.3)
                    elif i == 2:
                        pub_thread.update(0, -1, 0, 0, speed, turn)
                        # rospy.sleep(0.3)

def saveTerminalSettings():
    if sys.platform == 'win32':
        return None
    return termios.tcgetattr(sys.stdin)

def restoreTerminalSettings(old_settings):
    if sys.platform == 'win32':
        return
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def vels(speed, turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

if __name__=="__main__":
    cap = cv2.VideoCapture(0)

    rospy.init_node('teleop_twist_keyboard')

    speed = rospy.get_param("~speed", 0.3)
    turn = rospy.get_param("~turn", 0.2)
    speed_limit = rospy.get_param("~speed_limit", 1.0)
    turn_limit = rospy.get_param("~turn_limit", 1.0)
    repeat = rospy.get_param("~repeat_rate", 0.0)
    key_timeout = rospy.get_param("~key_timeout", 0.52)
    stamped = rospy.get_param("~stamped", False)
    twist_frame = rospy.get_param("~frame_id", '')
    if stamped:
        TwistMsg = TwistStamped

    pub_thread = PublishThread(repeat)

    x = 0
    y = 0
    z = 0
    th = 0
    status = 0

    try:
        pub_thread.wait_for_subscribers()
        pub_thread.update(x, y, z, th, speed, turn)

        print(msg)
        print(vels(speed,turn))
        while(1):
            
            ret , frame = cap.read()
            key = process_frame(frame)
            print(key)
            
            # if key == 'r':
            #     count = 0
            #     while(1):
            #         count = count+1
            #         if(count < 40):
            #             x = moveBindings['s'][0]
            #             y = moveBindings['s'][1]
            #             z = moveBindings['s'][2]
            #             th = moveBindings['s'][3]
            #             pub_thread.update(x, y, z, th, 0.5, turn)
            #         elif(count == 40):
            #             x = moveBindings['J'][0]
            #             y = moveBindings['J'][1]
            #             z = moveBindings['J'][2]
            #             th = moveBindings['J'][3]
            #             pub_thread.update(x, y, z, th, 0.5, turn)
            #         elif(count == 80):
            #             x = moveBindings['i'][0]
            #             y = moveBindings['i'][1]
            #             z = moveBindings['i'][2]
            #             th = moveBindings['i'][3]
            #             pub_thread.update(x, y, z, th, 0.5, turn)    
            #         elif(count == 80):
            #             x = moveBindings['L'][0]
            #             y = moveBindings['L'][1]
            #             z = moveBindings['L'][2]
            #             th = moveBindings['L'][3]
            #             pub_thread.update(x, y, z, th, 0.5, turn)
            #         elif(count == 100):
            #             break  

            if key in moveBindings.keys():
                x = moveBindings[key][0]
                y = moveBindings[key][1]
                z = moveBindings[key][2]
                th = moveBindings[key][3]
           # elif key in speedBindings.keys():
           #     speed = min(speed_limit, speed * speedBindings[key][0])
           #     turn = min(turn_limit, turn * speedBindings[key][1])
           #     if speed == speed_limit:
           #         print("Linear speed limit reached!")
           #     if turn == turn_limit:
           #         print("Angular speed limit reached!")
           #     print(vels(speed,turn))
           #     if (status == 14):
           #         print(msg)
           #     status = (status + 1) % 15
            else:
                # Skip updating cmd_vel if key timeout and robot already
                # stopped.
                if key == '' and x == 0 and y == 0 and z == 0 and th == 0:
                    continue
                x = 0
                y = 0
                z = 0
                th = 0
                if (key == '\x03'):
                    break
            # elif key ==None :
            #     x = 0
            #     y = 0
            #     z = 0
            #     th = 1

            pub_thread.update(x, y, z, th, speed, turn)

    except Exception as e:
        print(e)

    finally:
        pub_thread.stop()
        #restoreTerminalSettings(settings)
        cap.release()
