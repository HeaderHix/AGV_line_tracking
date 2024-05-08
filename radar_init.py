import subprocess
import sys
import threading
import time
import traceback
import socket
import json

from pymycobot.myagv import MyAgv
import os
import cv2
if os.name == "posix":
    import RPi.GPIO as GPIO


def radar_open():
    def radar_high():
        GPIO.setmode(GPIO.BCM)
        time.sleep(0.1)
        GPIO.setup(20,GPIO.OUT)
        GPIO.output(20,GPIO.HIGH)

    radar_high()
    time.sleep(0.05)
    launch_command = "roslaunch myagv_odometry myagv_active.launch" #ros open
    subprocess.run(
            ['gnome-terminal', '-e', f"bash -c '{launch_command}; exec $SHELL'"])

def radar_close( run_launch):
    def radar_low():
        GPIO.setmode(GPIO.BCM)
        time.sleep(0.1)
        GPIO.setup(20,GPIO.OUT)
        GPIO.output(20,GPIO.LOW)

    radar_low()
    time.sleep(0.05)

    # subprocess.run(close_command, shell=True)


while True:
    user_in = input("please input : ")
    if user_in == 'i':
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(21,GPIO.OUT)
        GPIO.output(21,GPIO.HIGH)
        radar_open()
        # radar_open = threading.Thread(target = radar_open, daemon = True)
        # radar_open.start()
    elif user_in == 'd':
        radar_close()
