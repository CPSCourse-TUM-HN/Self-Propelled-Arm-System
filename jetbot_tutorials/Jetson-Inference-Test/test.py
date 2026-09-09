from jetbot import Robot
import time

robot = Robot()

try:
    robot.forward(0.15)
    time.sleep(0.2)
finally:
    robot.stop()