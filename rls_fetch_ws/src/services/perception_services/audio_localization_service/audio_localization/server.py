#!/usr/bin/env python3

from flask import request, jsonify, json
from flask_api import FlaskAPI
import rospy
import threading
from std_msgs.msg import ColorRGBA, Int16
import os

#
import roslibpy
import com.nus.speech.server.config as cfg
from com.nus.speech.server.project_manager import ProjectManager as pm

__author__ = 'Cai Qing, jibin'
__copyright__ = 'Copyright 2019, The speech location '
__version__ = '2019a'
__maintainer__ = 'Cai Qing,  jibin'
__status__ = 'Prototype'

# ws_client = ros_ws('127.0.0.1', 9090) # ip, port, name of client
# ws_client.connect()
# Color_pub = rospy.Publisher('status_led', ColorRGBA, queue_size=10)
# Ang_pub = rospy.Publisher('spearker_pos', Int16, queue_size=10)
# client = roslibpy.Ros(host='localhost', port=9090)
#client = roslibpy.Ros(host='192.168.99.150', port=9090)
client = roslibpy.Ros(host='kris-NUC8i7BEH', port=9090)
client.run()
talker_color = roslibpy.Topic(client, '/status_led', 'std_msgs/ColorRGBA')
talker_angle = roslibpy.Topic(client, '/spearker_pos', 'std_msgs/Int16')
# service = \
#     roslibpy.Service(client, '/get_trans_audio', 'speech_recognition_msgs/SpeechRecognition')
# request = roslibpy.ServiceRequest()

def visualization(angle):
    num = int(angle / 30)
    if num <= 5:
        num = -num + 5
    else:
        num = -num + 17
    # color = ColorRGBA()
    # color.r = num
    rotation = Int16(angle)
    print(num)
    print("waiting publish")
    # C_pub.publish(color)
    # A_pub.publish(45)
    talker_color.publish(roslibpy.Message({'r': num}))
    talker_angle.publish(roslibpy.Message({'data': angle}))
    # ws_client.publish("/status_led", color)
    # ws_client.publish("/spearker_pos", rotation)
    print(" published")

app = FlaskAPI(__name__)

@app.route('/sound', methods=['GET', 'POST'])
def sound_request():
    print("get request")
    # print(path)
    # result = service.call(request)
    # print result
    project = pm.get_project(cfg.start_project_id)
    print("project finished")
    # args = []
    result = project.get_calculate_results()
    print result["changeFlag"]
    if(result["changeFlag"]):
        print result["location"]
        visualization(result["location"])
    return json.dumps(result)


if __name__ == '__main__':
    print("engine server start: ")
    app.run(port=cfg.PORT, debug=True)
