#!/usr/bin/env python

import torch
import numpy as np
import os
import rospy
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Int16
from speech_recognition_msgs.srv import SpeechRecognition, SpeechRecognitionResponse
import math
import time
import json

from audio_localization.com.nus.speech.server.model.sl_inference_small_batch import sMLP
from audio_localization.com.nus.speech.server.model.sl_inference_robot import EncodeSpiking
# from com.nus.speech.server.RespeakerTrans import Listener
from audio_localization.com.nus.speech.server.model.lib import angular_distance_compute, testing
import audio_localization.com.nus.speech.server.config as cfg


class AudioLocalizer(object):
    def __init__(self):
        print("init project")
        self.step = 30
        self.Tsim = 10
        self.device = 'cpu'
        model = sMLP(self.Tsim)
        self.model = model.to(self.device)
        self.encode_spiking = EncodeSpiking()

        print("load model")
        abs_path = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(abs_path, "../audio_localization/com/nus/speech/server/model")
        self.checkpoint = torch.load(os.path.join(self.path, "{0}.pt.tar".format("best_robot")), map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(self.checkpoint['model_state_dict'])  # restore the model with best_loss
        self.step = 30

        self.spiking_01 = np.full((10, 1000), 0)
        self.spiking_02 = np.full((10, 1000), 0)
        self.spiking_03 = np.full((10, 1000), 0)
        self.y_pred = np.full((360), 0)
        self.desition = 0,
        self.hist_copy = np.full((12), 0)
        print("opening ros")
        rospy.wait_for_service('/respeaker/get_trans_audio')
        self.get_audio = rospy.ServiceProxy('/respeaker/get_trans_audio', SpeechRecognition)
        print("opened ros")

    def get_audio_data(self):
        try:
            audio = self.get_audio()

            # rt_value = np.fromstring(audio.result.transcript[0], dtype=np.int16)
            rt_value = json.loads(audio.result.transcript[0])
            rt_value = np.array(rt_value)
            quantity = int(audio.result.confidence[0])
            print(rt_value.shape)
            rt_value = np.reshape(rt_value, (quantity, 2730, 4))
            print(rt_value.shape)
            current_data = rt_value.astype(np.float64)
            current_data = current_data / math.pow(2, 15)
            # print(current_data)
            return current_data, quantity
        except Exception as e:
            # print("Service call failed: %s".format(e))
            return None, None

    def get_audio_data_from_string(self, string, quantity):
        rt_value = json.loads(string)
        rt_value = np.array(rt_value)
        print(rt_value.shape)
        rt_value = np.reshape(rt_value, (quantity, 2730, 4))
        print(rt_value.shape)
        current_data = rt_value.astype(np.float64)
        current_data = current_data / math.pow(2, 15)
        # print(current_data)
        return current_data, quantity

    # def visualization(self, angle):
    #     num = int(angle / 30)
    #     if num <= 5:
    #         num = -num + 5
    #     else:
    #         num = -num + 17
    #     color = ColorRGBA()
    #     color.r = num
    #     rotation = Int16(angle)
    #     print(num)
    #     print("waiting publish")
    #     self.Color_pub.publish(color)
    #     self.Ang_pub.publish(45)
    #     print(" published")

    def localize_audio(self, audio_data, quantity):
        # print("get test results")
        change_flag = False
        # audio_data= np.ones((5, 2730, 4))
        # audio_data, quantity = self.get_audio_data()
        if audio_data is None:
            return None
        # print("get test audio_data", audio_data.shape)
        if quantity > 1:
            print(quantity)
            encode_spiking = self.encode_spiking.encode_robot_read(audio_data)
            Xte = encode_spiking['Xte2']
            num_sample = Xte.shape[0]
            Xmean = encode_spiking['Xte2'].mean(axis=1)
            Xte = torch.tensor((Xte.T - Xmean.reshape(1, -1)).T)
            # print(type(encode_spiking['Xte2']))
            # print(encode_spiking['Xte2'].shape)
            x_spike1, x_spike2, x_spike3, out, y_pred = testing(self.model, Xte, self.device)

            # Result Analysis
            angle_pred = torch.argmax(y_pred, 1)
            print(angle_pred)

            bin_range = np.arange((0 - self.step / 2), 360 + (self.step / 2) + 1, self.step)

            hist, bin = np.histogram(angle_pred, bin_range)
            hist_copy = hist[0:12]
            hist_copy[0] = hist_copy[0] + hist[12]

            decision = bin_range[np.argmax(hist_copy)] + self.step / 2

            change_flag = True
            print("finish testing and the decision is ", decision)
            return {cfg.grid_conv_1: x_spike1[-1].tolist(),
                    cfg.grid_conv_2: x_spike2[-1].tolist(),
                    cfg.grid_conv_3: x_spike3[-1].tolist(),
                    cfg.grid_conv_4: y_pred[-1].tolist(),
                    cfg.location: decision,
                    cfg.locationBins: hist_copy.tolist(),
                    cfg.change_flag: change_flag}
        else:
            return {cfg.grid_conv_1: self.spiking_01[-1].tolist(),
                    cfg.grid_conv_2: self.spiking_02[-1].tolist(),
                    cfg.grid_conv_3: self.spiking_03[-1].tolist(),
                    cfg.grid_conv_4: self.y_pred[-1].tolist(),
                    cfg.location: self.decision,
                    cfg.locationBins: self.hist_copy.tolist(),
                    cfg.change_flag: change_flag}

    def visualize(self, angle):
        num = int(angle / 30)
        if num <= 5:
            num = -num + 5
        else:
            num = -num + 17
        rotation = Int16(angle)
        print(num)

    def demo(self):
        audio_data, quantity = self.get_audio_data()
        self.localize_audio(audio_data, quantity)

if __name__ == '__main__':
    rospy.init_node("AudioLocalizer")
    srv = AudioLocalizer()
    while not rospy.is_shutdown():
        result = srv.localize_audio()
        if not result:
            continue
        print(result["changeFlag"])
        if(result["changeFlag"]):
            print(result["location"])
            srv.visualize(result["location"])
