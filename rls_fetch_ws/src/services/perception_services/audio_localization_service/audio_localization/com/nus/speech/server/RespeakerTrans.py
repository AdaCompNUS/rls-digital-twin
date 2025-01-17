import rospy
import audio_common_msgs.msg
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
#import Robot_process
#from pixel_ring import usb_pixel_ring_v2
import usb.core
import usb.util
import time
from std_msgs.msg import ColorRGBA


"""def set_led_color(pixel_ring, r, g, b, a):
    pixel_ring.set_brightness(int(20 * a))
    pixel_ring.set_color(r=int(r * 255), g=int(g * 255), b=int(b * 255))"""


class Listener:

    def __init__(self):
        self.flag = True
        self.current_data = np.zeros((2730, 4))
        self.previous_data = np.zeros((2730, 4))
        self.data_audio = []
        self.direct = {}
        self.count = 0
        #self.inference = Robot_process.SnnInf()
        rospy.init_node('listener')
        rospy.Subscriber('/audio', audio_common_msgs.msg.AudioData, self.callback)
        self.Color_pub = rospy.Publisher('status_led', ColorRGBA, queue_size=10)
        rospy.spin()

    def calculate_absolute_mean(self, data):
        mean = 0.0
        length = float(len(data))
        for i in data:
            mean = mean + abs(i / length)
        return mean

    def visualization(self, angle):
        num = int(angle / 30)
        if num <= 5:
            num = -num + 5
        else:
            num = -num + 17
        color = ColorRGBA()
        color.r = num
        self.Color_pub.publish(color)

    def detect_speech(self, data):
        count = 0
        for i in range(0, 4):
            if self.calculate_absolute_mean(data[:, i]) > 100:
                count = count + 1
        if count >= 3:
            return True
        else:
            return False

    def combine_data(self):
        half_data = self.previous_data[1365:2730, :]
        transition = np.concatenate((half_data, self.current_data[0: 1365, :]), axis=0)
        if self.detect_speech(transition):
           # print('transition detects a speech')
            self.count = self.count + 1
            self.data_audio.append(transition)

    # def evaluate(self):
    #     print("Analysis begins......")
    #     output = np.array(self.data_audio)
    #     output = output / math.pow(2, 15)
    #     self.data_audio = []
    #     self.count = 0
    #     return self.inference.inference_direction(output)

    def get_audio_data(self):
        print("get audio data")
        output = []
        if self.count >= 2:
            output = np.array(self.data_audio)
            output = output / math.pow(2, 15)
            self.data_audio = []
            self.count = 0
        return output

    def update_visualization(self, angle):
        print("update visualization")
        self.visualization(angle)

    def callback(self, data):
        rt_value = np.frombuffer(data.data, dtype=np.int16)
        rt_value = np.reshape(rt_value, (2730, 4))
        self.current_data = rt_value.astype(np.float64)

        if self.flag:
            self.flag = False
        else:
            self.combine_data()
        self.previous_data = self.current_data
        if self.detect_speech(self.current_data):
            self.data_audio.append(self.current_data)
            self.count = self.count + 1
        if self.count >= 2:
            # prediction = self.evaluate()
            # self.visualization(prediction)
            print("need update visualization")


        """
        if self.detect_speech(self.current_data):
            print('detects a speech')
        """
        '''
        micro_data = []
        for i in range(1, 5):
            micro_data.append(data.src[i].wavedata)
        npOutput = np.array(micro_data)
        npOutput = npOutput / math.pow(2, 15)
        print(npOutput.shape)
         code below is employed to visualize microphone data

        number = len(mic1)
        number_vec = range(number)
        plt.plot(number_vec, mic1)
        plt.show()
        rospy.sleep(2)
        '''



if __name__ == '__main__':
    system = Listener()

