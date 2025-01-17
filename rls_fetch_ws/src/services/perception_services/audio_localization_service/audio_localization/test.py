from com.nus.speech.server.project import Project
import numpy as np
import os
import scipy.io as spio
from scipy.io.wavfile import write
from flask import jsonify, json
from com.nus.speech.server.project_manager import ProjectManager as pm
import com.nus.speech.server.config as cfg

if __name__ == '__main__':
    print("localization start")
    # project = Project()
    project = pm.get_project(cfg.start_project_id)

    fileID = 2
    filename = 'data_ang' + str(fileID) + '.mat'
    test_path = os.path.join('Data/all', filename)
    abs_path = os.path.abspath(__file__).split('/')
    path = ""
    for i in range(len(abs_path)-1):
        path = path + abs_path[i]
        path = path + "/"
    path += test_path
    print(path)

    args = []

    # result = project.get_calculate_results(path, args)
    # print(result)
    # print(result["maxOut"], np.shape(result["maxOut"]))
    print(json.dumps(project.get_calculate_results(path, args)))

    # fileID = 2
    # rate = 10
    # filename = 'data_ang' + str(fileID) + '.mat'
    #
    # filename = 'data_ang' + str(fileID) + '.mat'
    # test_path = os.path.join('Data/all', filename)
    # abs_path = os.path.abspath(__file__).split('/')
    # path = ""
    # for i in range(len(abs_path)-1):
    #     path = path + abs_path[i]
    #     path = path + "/"
    # path += test_path
    # print(path)
    # mat = spio.loadmat(path)
    # wav, x, y = mat['wv'], mat['X'], mat['Y']  # Dim: (8192, 4) (51, 40, 6)
    #
    # wav1 = []
    # for subWav in wav:
    #     wav1.append(subWav[0])
    # audio1 = "audio1.wav"
    # # packetFile = open(audio1, "w")
    # write(audio1, rate, np.array(wav1))

