import gym
import keyboard
import os
import numpy as np

class CreateData:
    def __init__(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        self.path = path
        self.endgame = False

    def keyPress(self):
        act = [0,0,0]
        if keyboard.is_pressed("a"):
            act[0] = -1
            act[1] = 0
        elif keyboard.is_pressed("d"):
            act[0] = 1
            act[1] = 0
        if keyboard.is_pressed("w"):
            if act[0] == 0:
                act[1] = 1
        elif keyboard.is_pressed("s"):
            act[2] = 1
        if keyboard.is_pressed("esc"):
            self.endgame = True
        else:
            self.endgame = False
        return act, self.endgame

    def saveData(self, actionsArr, observationsArr):
        element_count = len(os.listdir(self.path))/2
        np.save(self.path + f"//actions[{int(element_count)}]", actionsArr)
        np.save(self.path + f"//observations[{int(element_count)}]", observationsArr)

    def loadData(self, start, stop):
        element_count = int(len(os.listdir(self.path))/2)
        observationArr = []
        actionArr = []

        for i in range(start, stop+1):
            dataAct = np.load(self.path + f"//actions[{i}].npy")
            dataObs = np.load(self.path + f"//observations[{i}].npy")

            actionArr.append(dataAct)
            observationArr.append(dataObs)

        return np.array(actionArr), np.array(observationArr)


if __name__ == "__main__":
    dataHandler = CreateData("./dataset/firstdata")
    env = gym.make("CarRacing-v1")
    while True:
        observations, actions = [], []
        env.reset()
        isEscPressed = False
        while True:
            env.render()
            action, isEscPressed = dataHandler.keyPress()
            observation, reward, done,info = env.step(action)
            observations.append(observation)
            actions.append(action)
            if done or isEscPressed:
                break

        # save data here
        if not isEscPressed:
            dataHandler.saveData(actions, observations)
