#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

import os
import datetime # for logging timestamp

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.2 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
BOOTSTRAP_K = 10 # number of bootstrap heads

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

# https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)

def buildmodel(bootstrap_head = None):
    if bootstrap_head:
        print("Now we build the model for bootstrap_head = %d" % (bootstrap_head))
    else:
        print("Now we build the model")
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
   
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    print("We finish building the model")
    return model

def trainNetwork(models,args):
    SAME_LINE = False
    # open up a game state to communicate with emulator

    log_file_name = datetime.datetime.now().strftime("log_%Y_%m_%d_%H_%M_%S.txt")
    log_file = open(log_file_name, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log_file)
    

    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, curr_score = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

    

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weights")
        for i in range(BOOTSTRAP_K):
            if os.path.isfile("model_%d.h5" % (i)):
                models[i].load_weights("model_%d.h5" % (i))
                print ("Weight for head %d load successfully", (i))
            models[i].compile(loss='mse',optimizer=Adam(lr=LEARNING_RATE))
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    t = 0
    total_reward = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])

        #choose bootstrap head at random
        chosen = np.random.randint(BOOTSTRAP_K)

        #instead of epsilon greedy, we choose the best action
        #that maximizes Q_chosen
        if t % FRAME_PER_ACTION == 0:
            q = models[chosen].predict(s_t)       #input a stack of 4 images, get the prediction
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[max_Q] = 1

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal, curr_score = game_state.frame_step(a_t)
        terminal_check = terminal

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        #calculate bootstrap mask
        #the authors use Bernoulli(0.5), but that essentially means
        #choose with 0.5 probability on each head
        mask = np.random.choice(2, BOOTSTRAP_K, p=[0.5,]*2)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal, mask))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                mask = minibatch[i][5]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                targets[i] = models[chosen].predict(state_t)  # Hitting each buttom probability on Q_chosen
                Q_sa = models[chosen].predict(state_t1)

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # run on those Q which are in the boostrap mask
            for idx in range(BOOTSTRAP_K):
                if mask[idx] == 1:
                    loss += models[idx].train_on_batch(inputs, targets)

        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save models")
            for i in range(BOOTSTRAP_K):
                models[i].save_weights("model_%d.h5" % (i), overwrite=True)
                with open("model_%d.json" % (i), "w") as outfile:
                    json.dump(models[i].to_json(), outfile)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        printInfo(t, state, action_index, r_t, Q_sa, loss, chosen)

        if not SAME_LINE:
            score_file = open("scores","aw") 
            score_file.write(str(curr_score)+"\n")
            score_file.close()
            SAME_LINE = True
        else:
            score_file = open("scores","r")
            score_file_lines = score_file.readlines()[:-1]
            score_file.close()
            score_file = open("scores","w")
            score_file.writelines(score_file_lines)
            score_file.write(str(curr_score)+"\n")
            score_file.close()

        if terminal_check:
            print("Total rewards: ", total_reward) 
            out_file = open("total_reward","aw") 
            out_file.write(str(total_reward)+"\n")
            out_file.close()    
            total_reward = 0
            SAME_LINE = False
        else:
            total_reward = total_reward + r_t

    print("Episode finished!")
    print("************************")

def printInfo(t, state, action_index, r_t, Q_sa, loss, chosen):
    print("TIMESTEP", t, "/ STATE", state, \
          "/ ACTION", action_index, "/ REWARD", r_t, \
          "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss, "/ CHOSEN HEAD", chosen)

def playGame(args):
    models = [buildmodel(i) for i in range(BOOTSTRAP_K)]
    trainNetwork(models,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
