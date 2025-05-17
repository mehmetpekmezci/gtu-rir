import math
import logging
import csv
import glob
import sys
import os
import argparse
import numpy as np
import time
import random
import pickle
import matplotlib
import pymeshlab as ml
import PIL
import traceback
import trimesh
import copy
import pyglet
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
from numba import jit
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv




def generate_ray_directions(RESOLUTION_QUOTIENT=16):
     alfas=z_directions=np.arange(0               ,    2*np.pi     ,    np.pi/(RESOLUTION_QUOTIENT/2)) # 256 values
     betas=y_directions=np.arange(-np.pi/2,    np.pi/2     ,    np.pi/RESOLUTION_QUOTIENT) #  256 values
     unit_vector=[1,0,0]
     ray_directions={}
     for i in range(alfas.shape[0]):
         if i not in ray_directions:
             ray_directions[i]={}
         for j in range(betas.shape[0]):
                   alfa=z_directions[i]
                   beta=y_directions[j]
                   gamma=0
                   rotation_around_z= [  [ np.cos(alfa) , -np.sin(alfa), 0 ],  [ np.sin(alfa), -np.cos(alfa), 0 ],  [ 0 , 0, 1] ]
                   rotation_around_y= [  [ np.cos(beta) ,0, np.sin(beta)],  [0,1,0 ],  [-np.sin(beta) , 0, np.cos(beta)] ]
                   rotation_around_x= [  [1,0,0], [ 0,np.cos(gamma) , -np.sin(gamma) ],  [ 0, np.sin(gamma), np.cos(gamma) ] ]
                   rotation=np.matmul(rotation_around_z,np.matmul(rotation_around_y,rotation_around_x))
                   ray_direction=np.matmul(unit_vector,rotation)
                   ray_directions[i][j]=ray_direction

     return ray_directions

def generate_ray_directions_(RESOLUTION_QUOTIENT=10):
     alfas=z_directions_360=np.arange(0               ,    2*np.pi     ,    np.pi/(180/RESOLUTION_QUOTIENT))   # 360 VALUES if RESOLUTION_QUOTİENT==1, 36 VALUEs if RESOLUTION_QUOTİENT=10
     betas=y_directions_180=np.arange(-np.pi/2,    np.pi/2     ,    np.pi/(180/RESOLUTION_QUOTIENT)) # 180 VALUES if RESOLUTION_QUOTİENT==1, 18 VALUEs if RESOLUTION_QUOTİENT=10
     unit_vector=[1,0,0]
     ray_directions={}
     for i in range(alfas.shape[0]):
         if i not in ray_directions:
             ray_directions[i]={}
         for j in range(betas.shape[0]):
                   alfa=z_directions_360[i]
                   beta=y_directions_180[j]
                   gamma=0 
                   rotation_around_z= [  [ np.cos(alfa) , -np.sin(alfa), 0 ],  [ np.sin(alfa), -np.cos(alfa), 0 ],  [ 0 , 0, 1] ]
                   rotation_around_y= [  [ np.cos(beta) ,0, np.sin(beta)],  [0,1,0 ],  [-np.sin(beta) , 0, np.cos(beta)] ]
                   rotation_around_x= [  [1,0,0], [ 0,np.cos(gamma) , -np.sin(gamma) ],  [ 0, np.sin(gamma), np.cos(gamma) ] ]
                   rotation=np.matmul(rotation_around_z,np.matmul(rotation_around_y,rotation_around_x))
                   #yaw=rotation_around_z= [  [ cos_alfa , - sin_alfa, 0 ],  [ sin_alfa , - cos_alfa, 0 ],  [ 0 , 0, 1] ]
                   #pitch=rotation_around_y= [  [ cos_beta , 0,  sin_beta],  [ 0, 1, 0 ],  [ -sin_beta , 0, cos_beta] ].
                   #roll=rotation_around_x= [ [1,0,0 ], [0, cos_gamma,-sin_gamma], [0,sin_gamma,cos_gamma]]
                   #R=rotation_around_z * rotation_around_y * rotation_around_x
                   ray_direction=np.matmul(unit_vector,rotation)
                   ray_directions[i][j]=ray_direction

     return ray_directions


