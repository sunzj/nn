#! /usr/bin/env python

import numpy as np

simples = [1,0]

def sigmd(x):
    return ( 1 / ( 1 + np.exp(-x) ) )

def dsigmd(x):
    return (sigmd(x)*( 1- sigmd(x)))

#
#Cross-entropy loss
#

def loss(y_lab,f):
    return (y_lab*np.log(f) + (1-y_lab)*np.log(1 - f))*(-1)

def nn(w,b,x):
    return sigmd(w*x+b)

def dw(x_lab,f,y_lab):
    return x_lab*(f - y_lab)

def db(f,y_lab):
    return (f - y_lab)

w = 1.0
b = 2.0
V_w_1 = 0.0
V_b_1 = 0.0

for i in range(300):
   d_w = 0.0
   d_b = 0.0
   for j in range(2):
       x_label = j
       y_label = simples[j]
       tmp_w   = w - 0.9 * V_w_1
       tmp_b   = b - 0.9 * V_b_1
       forward_nn = nn(tmp_w,tmp_b,x_label)
       l  = loss(y_label,forward_nn)
       d_w += dw(x_label,forward_nn,y_label)
       d_b += db(forward_nn,y_label)
   V_w = 0.9 * V_w_1 + 0.15 * (d_w / 20)
   w   = w - V_w
   V_b = 0.9 * V_b_1 + 0.15 * (d_b / 20)
   b   = b - V_b
   V_w_1 = V_w
   V_b_1 = V_b

   if((i%10) == 0):
       print('Now w: %f, b: %f,loss: %f' % (w,b,l))
       print('verify 0: %f,1: %f' % (nn(w,b,0),nn(w,b,1)))
