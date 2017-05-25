#! /usr/bin/env python

import numpy as np

y_l  = np.array([[0],[0],[0],[1]])
x_l  = np.array([[0,0],[0,1],[1,0],[1,1]])
w    = np.array([[12],[30]])
b    = np.array([[1]])

def loss( y_l , y ):
    l = 0.0
    l  += y_l * np.log( y ) + ( 1-y_l ) * np.log( 1 - y )
    return l*(-1)

def sigmd( x ):
    return ( 1 / ( 1 + np.exp( -x ) ) )

def ddsigmd( x ):
    return sigmd( x ) * ( 1 - sigmd(x) )

def nn( x, w, b ):
    z = 0.0
    z = np.dot(x,w) + b
    return sigmd(sum(z))

def dw(x,y,y_l):
    tdw = x * ( y - y_l )
    dw = tdw.reshape(2,1)
    return dw

def db(y,y_l):
    return ( y-y_l )

v_w    = np.array([[0],[0]])
v_b    = 0.0
v_w_1  = np.array([[0],[0]])
v_b_1  = 0.0
temp_w = np.array([[0],[0]])
temp_b = 0.0

batch_size = x_l.shape[0]
epochs     = 1000
 
for i in range(epochs):
    d_w = np.array([[0.0],[0.0]])
    d_b = 0.0
    temp_w = w - 0.9*v_w_1
    temp_b = b - 0.9*v_b_1
    for t in range(batch_size):
        forward = nn(x_l[t],temp_w,temp_b)
        d_w += dw(x_l[t],forward,y_l[t])
        d_b += db(forward,y_l[t])
    v_w = 0.9*v_w_1 + 0.15 * d_w / batch_size
    v_b = 0.9*v_b_1 + 0.15 * d_b / batch_size
    w   = w - v_w
    b   = b - v_b
    v_w_1 = v_w
    v_b_1 = v_b

    if i%100 == 0 :
        for v in range(x_l.shape[0]):
            forward = nn(x_l[v],w,b)
            l       = loss(y_l[v],forward)
            print(x_l[v],forward,l)
  
