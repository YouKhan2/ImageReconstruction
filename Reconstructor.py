import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math as math
from ipywidgets import IntProgress
from IPython.display import display
import time


def grad_x(img, adjoint):
    sx , sy = np.shape(img)
    diff_x = np.copy(img)
    
    if adjoint==0:
        for x in range(sx):
            if x==sx-1:
                xnext=0
            else:
                xnext=x+1
            for y in range(sy):
                diff_x[x,y] = img[xnext,y]- img[x,y]
    else:
        for x in range(sx):
            if x==0:
                xprev=sx-1
            else:
                xprev=x-1
            for y in range(sy):
                diff_x[x,y] = img[xprev,y]- img[x,y]
    
    return diff_x

def verifie_grad_x(in1,in2):
    tmp1 = sum(sum(grad_x(in1, 0) * in2))
    tmp2 = sum(sum(in1 * grad_x(in2, 1)))
    print(tmp1, ' = ', tmp2,' ?')
    
    return tmp1==tmp2
    
def grad_y(img, adjoint):
    sx , sy = np.shape(img)
    diff_y =  np.copy(img)

    if adjoint==0:
    
        for y in range(sy):
            if y==sy-1:
                ynext=0
            else:
                ynext=y+1
            for x in range(sx):
                diff_y[x,y] = img[x,ynext]- img[x,y]
    else:
        for y in range(sy):
            if y==0:
                yprev=sy-1
            else:
                yprev=y-1
            for x in range(sx):
                diff_y[x,y] = img[x,yprev]- img[x,y]
    
    return diff_y

def verifie_grad_y(in1,in2):
    tmp1 = sum(sum(grad_y(in1, 0) * in2))
    tmp2 = sum(sum(in1 * grad_y(in2, 1)))
    print(tmp1, ' = ', tmp2,' ?')
    
################################
def phi(img,tau):
    tmp = np.abs( np.copy(img))-tau/2
    
    return tmp * tmp * (tmp>0)

def phi_p(img,tau):
    dom = (np.abs(img)-tau/2 ) >0
    
    return (2*img - np.sign(img)*tau)*dom

def grad_E(out,v,tau,Lambda):
    tmpx = grad_x(out,0)
    tmpx1 = grad_x(tmpx,1)
    tmpy = grad_y(out,0)
    tmpy1 = grad_y(tmpy,1)
    
    grad = 2 * (tmpx1 + tmpy1) + Lambda * phi_p(out-v,tau)
    
    return grad

def E(out,v,tau,Lambda):
    sx , sy = np.shape(out)
    gx = np.linalg.norm( grad_x(out,0), ord='fro')
    gy = np.linalg.norm( grad_y(out,0), ord='fro')
    data = np.linalg.norm( phi(out - v,tau) , ord='fro')
    
    return (gx*gx + gy*gy + Lambda *data)/(sx*sy)
class reconstructor():
    def __init__(self,image=np.zeros((16,16))):
        self.image=image

    #denoising images
    def denoise(self):
        Lambda = 10
        v=self.image
        fftV = np.fft.fft2(v)
        fftOut=np.copy(fftV)

        sx , sy = np.shape(v)
        for x in range(sx):
            for y in range(sy):
                fftOut[x][y]= Lambda * fftV[x][y]/(Lambda + 4 - 2*(math.cos(2*math.pi*x/sx)+math.cos(2*math.pi*y/sy)))

        out =  np.fft.ifft2(fftOut) 
        out = out.real

        return out
    
    #dequantify images
    
    def dequantify(self):
        v=self.image
        tau=30
        Lambda=10
        nbIter=100

        step = 1/(8+2*Lambda)

        out = np.copy(v)

        f = IntProgress(min=0, max=nbIter) # instantiate the bar
        display(f) # display the bar

        for it in range(nbIter):
            d = grad_E(out,v,tau,Lambda)
            out = out - step * d
            f.value += 1 # signal to increment the progress bar
            time.sleep(.1)
        return out