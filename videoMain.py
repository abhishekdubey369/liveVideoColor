#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
from guidedfilter import guided_filter
from PIL import Image

np.set_printoptions(threshold=np.inf)

L = 256


def get_illumination_channel(I, w):
    M, N, _ = I.shape
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')

    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))
    
    for i, j in np.ndindex(darkch.shape):
        #darkch[i, j]  =  np.min(padded[i:i + w, j:j + w, :]) # patch level
        darkch[i, j]  =  np.min(padded[i, j, :]) # pixel level
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :])
    
    return darkch, brightch
    
    
def get_atmosphere(I, brightch, p):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)
    flatbright = brightch.ravel() # make array flatten
    
    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # find top M * N * p indexes. argsort() returns sorted (ascending) index.
    
    # return the mean intensity for each channel
    A = np.mean(flatI.take(searchidx, axis=0),dtype=np.float64, axis=0) # 'take' get value from index.
    
    return A


def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    
    init_t = (brightch-A_c)/(1.-A_c) # original

    ### maybe this make image too bright (almost hazy)
    #bright_norm = (brightch - np.min(brightch))/(np.max(brightch) - np.min(brightch))
    #J_bright = (bright_norm**0.3)*(55./255.) + (200./255.) # referred to "Automatic local exposure correction using bright channel prior for under-exposed images"
    #init_t = (brightch-A_c)/(J_bright-A_c)

    ### 3 channels version ###
    
    #init_t = np.zeros((brightch.shape[0], brightch.shape[1], 3))

    #for i in range(3):
        #init_t[:,:,i] = (brightch-A[i])/(1.-A[i])
        #init_t[:,:,i] = (init_t[:,:,i] - np.min(init_t[:,:,i]))/(np.max(init_t[:,:,i]) - np.min(init_t[:,:,i]))

    
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # min-max normalization.

    
def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im3 = np.empty(I.shape, I.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = I[:,:,ind]/A[ind]

    dark_c, _ = get_illumination_channel(im3, w)
    dark_t = 1 - omega*dark_c
    #dark_t = (dark_t - np.min(dark_t))/(np.max(dark_t) - np.min(dark_t)) # min-max normalization. But, it's no need by the above equation.
    
    #print('dark channel prior between [%.4f, %.4f]' % (dark_c.min(), dark_c.max())) #dark channel prior between [0.0000, 1.5222]
    #print('initial dark transmission rate between [%.4f, %.4f]' % (dark_t.min(), dark_t.max()))
    
    corrected_t = init_t
    diffch = brightch - darkch
    
    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i,j]<alpha):
                corrected_t[i,j] = dark_t[i,j]*init_t[i,j]
                
    #corrected_t = (corrected_t - np.min(corrected_t))/(np.max(corrected_t) - np.min(corrected_t)) # min-max normalization. this is not good.
    
    return np.abs(corrected_t), dark_t
    

def get_final_image(I, A, corrected_t, tmin):
    corrected_t_broadcasted = np.broadcast_to(corrected_t[:,:,None], (corrected_t.shape[0], corrected_t.shape[1], 3))
    J = (I-A)/(np.where(corrected_t_broadcasted < tmin, tmin, corrected_t_broadcasted)) + A
    #J = (I-A)/(np.where(corrected_t < tmin, tmin, corrected_t)) + A # this is used when corrected_t has 3 channels
    print('J between [%.4f, %.4f]' % (J.min(), J.max()))
    
    return (J - np.min(J))/(np.max(J) - np.min(J)) # min-max normalization.
    

def dehaze(frame, tmin, w, alpha, omega, p, eps):
    m, n, _ = frame.shape
    Idark, Ibright = get_illumination_channel(frame, w)
    
    print("dark max:{} min:{}".format(np.max(Idark), np.min(Idark))) # dark max:0.776470588235 min:0.0
    print("bright max:{} min:{}".format(np.max(Ibright), np.min(Ibright))) # bright max:0.988235294118 min:0.00392156862745
    
    white = np.full_like(Idark, L - 1)
    
    A = get_atmosphere(frame, Ibright, p)
    print('atmosphere:{}'.format(A))
    
    #################################################################
    
    init_t = get_initial_transmission(A, Ibright)
    # cv2.imshow('Transmission Init', init_t*white)
    print('initial (bright) transmission rate between [%.4f, %.4f]' % (init_t.min(), init_t.max()))
    
    J_init = get_final_image(frame, A, init_t, tmin)
    # cv2.imshow('J Init', J_init)
    
    #################################################################
    
    corrected_t, dark_t = get_corrected_transmission(frame, A, Idark, Ibright, init_t, alpha, omega, w)
    # cv2.imshow('Transmission Corrected', corrected_t*white)
    print('corrected transmission rate between [%.4f, %.4f]' % (corrected_t.min(), corrected_t.max()))
    
    cv2.imshow('Transmission Dark', dark_t*white)
    print('dark transmission rate between [%.4f, %.4f]' % (dark_t.min(), dark_t.max()))
    
    J_corrected = get_final_image(frame, A, corrected_t, tmin)
    cv2.imshow('J Corrected', J_corrected)
    
    J_dark = get_final_image(frame, A, dark_t, tmin)
    cv2.imshow('J Dark', J_dark)
    
    #################################################################
    
    # guided filter
    normI = (frame - frame.min()) / (frame.max() - frame.min())  # min-max normalize I
    refined_t = guided_filter(normI, corrected_t, w, eps)
    refined_dark_t = guided_filter(normI, dark_t, w, eps)
    
    #refined_t = (refined_t - np.min(refined_t))/(np.max(refined_t) - np.min(refined_t)) # min-max normalization.
    #refined_dark_t = (refined_dark_t - np.min(refined_dark_t))/(np.max(refined_dark_t) - np.min(refined_dark_t)) # min-max normalization.
    
    # cv2.imshow('Refined Transmission', refined_t*white)
    print('refined transmission rate between [%.4f, %.4f]' % (refined_t.min(), refined_t.max()))
    
    J_refined = get_final_image(frame, A, refined_t, tmin)
    cv2.imshow('J Refined', J_refined)
    
    # cv2.imshow('Refined Dark Transmission', refined_dark_t*white)
    print('refined dark transmission rate between [%.4f, %.4f]' % (refined_dark_t.min(), refined_dark_t.max()))
    
    J_refined_dark = get_final_image(frame, A, refined_dark_t, tmin)
    # cv2.imshow('J Refined Dark', J_refined_dark)


if __name__ == '__main__':
    tmin=0.1   # minimum value for t to make J image
    w=15       # window size, which determine the corseness of prior images
    alpha=0.4  # threshold for transmission correction. range is 0.0 to 1.0. The bigger number makes darker image.
    omega=0.75 # this is for dark channel prior. change this parameter to arrange dark_t's range. 0.0 to 1.0. bigger is brighter
    p=0.1      # percentage to consider for atmosphere. 0.0 to 1.0
    eps=1e-3   # for J image
    
    cap = cv2.VideoCapture(0)  # Open the default camera
    
    while True:
        ret, frame = cap.read()  # Read a frame from the video capture
        
        dehaze(frame, tmin, w, alpha, omega, p, eps)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
            break
    
    cap.release()  # Release the video capture
    cv2.destroyAllWindows()
