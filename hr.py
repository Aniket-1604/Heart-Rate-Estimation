import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
rgb=pd.read_csv("ppg-rgb.csv")
def gethr(arr,fps):
  plt.figure(1)
  plt.ylabel('Mean')
  plt.xlabel('time')
  plt.title("Value of mean")
  #plt.subplot(4,1,1)
  plt.plot(arr,'r')
  arr=np.array(arr)
  N=len(arr)
  T=1/fps
  arr_n=(arr-np.mean(arr))/np.std(arr)
  x = np.linspace(0.0, N*T, N, endpoint=False)
  yf = fft(arr_n)
  xf = fftfreq(N, T)[:N//2]
  x_plot=xf
  y_plot=2.0/N * np.abs(yf[0:N//2])
  plt.figure(2)
  plt.ylabel('Amplitude')
  plt.xlabel('frequency')
  plt.title("Fourier transform of above signal")
  #plt.subplot(4,1,2)
  plt.plot(x_plot,y_plot,'g')
  fil=np.ones([5])/5
  fil_y=np.convolve(y_plot,fil,mode='same')
  plt.figure(3)
  plt.title("Smoothened fourier")
  #plt.subplot(4,1,3)
  plt.ylabel('amplitude')
  plt.xlabel('frequency')
  plt.plot(x_plot,fil_y)


  valid=[]
  for i in range(int(len(x_plot)/7)):
    if x_plot[i]>1 and x_plot[i]<2:
      valid.append(i)
  #print(valid[0]) 
  #print(valid[-1])
  st=valid[0]
  end=valid[-1]
  plt.figure(4)
  #plt.subplot(4,1,4)
  plt.ylabel('Amplitude')
  plt.xlabel('frequency filtered')
  plt.plot(x_plot[st:end],y_plot[st:end],'r')
  plt.title("Frequency of intrest")
  print("heart_rate: " , x_plot[st+np.argmax(y_plot[st:end])]*60)
  plt.show()
#fp=float(input("Enter FPS of video"))
gethr(rgb['g'],29.73)