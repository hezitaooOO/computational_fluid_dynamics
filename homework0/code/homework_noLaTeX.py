import sys
import numpy as np
import scipy
import pylab as plt
from pdb import set_trace as keyboard

# Extracting PI constant
PI = np.pi

figure_folder = "../report/"

Nx = 101
x = np.linspace(0,2*PI,Nx)
y_cos = np.cos(x)
y_sin = np.sin(x)

Plot_CosineAndSine = 'y'
Plot_SomethingElse = 'y'

if Plot_CosineAndSine == 'y':

  figure_name = "cosine_and_sine_noLaTeX.pdf"

  figwidth       = 18
  figheight      = 6
  lineWidth      = 3
  textFontSize   = 28
  gcafontSize    = 30
  
  fig = plt.figure(0, figsize=(figwidth,figheight))
  ax_left   = fig.add_subplot(1,2,1)
  ax_right  = fig.add_subplot(1,2,2)
  
  ax = ax_left
  plt.axes(ax)
  ax.plot(x,y_cos,'-r',linewidth=lineWidth)
  plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
  plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
  ax.grid('on',which='both')
  #  ax.set_xticks()
  #  ax.set_xlim()
  #  ax.set_yticks()
  #  ax.set_ylim()
  ax.set_xlabel("x axis",fontsize=textFontSize)
  ax.set_ylabel("y axis",fontsize=textFontSize,rotation=90)

  ax = ax_right
  plt.axes(ax)
  ax.plot(x,y_sin,'-k',linewidth=lineWidth)
  plt.setp(ax.get_xticklabels(),fontsize=gcafontSize)
  plt.setp(ax.get_yticklabels(),fontsize=gcafontSize)
  ax.grid('on',which='both')
  ax.set_xlabel("x axis",fontsize=textFontSize)
  ax.set_ylabel("y axis",fontsize=textFontSize,rotation=90)

  figure_file_path = figure_folder + figure_name
  print "Saving figure: " + figure_file_path
  plt.tight_layout()
  plt.savefig(figure_file_path)
  plt.close()

if Plot_SomethingElse == 'y':

  figure_name = "other_figure.pdf"
  # Put here other plots


 
