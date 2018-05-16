"""
plt_error.py
Plot the training and validation error from the npz file

Run command:
	python plt_error.py

@author: David Van Valen
"""

import numpy as np 
import matplotlib.pyplot as plt 
import os

#loss_hist_file is full path to .npz loss history file
#saved_direc is full path to directory where you want to save the plot
#plot_name is the name of plot

def plt_error(loss_hist_file, saved_direc, plot_name):
	loss_history = np.load(loss_hist_file) 
	loss_history = loss_history['loss_history'][()] 
	print type(loss_history)


	err = np.subtract(1, loss_history['acc'])
	val_err = np.subtract(1, loss_history['val_acc'])

	epoch = np.arange(1, len(err) + 1, 1)
	plt.plot(epoch, err)
	plt.plot(epoch, val_err)
	plt.title('Model Error')
	plt.xlabel('Epoch')
	plt.ylabel('Model Error')
	plt.legend(['Training error','Validation error'], loc='upper right')

	filename = os.path.join(saved_direc, plot_name)
	plt.savefig(filename, format = 'pdf')

loss_hist_file = os.path.join('/home/vanvalen/Data/MIBI/trained_networks/', '2017-10-31_Samir_1e4_61x61_bn_feature_net_61x61_0.npz')
saved_direc = os.path.join('/home/vanvalen/deepcell-tf')
plot_name = 'MIBI.pdf'

plt_error(loss_hist_file, saved_direc, plot_name)
