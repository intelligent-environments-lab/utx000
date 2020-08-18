import matplotlib.pyplot as plt 
from datetime import datetime


class single_var:
	'''
	Visualizations that incorporate only one variable
	'''

	def __init__(self):
		pass

	def timeseries(self, t, y, **kwargs):
		'''
		Plots a time series of the data
		'''

		if 'figsize' in kwargs.keys():
			fig, ax = plt.subplot(111,figsize=kwargs['figsize'])
		else:
			fig, ax = plt.subplot(111,figsize=(16,8))

		ax.plot()
