"""
save_image_urls.py

Code for creating csv files with links to the google or amazon aws bucket containing the training data

@author: David Van Valen
"""

"""
Import python packages
"""

import os
import pandas as pd

"""
Create dataframe
"""

list_of_urls = []
frame_number = []
horizontal_quadrant = []
vertical_quadrant = []

cell_type = 'hela'
data_type = 'nuclear'
set_number = 'set4'
for j in xrange(12):
	base_direc = os.path.join('https://storage.googleapis.com/daves-new-bucket/data/', data_type, cell_type, set_number)
	# base_direc = os.path.join('https://s3-us-west-1.amazonaws.com/daves-amazons3-bucket/data/', data_type, cell_type, set_number)
# 
	for i in xrange(1):
		for k in xrange(1):
			if data_type == 'nuclear':
				prefix = 'nuclear'
			elif data_type == 'cytoplasm':
				prefix = 'phase'

			file_name = prefix + '_' + str(j) + '.png'
			# file_name = prefix + '_' + str(j) + '_quad_' + str(i) + '_' + str(k) + '.png'
			list_of_urls += [os.path.join(base_direc, file_name)]
			frame_number += [j]
			horizontal_quadrant += [i]
			vertical_quadrant += [k]

data = {'image_url': list_of_urls, 'frame_number': frame_number, 'horizontal_quadrant': horizontal_quadrant, 'vertical_quadrant': vertical_quadrant}
dataframe = pd.DataFrame(data = data)

"""
Save as csv
"""
csv_name = cell_type + '_' + data_type + '_' + set_number + '.csv'
dataframe.to_csv(csv_name, index = False)