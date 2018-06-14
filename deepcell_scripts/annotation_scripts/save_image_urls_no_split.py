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

# cell_types = ["HEK293", "HeLa-S3", "MCF10A", "MouseBrain", "NIH-3T3", "RAW264.7"]
# list_of_number_of_sets = [3, 8, 1, 1, 3, 8]
# list_of_number_of_frames = [71, 45, 69, 41, 71, 45]

cell_types = ["MouseBrain"]
list_of_number_of_sets = [7]
list_of_number_of_frames = [30]

for cell_type, number_of_sets, number_of_frames in zip(cell_types, list_of_number_of_sets, list_of_number_of_frames):
	for set_number in xrange(number_of_sets):
		list_of_urls = []

		set_number = 'set' + str(set_number)
		data_type = 'nuclear'

		base_direc = os.path.join('https://storage.googleapis.com/daves-new-bucket/data/', data_type, cell_type, set_number, 'Montage')
		
		for i in xrange(4):
			for j in xrange(4):
				# for j in xrange(number_of_frames):
				file_name = 'montage' + '_' + str(i) + '_' + str(j) + '.png'
				list_of_urls += [os.path.join(base_direc, file_name)]

		data = {'image_url': list_of_urls, 'cell_type': cell_type, 'set_number': set_number}
		dataframe = pd.DataFrame(data=data)
		csv_name = os.path.join('/data/training_data_csv/' + cell_type + '_' + data_type + '_' + set_number + '.csv')
		dataframe.to_csv(csv_name, index = False)

