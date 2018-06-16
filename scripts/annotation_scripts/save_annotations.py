"""
save_annotations.py

Code for saving image annotations from crowdflower

@author: David Van Valen
"""

"""
Import python packages
"""

import os
import pandas as pd
import urllib

"""
Load csv file
"""

csv_direc = '/home/vanvalen/Data/annotations/'
csv_name = 'f1207357.csv'

csv_file = os.path.join(csv_direc, csv_name)
df = pd.DataFrame.from_csv(csv_file)

urls = df.loc[:,['annotation', 'image_url', 'broken_link']]
for index, row in df.iterrows():
	print row

	annotation_url = row['annotation'][8:-2]
	image_url = row['image_url']
	broken_link = row['broken_link'] 

	print broken_link
	# Get image_name
	if broken_link is False:
		image_url_split = image_url.split("/")
		image_name = image_url_split[-1][:-4] + "_annotation.png"
		image_path = os.path.join(csv_direc, 'nuclear', 'hela', 'set3', image_name)

		print image_url_split

		# Download annotation	
		image = urllib.URLopener()
		image.retrieve(annotation_url, image_path)
