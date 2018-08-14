import numpy as np
from skimage.measure import regionprops
from skimage.transform import resize
from scipy.optimize import linear_sum_assignment

from tensorflow.python.keras import backend as K

class cell_tracker():
    def __init__(self, 
                 movie, 
                 annotation, 
                 model, 
                 crop_dim=32, 
                 death=0.5, 
                 birth=0.9,
                 max_distance=200,
                 track_length=1,
                 data_format=None):
        
        # TODO: Use a model that is served by tf-serving, not one on a local machine
        
        if data_format is None:
            data_format = K.image_data_format()
            
        self.x = movie
        self.y = annotation
        self.model = model
        self.crop_dim = crop_dim
        self.death = death
        self.birth = birth
        self.max_distance = max_distance
        self.data_format = data_format
        self.track_length = track_length
        self.channel_axis = 0 if data_format == 'channels_first' else -1
        
        # Clean up annotations
        self._clean_up_annotations()
        
        # Initialize tracks
        self._initialize_tracks()
        
    def _clean_up_annotations(self):
        """
        This function relabels every frame in the label matrix
        Cells will be relabeled 1 to N
        """
        y = self.y
        number_of_frames = self.y.shape[0]
        
        for frame in range(number_of_frames):
            unique_cells = np.unique(y[frame])
            y_frame_new = np.zeros(y[frame].shape)
            for new_label, old_label in enumerate(list(unique_cells)):
                y_frame_new[y[frame] == old_label] = new_label
            y[frame] = y_frame_new
        self.y = y
        
    def _initialize_tracks(self):
        """
        This function intializes the tracks
        Tracks are stored in a dictionary
        """
        self.tracks = {}
        unique_cells = np.unique(self.y[0])
        # Remove background that has value 0
        unique_cells = np.delete(unique_cells, np.where(unique_cells == 0))
        
        for track_counter, label in enumerate(unique_cells):
            # Get the appearance
            appearance, centroid = self._get_appearances(self.x, self.y, [0], [label])
            # Save track information
            self.tracks[track_counter] = {
                'label': label,
                'frames': [0],
                'daughters': [],
                'parent': None,
                'appearances': appearance,
                'centroids': centroid
                }
            # Start a tracked label array
            self.y_tracked = self.y[0]
        return None
        

    def _get_cost_matrix(self, frame):
        """
        This function uses the model to create the cost matrix for 
        assigning the cells in frame to existing tracks.
        """
        
        # Initialize matrices
        number_of_tracks = len(self.tracks.keys())
        number_of_cells = np.amax(self.y[frame])
        
        cost_matrix = np.zeros((number_of_tracks + number_of_cells, number_of_tracks + number_of_cells), dtype=K.floatx())
        assignment_matrix = np.zeros((number_of_tracks, number_of_cells), dtype=K.floatx())
        birth_matrix = np.zeros((number_of_cells, number_of_cells), dtype=K.floatx())
        death_matrix = np.zeros((number_of_tracks, number_of_tracks), dtype=K.floatx())
        mordor_matrix = np.zeros((number_of_cells, number_of_tracks), dtype=K.floatx()) # Bottom right matrix - no assignments here
            
        # Compute assignment matrix
        track_appearances = self._fetch_track_appearances()
        track_centroids = self._fetch_track_centroids()
        
        cell_appearances = np.zeros((number_of_cells, 1, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), dtype=K.floatx())
        cell_centroids = np.zeros((number_of_cells, 2), dtype=K.floatx())
        
        for cell in range(number_of_cells):
            cell_appearance, cell_centroid = self._get_appearances(self.x, self.y, [frame], [cell+1])
            cell_appearances[cell] = cell_appearance
            cell_centroids[cell] = cell_centroid
        
        # Compute assignment matrix - Initialize and get model inputs
        input_1 = np.zeros((number_of_tracks, number_of_cells, self.track_length, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), dtype = K.floatx())
        input_2 = np.zeros((number_of_tracks, number_of_cells, 1, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), dtype = K.floatx())
        input_3 = np.zeros((number_of_tracks, number_of_cells, self.track_length, 2))
        input_4 = np.zeros((number_of_tracks, number_of_cells, 1, 2))
        
        for track in range(number_of_tracks):
            for cell in range(number_of_cells):
                input_1[track,cell,:,:,:,:] = track_appearances[track]
                input_2[track,cell,:,:,:,:] = cell_appearances[cell]
                
                centroids = np.concatenate([track_centroids[track], cell_centroids[[cell]]], axis=0)
                distances = np.diff(centroids, axis=0)
                zero_pad = np.zeros((1,2), dtype=K.floatx())
                distances = np.concatenate([zero_pad, distances], axis=0)
                
                # Make sure the distances are all less than max distance
                for j in range(distances.shape[0]):
                    dist = distances[j,:]
                    if np.linalg.norm(dist) > self.max_distance:
                        distances[j,:] = distances[j,:]/np.linalg.norm(dist)*self.max_distance
                
                input_3[track,cell,:,:] = distances[0:-1,:]
                input_4[track,cell,:,:] = distances[-1,:]
        
        # Compute assignment matrix - reshape model inputs for prediction
        input_1 = np.reshape(input_1, (number_of_tracks*number_of_cells, self.track_length, self.crop_dim, 
                                       self.crop_dim, self.x.shape[self.channel_axis]))
        input_2 = np.reshape(input_2, (number_of_tracks*number_of_cells, 1, self.crop_dim,
                                      self.crop_dim, self.x.shape[self.channel_axis]))
        input_3 = np.reshape(input_3, (number_of_tracks*number_of_cells, self.track_length, 2))
        input_4 = np.reshape(input_4, (number_of_tracks*number_of_cells, 1, 2))
        model_input = [input_1, input_2, input_3, input_4]
        
        predictions = self.model.predict(model_input) #TODO: implement some splitting function in case this is too much data
        predictions = np.reshape(predictions, (number_of_tracks, number_of_cells, 3))
        assignment_matrix = predictions[:,:,0]
        
        # Compute birth matrix
        predictions_birth = predictions[:,:,2]
        birth_diagonal = 1-np.amax(predictions_birth, axis=0)
        #birth_diagonal = [self.birth]*number_of_cells
        birth_matrix = np.diag(birth_diagonal) + np.ones(birth_matrix.shape) - np.eye(number_of_cells)
        print(birth_diagonal)
        
        # Compute death matrix
        death_matrix = self.death * np.eye(number_of_tracks) + np.ones(death_matrix.shape) - np.eye(number_of_tracks)
        
        # Compute mordor matrix
        mordor_matrix = assignment_matrix.T
        
        # Assemble full cost matrix
        cost_matrix[0:number_of_tracks, 0:number_of_cells] = assignment_matrix
        cost_matrix[number_of_tracks:, 0:number_of_cells] = birth_matrix
        cost_matrix[0:number_of_tracks,number_of_cells:] = death_matrix
        cost_matrix[number_of_tracks:,number_of_cells:] = mordor_matrix
        return cost_matrix
    
    def _run_lap(cost_matrix):
        """
        This function runs the linear assignment function on a cost matrix.
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignments = np.stack([row_ind, col_ind], axis=1)
        return assignments
    
    def _update_tracks(self, assignments, frame):
        """
        This function will update the tracks if given the assignment matrix
        and the frame that was tracked.
        """
        number_of_tracks = len(self.tracks.keys())
        number_of_cells = np.amax(self.y[frame])
        
        existing_tracks = list(np.arange(len(self.tracks.keys())))
        cells_to_track = list(np.arange(np.amax(self.y[frame])))
        
        y_tracked_update = np.zeros((1, self.y.shape[1], self.y.shape[2], 1), dtype=K.floatx())
        for a in range(assignments.shape[0]):
            track, cell = assignments[a]
            
            # Take care of everything if cells are tracked
            if track < number_of_tracks and cell < number_of_cells:
                self.tracks[track]['frames'].append(frame)
                appearance, centroid = self._get_appearances(self.x, self.y, [frame], [cell])                
                self.tracks[track]['appearances'] = np.concatenate([self.tracks[track]['appearances'], appearance], axis = 0)
                self.tracks[track]['centroids'] = np.concatenate([self.tracks[track]['centroids'], centroid], axis=0)
                
                y_tracked_update[self.y[[frame]] == cell] = track
                
                existing_tracks.remove(track)
                cells_to_track.remove(cell)
                
            # Create a new track if there was a birth
            if track > number_of_tracks - 1 and cell < number_of_cells:
                print('Adding new track')
                new_track_id = np.amax(list(self.tracks.keys())) + 1
                self.tracks[new_track_id] = {}
                self.tracks[new_track_id]['label'] = new_track_id
                self.tracks[new_track_id]['frames'] = [frame]
                self.tracks[new_track_id]['daughters'] = []
                
                appearance, centroid = self._get_appearances(self.x, self.y, [frame], [cell])
                self.tracks[new_track_id]['appearances'] = appearance
                self.tracks[new_track_id]['centroids'] = centroid
                
                parent = self._get_parent(frame, cell)
                if parent is not None:
                    self.tracks[new_track_id]['parent'] = parent
                    self.tracks[parent]['daughters'].append(new_track_id)
                    
                cells_to_track.remove(cell)
                y_tracked_update[self.y[[frame]] == cell] = new_track_id
            
            # Dont touch anything if there was a cell that "died"
            if track < number_of_tracks and cell > number_of_cells - 1:
                existing_tracks.remove(track)
                continue
            
            # Update the tracked label array
            self.y_tracked = np.concatenate([self.y_tracked, y_tracked_update], axis=0)
            
        return None
    
    def _get_parent(self, frame, cell):
        """
        This function searches the tracks for the parent of a given cell
        It returns the parent cell's id or None if no parent exists.
        """
        track_appearances = self._fetch_track_appearances()
        track_centroids = self._fetch_track_centroids()
        
        cell_appearance, cell_centroid = self._get_appearances(self.x, self.y, [frame], [cell])
        cell_centroid = np.stack([cell_centroid]*track_centroids.shape[0], axis=0)
        cell_appearances = np.stack( [cell_appearance]*track_appearances.shape[0] , axis=0)
        
        all_centroids = np.concatenate([track_centroids, cell_centroid], axis=1)
        distances = np.diff(all_centroids, axis=1)
        zero_pad = np.zeros((track_centroids.shape[0],1,2), dtype = K.floatx())
        distances = np.concatenate([zero_pad,distances], axis=1)
        
        track_distances = distances[:,0:-1,:]
        cell_distances = distances[:,[-1],:]
                
        # Compute the probability the cell is a daughter of a track
        model_input = [track_appearances, cell_appearances, track_distances, cell_distances]
        predictions = model.predict(model_input)
        probs = predictions[:,2]
        
        # Find out if the cell is a daughter of a track
        max_prob = np.amax(probs)
        parent_id = np.where(probs == max_prob)[0][0]
        if max_prob > self.birth:
            parent = parent_id
        else:
            parent = None
        return parent
    
    def _fetch_track_appearances(self):
        """
        This function fetches the appearances for all of the existing tracks.
        If tracks are shorter than the track length, they are filled in with
        the first frame.
        """
        track_appearances = np.zeros((len(self.tracks.keys()), self.track_length, 
                                        self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), 
                                        dtype = K.floatx())
        
        for track in self.tracks.keys():
            app = self.tracks[track]['appearances']
            if app.shape[0] > self.track_length - 1:
                track_appearances[track] = app[-self.track_length:,:,:,:]
            else:
                track_length = app.shape[0]
                missing_frames = self.track_length - track_length
                frames = np.array(list(range(-1,-track_length-1,-1)) + [-track_length]*missing_frames)
                track_appearances[track] = app[frames,:,:,:]
                
        return track_appearances
    
    def _fetch_track_centroids(self):
        """
        This function fetches the centroids for all of the existing tracks.
        If tracks are shorter than the track length they are filled in with 
        the centroids from the first frame.
        """
        track_centroids = np.zeros((len(self.tracks.keys()), self.track_length, 2), dtype = K.floatx())
        
        for track in self.tracks.keys():
            cen = self.tracks[track]['centroids']
            if cen.shape[0] > self.track_length - 1:
                track_centroids[track] = cen[-self.track_length:,:]
            else:
                track_length = cen.shape[0]
                missing_frames = self.track_length - track_length
                frames = np.array(list(range(-1,-track_length-1,-1)) + [-track_length]*missing_frames)
                track_centroids[track] = cen[frames,:]
                    
        return track_centroids
    
    def _get_appearances(self, X, y, frames, labels):
        """
        This function gets the appearances and centroids of a list of cells.
        Cells are defined by lists of frames and labels. The i'th element of 
        frames and labels is the frame and label of the i'th cell being grabbed.
        """
        channel_axis = self.channel_axis
        if self.data_format == 'channels_first':
            appearance_shape = (X.shape[channel_axis],
                                len(frames),
                                self.crop_dim,
                                self.crop_dim)
        else:
            appearance_shape = (len(frames),
                                self.crop_dim,
                                self.crop_dim,
                                X.shape[channel_axis])
        centroid_shape = (len(frames), 2)

        # Initialize storage for appearances and centroids
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        centroids = np.zeros(centroid_shape, dtype=K.floatx())

        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            # Get the bounding box
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]
            props = regionprops(np.int32(y_frame == cell_label))
            minr, minc, maxr, maxc = props[0].bbox
            centroids[counter] = props[0].centroid

            # Extract images from bounding boxes
            if self.data_format == 'channels_first':
                appearance = X[:, frame, minr:maxr, minc:maxc]
                resize_shape = (X.shape[channel_axis], self.crop_dim, self.crop_dim)
            else:
                appearance = X[frame, minr:maxr, minc:maxc, :]
                resize_shape = (self.crop_dim, self.crop_dim, X.shape[channel_axis])

            # Resize images from bounding box
            max_value = np.amax([np.amax(appearance), np.absolute(np.amin(appearance))])
            appearance /= max_value
            appearance = resize(appearance, resize_shape)
            appearance *= max_value
            if self.data_format == 'channels_first':
                appearances[:, counter] = appearance
            else:
                appearances[counter] = appearance

        return appearances, centroids
        
    def _track_cells(self):
        """
        This function tracks all of the cells in every frame.
        """
        for frame in range(1, self.x.shape[0]):
            print(np.amax(self.y[frame]))
            print('Tracking frame ' + str(frame))
            cost_matrix = self._get_cost_matrix(frame)
            assignments = self._run_lap(cost_matrix)
            self._update_tracks(assignments, frame)
        return None