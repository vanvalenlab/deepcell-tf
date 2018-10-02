import numpy as np
import copy
from skimage.measure import regionprops
from skimage.transform import resize
from scipy.optimize import linear_sum_assignment

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.preprocessing.image import apply_transform
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from deepcell.image_generators import MovieDataGenerator

class cell_tracker():
    def __init__(self,
                 movie,
                 annotation,
                 model,
                 crop_dim=32,
                 death=0.9,
                 birth=0.9,
                 division=0.2,
                 max_distance=200,
                 track_length=1,
                 occupancy_grid_size=10,
                 occupancy_window=100,
                 data_format=None):

        # TODO: Use a model that is served by tf-serving, not one on a local machine

        if data_format is None:
            data_format = K.image_data_format()

        self.x = copy.copy(movie)
        self.y = copy.copy(annotation)
        self.model = model
        self.crop_dim = crop_dim
        self.death = death
        self.birth = birth
        self.division = division
        self.max_distance = max_distance
        self.occupancy_grid_size=occupancy_grid_size
        self.occupancy_window=occupancy_window
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

    def _create_new_track(self, frame, cell_id=None):
        """
        This function creates new tracks
        """
        if len(list(self.tracks.keys())) > 0:
            new_track = np.amax(list(self.tracks.keys())) + 1
        else:
            new_track = 0
        self.tracks[new_track] = {}

        if cell_id is None:
            self.tracks[new_track]['label'] = new_track + 1
        else:
            self.tracks[new_track]['label'] = cell_id

        self.tracks[new_track]['frames'] = [frame]
        self.tracks[new_track]['daughters'] = []
        self.tracks[new_track]['capped'] = False
#        self.tracks[track]['death_frame'] = None
        self.tracks[new_track]['parent'] = None

        appearance, centroid, occupancy_grid = self._get_appearances(self.x, self.y, [frame], [cell_id])
        self.tracks[new_track]['appearances'] = appearance
        self.tracks[new_track]['centroids'] = centroid
        self.tracks[new_track]['occupancy_grids'] = occupancy_grid

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
            self._create_new_track(0, label)

        # Start a tracked label array
        self.y_tracked = self.y[[0]]

    def _get_cost_matrix(self, frame):
        """
        This function uses the model to create the cost matrix for
        assigning the cells in frame to existing tracks.
        """

        # Initialize matrices
        number_of_tracks = np.int(len(self.tracks.keys()))
        number_of_cells = np.int(np.amax(self.y[frame]))

        cost_matrix = np.zeros((number_of_tracks + number_of_cells, number_of_tracks + number_of_cells), dtype=K.floatx())
        assignment_matrix = np.zeros((number_of_tracks, number_of_cells), dtype=K.floatx())
        birth_matrix = np.zeros((number_of_cells, number_of_cells), dtype=K.floatx())
        death_matrix = np.zeros((number_of_tracks, number_of_tracks), dtype=K.floatx())
        mordor_matrix = np.zeros((number_of_cells, number_of_tracks), dtype=K.floatx()) # Bottom right matrix

        # Compute assignment matrix
        try:
            track_appearances = self._fetch_track_appearances()
        except:
            print("im breaking here ", frame)

        track_centroids = self._fetch_track_centroids()
        track_occupancy_grids = self._fetch_track_occupancy_grids()

        cell_appearances = np.zeros((number_of_cells, 1, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), dtype=K.floatx())
        cell_centroids = np.zeros((number_of_cells, 2), dtype=K.floatx())
        cell_occupancy_grids = np.zeros((number_of_cells,1,
                                         2*self.occupancy_grid_size+1,
                                         2*self.occupancy_grid_size+1, 1), dtype=K.floatx())

        for cell in range(number_of_cells):
            cell_appearance, cell_centroid, cell_occupancy_grid = self._get_appearances(self.x, self.y, [frame], [cell+1])
            cell_appearances[cell] = cell_appearance
            cell_centroids[cell] = cell_centroid
            cell_occupancy_grids[cell] = cell_occupancy_grid

        # Compute assignment matrix - Initialize and get model inputs
        input_1 = np.zeros((number_of_tracks, number_of_cells, self.track_length, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), dtype = K.floatx())
        input_2 = np.zeros((number_of_tracks, number_of_cells, 1, self.crop_dim, self.crop_dim, self.x.shape[self.channel_axis]), dtype = K.floatx())
        input_3 = np.zeros((number_of_tracks, number_of_cells, self.track_length, 2))
        input_4 = np.zeros((number_of_tracks, number_of_cells, 1, 2))
        input_5 = np.zeros((number_of_tracks, number_of_cells, self.track_length,
                            2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1,1))
        input_6 = np.zeros((number_of_tracks, number_of_cells, 1,
                            2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1,1))

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
                        distances[j,:] = dist/np.linalg.norm(dist)*self.max_distance

                input_3[track,cell,:,:] = distances[0:-1,:]
                input_4[track,cell,:,:] = distances[-1,:]

                input_5[track,cell,:,:,:,:] = track_occupancy_grids[track]
                input_6[track,cell,:,:,:,:] = cell_occupancy_grids[cell]

        # Compute assignment matrix - reshape model inputs for prediction
        input_1 = np.reshape(input_1, (number_of_tracks*number_of_cells, self.track_length, self.crop_dim,
                                       self.crop_dim, self.x.shape[self.channel_axis]))
        input_2 = np.reshape(input_2, (number_of_tracks*number_of_cells, 1, self.crop_dim,
                                      self.crop_dim, self.x.shape[self.channel_axis]))
        input_3 = np.reshape(input_3, (number_of_tracks*number_of_cells, self.track_length, 2))
        input_4 = np.reshape(input_4, (number_of_tracks*number_of_cells, 1, 2))
        input_5 = np.reshape(input_5, (number_of_tracks*number_of_cells, self.track_length,
                                      2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1, 1))
        input_6 = np.reshape(input_6, (number_of_tracks*number_of_cells, 1,
                                      2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1, 1))
        model_input = [input_1, input_2, input_3, input_4, input_5, input_6]

        predictions = self.model.predict(model_input) #TODO: implement some splitting function in case this is too much data
        predictions = np.reshape(predictions, (number_of_tracks, number_of_cells, 3))
        assignment_matrix = 1-predictions[:,:,1]

        # Make sure capped tracks are not allowed to have assignments
        for track in range(number_of_tracks):
            if self.tracks[track]['capped']:
                assignment_matrix[track,0:number_of_cells] = 1

#         print(np.round(assignment_matrix, decimals=2))

        # Compute birth matrix
        predictions_birth = predictions[:,:,2]
        birth_diagonal = np.array([self.birth] * number_of_cells) # 1-np.amax(predictions_birth, axis=0)
        birth_matrix = np.diag(birth_diagonal) + np.ones(birth_matrix.shape) - np.eye(number_of_cells)
#         print(np.round(1-birth_diagonal, decimals=2))
#         print(np.round(np.amax(predictions[:,:,2],axis=0), decimals=2))

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

    def _run_lap(self, cost_matrix):
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

        y_tracked_update = np.zeros((1, self.y.shape[1], self.y.shape[2], 1), dtype=K.floatx())

        for a in range(assignments.shape[0]):
            track, cell = assignments[a]
            track_id = track + 1 # Labels and indices differ by 1
            cell_id = cell + 1

            # Take care of everything if cells are tracked
            if track < number_of_tracks and cell < number_of_cells:
                self.tracks[track]['frames'].append(frame)
                appearance, centroid, occupancy_grid = self._get_appearances(self.x, self.y, [frame], [cell_id])
                self.tracks[track]['appearances'] = np.concatenate([self.tracks[track]['appearances'], appearance], axis = 0)
                self.tracks[track]['centroids'] = np.concatenate([self.tracks[track]['centroids'], centroid], axis=0)
                self.tracks[track]['occupancy_grids'] = np.concatenate([self.tracks[track]['occupancy_grids'], occupancy_grid], axis=0)
                y_tracked_update[self.y[[frame]] == cell_id] = track_id


            # Create a new track if there was a birth
            if track > number_of_tracks - 1 and cell < number_of_cells:
                self._create_new_track(frame, cell+1)
                new_track_id = np.amax(list(self.tracks.keys()))

                # See if the new track has a parent
                parent = self._get_parent(frame, cell_id)
                if parent is not None:
                    print('Division detected')
                    self.tracks[new_track_id]['parent'] = parent
                    self.tracks[parent]['daughters'].append(new_track_id)
                else:
                    self.tracks[new_track_id]['parent'] = None

                y_tracked_update[self.y[[frame]] == cell+1] = new_track_id+1

            # Dont touch anything if there was a cell that "died"
            if track < number_of_tracks and cell > number_of_cells - 1:
#               self.tracks[track]['death_frame'] = frame - 1
                continue

        # Cap the tracks of cells that divided
        number_of_tracks = len(self.tracks.keys())
        for track in range(number_of_tracks):
            if len(self.tracks[track]['daughters']) > 0:
                self.tracks[track]['capped'] = True

        # Check and make sure cells that divided did not get assigned to the same cell
        for track in range(number_of_tracks):
            if len(self.tracks[track]['daughters']) > 0:
                if frame in self.tracks[track]['frames']:

                    print("appearances removed from track ", track)
                    print("frames in the track ", self.tracks[track]['frames'])
                    print("length of daughter track ", len(self.tracks[track]['daughters']))
                    print("new track id ", new_track_id)
                    print("frame being removed ", frame)

                    # Create new track
                    cell_id = self.tracks[track]['label']
                    self._create_new_track(frame, cell_id)
                    new_track_id = np.amax(list(self.tracks.keys()))
                    self.tracks[new_track_id]['appearances'] = self.tracks[track]['appearances'][[-1],:,:,:]
                    self.tracks[new_track_id]['centroids'] = self.tracks[track]['centroids'][[-1],:]
                    self.tracks[new_track_id]['occupancy_grids'] = self.tracks[track]['occupancy_grids'][[-1],:,:,:]
                    self.tracks[new_track_id]['parent'] = track

                    # Remove frame from old track
                    self.tracks[track]['frames'].remove(frame)
                    self.tracks[track]['appearances'] = self.tracks[track]['appearances'][0:-1,:,:,:]
                    self.tracks[track]['centroids'] = self.tracks[track]['centroids'][0:-1,:]
                    self.tracks[track]['occupancy_grids'] = self.tracks[track]['occupancy_grids'][0:-1,:,:,:]
                    self.tracks[track]['daughters'].append(new_track_id)

                    # Change y_tracked_update
                    y_tracked_update[y_tracked_update == track+1] = new_track_id + 1

        # Update the tracked label array
        self.y_tracked = np.concatenate([self.y_tracked, y_tracked_update], axis=0)

    def _get_parent(self, frame, cell):
        """
        This function searches the tracks for the parent of a given cell
        It returns the parent cell's id or None if no parent exists.
        """
        try:
            track_appearances = self._fetch_track_appearances()
        except:
            print("i broke on parents ", frame)
        track_centroids = self._fetch_track_centroids()
        track_occupancy_grids = self._fetch_track_occupancy_grids()

        cell_appearance, cell_centroid, cell_occupancy_grid = self._get_appearances(self.x, self.y, [frame], [cell])
        cell_centroid = np.stack([cell_centroid]*track_centroids.shape[0], axis=0)
        cell_appearances = np.stack( [cell_appearance]*track_appearances.shape[0] , axis=0)
        cell_occupancy_grids = np.stack([cell_occupancy_grid]*track_occupancy_grids.shape[0], axis=0)

        all_centroids = np.concatenate([track_centroids, cell_centroid], axis=1)
        distances = np.diff(all_centroids, axis=1)
        zero_pad = np.zeros((track_centroids.shape[0],1,2), dtype = K.floatx())
        distances = np.concatenate([zero_pad,distances], axis=1)

        track_distances = distances[:,0:-1,:]
        cell_distances = distances[:,[-1],:]

        # Make sure the distances are all less than max distance
        for j in range(cell_distances.shape[0]):
            dist = cell_distances[j,:,:]
            if np.linalg.norm(dist) > self.max_distance:
                cell_distances[j,:,:] = dist/np.linalg.norm(dist)*self.max_distance

        occupancy_grids = np.concatenate([track_occupancy_grids, cell_occupancy_grids], axis=1)
        occupancy_generator = MovieDataGenerator(rotation_range=0, horizontal_flip=False, vertical_flip=False)

        for batch in range(occupancy_grids.shape[0]):
            og_batch = occupancy_grids[batch]
            og_batch = occupancy_generator.random_transform(og_batch)
            occupancy_grids[batch] = og_batch
        track_occupancy_grids = occupancy_grids[:,0:-1,:,:,:]
        cell_occupancy_grids = occupancy_grids[:,[-1],:,:,:]


        # Compute the probability the cell is a daughter of a track
        model_input = [track_appearances, cell_appearances, track_distances, cell_distances,
                       track_occupancy_grids, cell_occupancy_grids]
        predictions = self.model.predict(model_input)
        probs = predictions[:,2]

        print(np.round(probs, decimals=2))
        number_of_tracks = len(self.tracks.keys())

        # Make sure capped tracks can't be assigned parents
        for track in range(number_of_tracks):
            if self.tracks[track]['capped']:
                probs[track] = 0
                continue

        # Find out if the cell is a daughter of a track
        max_prob = np.amax(probs)
        parent_id = np.where(probs == max_prob)[0][0]
        print("New track")
        if max_prob > self.division:
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
                                        dtype=K.floatx())

        for track in self.tracks.keys():
            app = self.tracks[track]['appearances']
            if app.shape[0] > self.track_length - 1:
                track_appearances[track] = app[-self.track_length:,:,:,:]
            else:
                track_length = app.shape[0]
                missing_frames = self.track_length - track_length
                frames = np.array(list(range(-1,-track_length-1,-1)) + [-track_length]*missing_frames)
                track_appearances[track] = app[frames,:,:,:]
 #               except:
 #                   print("track ", track)
 #                   print("frames ", frames)
 #                   print("app.shape ", app.shape)
 #                   print("track length ", track_length)
 #                   print("are we sure about the track length... ", self.track_length)
 #                   print("missing_frames ", missing_frames)
 #                   print("self.tracks[track]['appearances']", self.tracks[track]['appearances'])

        return track_appearances

    def _fetch_track_centroids(self):
        """
        This function fetches the centroids for all of the existing tracks.
        If tracks are shorter than the track length they are filled in with
        the centroids from the first frame.
        """
        track_centroids = np.zeros((len(self.tracks.keys()), self.track_length, 2), dtype=K.floatx())

        for track in self.tracks.keys():
            cen = self.tracks[track]['centroids']
            if cen.shape[0] > self.track_length - 1:
                track_centroids[track] = cen[-self.track_length:,:]
            else:
                track_length = cen.shape[0]
                missing_frames = self.track_length - track_length
                frames = np.array(list(range(-1,-track_length-1,-1)) + [-track_length]*missing_frames)
                try:
                    track_centroids[track] = cen[frames,:]
                except:
                    print("track ", track)
                    print("frames ", frames)
                    print("cen.shape ", cen.shape)

        return track_centroids


    def _fetch_track_occupancy_grids(self):
        """
        This function gets the occupancy grids for all of the existing tracks.
        If tracks are shorter than the track length they are filled in with the
        centroids from the first frame.
        """
        track_occupancy_grids = np.zeros((len(self.tracks.keys()), self.track_length,
                                          2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1,1),
                                          dtype=K.floatx())
        for track in self.tracks.keys():
            og = self.tracks[track]['occupancy_grids']
            if og.shape[0] > self.track_length - 1:
                track_occupancy_grids[track] = og[-self.track_length:,:,:,:]
            else:
                track_length = og.shape[0]
                missing_frames = self.track_length - track_length
                frames = np.array(list(range(-1,-track_length-1,-1)) + [-track_length]*missing_frames)
                track_occupancy_grids[track] = og[frames,:,:,:]

        return track_occupancy_grids

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

        occupancy_grid_shape = (len(frames), 2*self.occupancy_grid_size+1,
                                2*self.occupancy_grid_size+1, 1)

        # Initialize storage for appearances and centroids
        appearances = np.zeros(appearance_shape, dtype=K.floatx())
        centroids = np.zeros(centroid_shape, dtype=K.floatx())
        occupancy_grids = np.zeros(occupancy_grid_shape, dtype=K.floatx())

        for counter, (frame, cell_label) in enumerate(zip(frames, labels)):
            # Get the bounding box
            X_frame = X[frame] if self.data_format == 'channels_last' else X[:, frame]
            y_frame = y[frame] if self.data_format == 'channels_last' else y[:, frame]
            props = regionprops(np.int32(y_frame == cell_label))
            minr, minc, maxr, maxc = props[0].bbox
            centroids[counter] = props[0].centroid

            # Extract images from bounding boxes
            if self.data_format == 'channels_first':
                appearance = np.copy(X[:, frame, minr:maxr, minc:maxc])
                resize_shape = (X.shape[channel_axis], self.crop_dim, self.crop_dim)
            else:
                appearance = np.copy(X[frame, minr:maxr, minc:maxc, :])
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

            # Get the occupancy grid
            occupancy_grid = np.zeros((2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1,1),
                                      dtype=K.floatx())
            X_padded = np.pad(X_frame, ((self.occupancy_window, self.occupancy_window),
                                        (self.occupancy_window, self.occupancy_window),
                                        (0,0)), mode='constant', constant_values=0)
            y_padded = np.pad(y_frame, ((self.occupancy_window, self.occupancy_window),
                                        (self.occupancy_window, self.occupancy_window),
                                        (0,0)), mode='constant', constant_values=0)
            props = regionprops(np.int32(y_padded == cell_label))
            center_x, center_y = props[0].centroid
            center_x, center_y = np.int(center_x), np.int(center_y)
            X_reduced = X_padded[center_x-self.occupancy_window:center_x+self.occupancy_window,
                                 center_y-self.occupancy_window:center_y+self.occupancy_window,:]
            y_reduced = y_padded[center_x-self.occupancy_window:center_x+self.occupancy_window,
                                 center_y-self.occupancy_window:center_y+self.occupancy_window,:]

            # Resize X_reduced in case it is used instead of the occupancy grid method
            resize_shape = (2*self.occupancy_grid_size+1, 2*self.occupancy_grid_size+1, X.shape[channel_axis])

            # Resize images from bounding box
            max_value = np.amax([np.amax(X_reduced), np.absolute(np.amin(X_reduced))])
            X_reduced /= max_value
            X_reduced = resize(X_reduced, resize_shape)
            X_reduced *= max_value

            # Fill up the occupancy grid
            center_x, center_y = self.occupancy_window, self.occupancy_window
            props = regionprops(np.int32(y_reduced))
            for prop in props:
                centroid_x, centroid_y = prop.centroid
                dist_x, dist_y = np.float(centroid_x - center_x), np.float(centroid_y - center_y)
                dist_x *= self.occupancy_grid_size/self.occupancy_window
                dist_y *= self.occupancy_grid_size/self.occupancy_window

                loc_x = np.int(np.floor(self.occupancy_grid_size + dist_x))
                loc_y = np.int(np.floor(self.occupancy_grid_size + dist_y))

                mark_grid = (loc_x >= 0) and (loc_x < occupancy_grid.shape[0]) and (loc_y >=0) and (loc_y < occupancy_grid.shape[1])

                if mark_grid:
                    occupancy_grid[loc_x, loc_y,0] = 1

            occupancy_grids[counter,:,:,:] = X_reduced #occupancy_grid


        return appearances, centroids, occupancy_grids

    def _track_cells(self):
        """
        This function tracks all of the cells in every frame.
        """
        for frame in range(1, self.x.shape[0]):
            print('Tracking frame ' + str(frame))
            cost_matrix = self._get_cost_matrix(frame)
            assignments = self._run_lap(cost_matrix)
            self._update_tracks(assignments, frame)

