# Copyright 2016-2021 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-tf/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Assortment of CNN (and GNN) architectures for tracking single cells"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, TimeDistributed, Layer
from tensorflow.keras.layers import Conv2D, Conv3D, LSTM
from tensorflow.keras.layers import Input, Concatenate, InputLayer
from tensorflow.keras.layers import Add, Subtract, Dense, Reshape
from tensorflow.keras.layers import MaxPool2D, MaxPool3D
from tensorflow.keras.layers import Activation, Softmax
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.layers import ZeroPadding2D, ZeroPadding3D

from tensorflow.keras.regularizers import l2

from deepcell.layers import DilatedMaxPool2D, DilatedMaxPool3D
from deepcell.layers import ImageNormalization2D
from deepcell.layers import Location2D, Location3D
from deepcell.layers import ReflectionPadding2D, ReflectionPadding3D

from spektral.layers import GCSConv
# from spektral.layers import GCNConv, GATConv

from functools import partial


def siamese_model(input_shape=None,
                  features=None,
                  neighborhood_scale_size=10,
                  reg=1e-5,
                  init='he_normal',
                  filter_size=61):
    """Creates a tracking model based on Siamese Neural Networks(SNNs).

    Args:
        input_shape (tuple): If no input tensor, create one with this shape.
        features (list): Number of output features
        neighborhood_scale_size (int): number of input channels
        reg (int): regularization value
        init (str): Method for initalizing weights
        filter_size (int): the receptive field of the neural network

    Returns:
        tensorflow.keras.Model: 2D FeatureNet
    """
    def compute_input_shape(feature):
        if feature == 'appearance':
            return input_shape
        elif feature == 'distance':
            return (None, 2)
        elif feature == 'neighborhood':
            return (None, 2 * neighborhood_scale_size + 1,
                    2 * neighborhood_scale_size + 1,
                    input_shape[-1])
        elif feature == 'regionprop':
            return (None, 3)
        else:
            raise ValueError('siamese_model.compute_input_shape: '
                             'Unknown feature `{}`'.format(feature))

    def compute_reshape(feature):
        if feature == 'appearance':
            return (64,)
        elif feature == 'distance':
            return (2,)
        elif feature == 'neighborhood':
            return (64,)
        elif feature == 'regionprop':
            return (3,)
        else:
            raise ValueError('siamese_model.compute_output_shape: '
                             'Unknown feature `{}`'.format(feature))

    def compute_feature_extractor(feature, shape):
        if feature == 'appearance':
            # This should not stay: channels_first/last should be used to
            # dictate size (1 works for either right now)
            N_layers = np.int(np.floor(np.log2(input_shape[1])))
            feature_extractor = Sequential()
            feature_extractor.add(InputLayer(input_shape=shape))
            # feature_extractor.add(ImageNormalization2D('std', filter_size=32))
            for layer in range(N_layers):
                feature_extractor.add(Conv3D(64, (1, 3, 3),
                                             kernel_initializer=init,
                                             padding='same',
                                             kernel_regularizer=l2(reg)))
                feature_extractor.add(BatchNormalization(axis=channel_axis))
                feature_extractor.add(Activation('relu'))
                feature_extractor.add(MaxPool3D(pool_size=(1, 2, 2)))

            feature_extractor.add(Reshape((-1, 64)))
            return feature_extractor

        elif feature == 'distance':
            return None
        elif feature == 'neighborhood':
            N_layers_og = np.int(np.floor(np.log2(2 * neighborhood_scale_size + 1)))
            feature_extractor_neighborhood = Sequential()
            feature_extractor_neighborhood.add(
                InputLayer(input_shape=shape)
            )
            for layer in range(N_layers_og):
                feature_extractor_neighborhood.add(Conv3D(64, (1, 3, 3),
                                                          kernel_initializer=init,
                                                          padding='same',
                                                          kernel_regularizer=l2(reg)))
                feature_extractor_neighborhood.add(BatchNormalization(axis=channel_axis))
                feature_extractor_neighborhood.add(Activation('relu'))
                feature_extractor_neighborhood.add(MaxPool3D(pool_size=(1, 2, 2)))

            feature_extractor_neighborhood.add(Reshape((-1, 64)))

            return feature_extractor_neighborhood
        elif feature == 'regionprop':
            return None
        else:
            raise ValueError('siamese_model.compute_feature_extractor: '
                             'Unknown feature `{}`'.format(feature))

    if features is None:
        raise ValueError('siamese_model: No features specified.')

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
        raise ValueError('siamese_model: Only channels_last is supported.')
    else:
        channel_axis = -1

    input_shape = tuple([None] + list(input_shape))

    features = sorted(features)

    inputs = []
    outputs = []
    for feature in features:
        in_shape = compute_input_shape(feature)
        re_shape = compute_reshape(feature)
        feature_extractor = compute_feature_extractor(feature, in_shape)

        layer_1 = Input(shape=in_shape, name='{}_input1'.format(feature))
        layer_2 = Input(shape=in_shape, name='{}_input2'.format(feature))

        inputs.extend([layer_1, layer_2])

        # apply feature_extractor if it exists
        if feature_extractor is not None:
            layer_1 = feature_extractor(layer_1)
            layer_2 = feature_extractor(layer_2)

        # LSTM on 'left' side of network since that side takes in stacks of features
        layer_1 = LSTM(64)(layer_1)
        layer_2 = Reshape(re_shape)(layer_2)

        outputs.append([layer_1, layer_2])

    dense_merged = []
    for layer_1, layer_2 in outputs:
        merge = Concatenate(axis=channel_axis)([layer_1, layer_2])
        dense_merge = Dense(128)(merge)
        bn_merge = BatchNormalization(axis=channel_axis)(dense_merge)
        dense_relu = Activation('relu')(bn_merge)
        dense_merged.append(dense_relu)

    # Concatenate outputs from both instances
    merged_outputs = Concatenate(axis=channel_axis)(dense_merged)

    # Add dense layers
    dense1 = Dense(128)(merged_outputs)
    bn1 = BatchNormalization(axis=channel_axis)(dense1)
    relu1 = Activation('relu')(bn1)
    dense2 = Dense(128)(relu1)
    bn2 = BatchNormalization(axis=channel_axis)(dense2)
    relu2 = Activation('relu')(bn2)
    dense3 = Dense(3, activation='softmax', name='classification', dtype=K.floatx())(relu2)

    # Instantiate model
    final_layer = dense3
    model = Model(inputs=inputs, outputs=final_layer)

    return model


# Utils for Tracking Model
class TempMerge(Layer):
    def __init__(self, encoder_dim=64, **kwargs):
        super(TempMerge, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        new_shape = [-1, tf.shape(b)[2], self.encoder_dim]
        return tf.reshape(a, new_shape)

class TempUnmerge(Layer):
    def __init__(self, encoder_dim=64, **kwargs):
        super(TempUnmerge, self).__init__(**kwargs)
        self.encoder_dim = encoder_dim

    def call(self, inputs):
        a = inputs[0]
        b = inputs[1]
        new_shape = [-1, tf.shape(b)[1], tf.shape(b)[2], self.encoder_dim]
        return tf.reshape(a, new_shape)


class GNNTrackingModel(object):
    """Creates a tracking model based on Graph Neural Networks(GNNs).

    Args:
        n_filters (int): Number of filters
        endcoder_dim (int): Dimension of embedding
        n_layers (int): number of layers
        time_window (int): number of frames to include in temporal merges
        max_cells (int): maximum number of tracks per movie in dataset
        track_length (int): track length (parameter defined in dataset obj)
    """
    def __init__(self,
                 n_filters=64,
                 encoder_dim=64,
                 embedding_dim=64,
                 n_layers=3,
                 time_window=5,
                 max_cells=39,
                 track_length=8)

        self.n_filters = n_filters
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.time_window = time_window
        self.appearance_shape = (max_cells, track_length, 32, 32, 1)
        self.morphology_shape = (max_cells, track_length, 3)
        self.centroid_shape = (max_cells, track_length, 2)
        self.adj_shape = (max_cells, max_cells, track_length)

        # Create encoders and decoders
        self._create_reshape_model()
        self._create_unmerge_embeddings_model()
        self._create_unmerge_centroids_model()
        self._create_embedding_temporal_merge_model()
        self._create_delta_temporal_merge_model()
        self._create_appearance_encoder()
        self._create_morphology_encoder()
        self._create_centroid_encoder()
        self._create_delta_encoders()
        self._create_neighborhood_encoder()
        self._create_tracking_decoder()

        # Create branches
        self._create_training_branch()
        self._create_inference_branch()

        # Create model
        self._create_models()

    def _comparison(self, arg):
        x = arg[0]
        y = arg[1]

        x = tf.expand_dims(x, 2)
        multiples = [1, 1, tf.shape(y)[1], 1, 1]
        x = tf.tile(x, multiples)

        y = tf.expand_dims(y, 1)
        multiples = [1, tf.shape(x)[1], 1, 1, 1]
        y = tf.tile(y, multiples)

        return tf.concat([x, y], axis=-1)

    def _create_embedding_temporal_merge_model(self):
        inputs = Input(shape=(None, None, self.encoder_dim),
                       name='embedding_temporal_merge_input')

# LSTM Alt to CNN
#         x = inputs
#         x = TempMerge(name='merge_emb_tm')([x, inputs])
#         x = LSTM(self.encoder_dim, return_sequences=True, name='lstm_tm')(x)
#         x = TempUnmerge(name='unmerge_emb_tm')([x, inputs])

        x = inputs
        x = Conv2D(self.encoder_dim, (1, self.time_window), padding='SAME', name='conv2d_tm')(x)
        x = BatchNormalization(axis=-1, name='bn_tm')(x)
        x = Activation('relu', name='relu_tm')(x)

        self.embedding_temporal_merge_model = Model(inputs=inputs, outputs=x, name='embedding_temporal_merge')

    def _create_delta_temporal_merge_model(self):
        inputs = Input(shape=(None, None, self.encoder_dim),
                       name='centroid_temporal_merge_input')

# LSTM Alt to CNN
#         x = inputs
#         x = TempMerge(name='merge_delta_tm')([x, inputs])
#         x = LSTM(self.encoder_dim, return_sequences=True, name='lstm_delta')(x)
#         x = TempUnmerge(name='unmerge_delta_tm')([x, inputs])

        x = inputs
        x = Conv2D(self.encoder_dim, (1, self.time_window), padding='SAME', name='conv2d_delta')(x)
        x = BatchNormalization(axis=-1, name='bn_delta')(x)
        x = Activation('relu', name='relu_delta')(x)

        self.delta_temporal_merge_model = Model(inputs=inputs, outputs=x, name='delta_temporal_merge')

    def _unmerge_embeddings(self, x):
        new_shape = [-1, track_length, self.appearance_shape[0], self.embedding_dim]
        new_x = tf.reshape(x, new_shape)
        new_x = tf.transpose(new_x, perm=(0,2,1,3))

        return new_x

    def _unmerge_centroids(self, x):
        new_shape = [-1, track_length, self.centroid_shape[0], self.centroid_shape[2]]
        new_x = tf.reshape(x, new_shape)
        new_x = tf.transpose(new_x, perm=(0,2,1,3))

        return new_x

    def _create_reshape_model(self):
        # Define inputs
        app_input = Input(shape=self.appearance_shape,
                          name='appearances')
        morph_input = Input(shape=self.morphology_shape,
                            name='morphologies')
        centroid_input = Input(shape=self.centroid_shape,
                               name='centroids')
        adj_input = Input(shape=self.adj_shape,
                          name='adj_matrices')

        inputs = [app_input,
                  morph_input,
                  centroid_input,
                  adj_input]

        # Merge batch and temporal dimensions
        new_app_shape = [-1,
                         self.appearance_shape[0],
                         self.appearance_shape[2],
                         self.appearance_shape[3],
                         self.appearance_shape[4]]
        transposed_app_input = Lambda(lambda t: tf.transpose(t, perm=(0,2,1,3,4,5)))(app_input)
        reshaped_app_input = Lambda(lambda t: tf.reshape(t, new_app_shape),
                                    name='reshaped_appearances')(transposed_app_input)

        new_morph_shape = [-1,
                           self.morphology_shape[0],
                           self.morphology_shape[2]]
        transposed_morph_input = Lambda(lambda t: tf.transpose(t, perm=(0,2,1,3)))(morph_input)
        reshaped_morph_input = Lambda(lambda t: tf.reshape(t, new_morph_shape),
                                      name='reshaped_morphologies')(transposed_morph_input)

        new_centroid_shape = [-1,
                              self.centroid_shape[0],
                              self.centroid_shape[2]]
        transposed_centroid_input = Lambda(lambda t: tf.transpose(t, perm=(0,2,1,3)))(centroid_input)
        reshaped_centroid_input = Lambda(lambda t: tf.reshape(t, new_centroid_shape),
                                         name='reshaped_centroids')(transposed_centroid_input)

        new_adj_shape = [-1,
                         self.adj_shape[0],
                         self.adj_shape[1]]
        transposed_adj_input = Lambda(lambda t: tf.transpose(t, perm=(0,3,1,2)))(adj_input)
        reshaped_adj_input = Lambda(lambda t: tf.reshape(t, new_adj_shape),
                                    name='reshaped_adj_matrices')(transposed_adj_input)

        outputs = [reshaped_app_input,
                   reshaped_morph_input,
                   reshaped_centroid_input,
                   reshaped_adj_input]

        self.reshape_model = Model(inputs=inputs, outputs=outputs)

    def _create_appearance_encoder(self):
        app_shape = [None,
                     self.appearance_shape[2],
                     self.appearance_shape[3],
                     self.appearance_shape[4]]
        inputs = Input(shape=app_shape,
                       name='encoder_app_input')

        x = inputs

        x = TimeDistributed(ImageNormalization2D(norm_method='whole_image',
                                                 name='imgnrm_ae'))(x)

        for i in range(5):  ## is this supposed to match self.time_window?
            x = Conv3D(self.n_filters,
                       (1, 3, 3),
                       strides=1,
                       padding='same',
                       use_bias=False, name='conv3d_ae{}'.format(i))(x)
            x = BatchNormalization(axis=-1, name='bn_ae{}'.format(i))(x)
            x = Activation('relu', name='relu_ae{}'.format(i))(x)
            x = MaxPool3D(pool_size=(1,2,2))(x)
        x = Lambda(lambda t: tf.squeeze(t, axis=(2,3)))(x)
        x = Dense(self.encoder_dim, name='dense_aeout')(x)
        x = BatchNormalization(axis=-1, name='bn_aeout')(x)
        x = Activation('relu', name='appearance_embedding')(x)

        self.appearance_encoder = Model(inputs=inputs, outputs=x)

    def _create_morphology_encoder(self):
        morph_shape = [None,
                       self.morphology_shape[2]]
        inputs = Input(shape=morph_shape,
                       name='encoder_morph_input')

        x = inputs

        x = Dense(self.encoder_dim, name='dense_me')(x)
        x = BatchNormalization(axis=-1, name='bn_me')(x)
        x = Activation('relu', name='morphology_embedding')(x)

        self.morphology_encoder = Model(inputs=inputs, outputs=x)

    def _create_centroid_encoder(self):
        centroid_shape = [None,
                          self.centroid_shape[2]]
        inputs = Input(shape=centroid_shape,
                       name='encoder_centroid_input')

        x = inputs

        x = Dense(self.encoder_dim)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

        self.centroid_encoder = Model(inputs=inputs, outputs=x)

    def _create_delta_encoders(self):
        inputs = Input(shape=(None, None, self.centroid_shape[-1]),
                       name='encoder_delta_input')

        inputs_across_frames = Input(shape=(None, None, None, self.centroid_shape[-1]),
                                     name='encoder_delta_across_frames_input')

        d = Dense(self.n_filters)
        a = Activation('relu')

        x_0 = d(inputs)
        x_0 = BatchNormalization(axis=-1)(x_0)
        x_0 = a(x_0)

        x_1 = d(inputs_across_frames)
        x_1 = BatchNormalization(axis=-1)(x_1)
        x_1 = a(x_1)

        self.delta_encoder = Model(inputs=inputs, outputs=x_0)
        self.delta_across_frames_encoder = Model(inputs=inputs_across_frames, outputs=x_1)

    def _create_neighborhood_encoder(self):

        app_input = self.appearance_encoder.input
        morph_input = self.morphology_encoder.input
        centroid_input = self.centroid_encoder.input

        adj_shape = [None, None]
        adj_input = Input(shape=adj_shape,
                          name='encoder_adj_input')

        print('App input shape: ', app_input.shape)
        print('morph input shape: ', morph_input.shape)
        print('Centroid input shape: ', centroid_input.shape)
        print('adj input shape: ', adj_input.shape)

        app_features = self.appearance_encoder.output
        morph_features = self.morphology_encoder.output
        centroid_features = self.centroid_encoder.output
        adj = adj_input

        print('App feat shape: ', app_features.shape)
        print('morph feat shape: ', morph_features.shape)
        print('Centroid feat shape: ', centroid_features.shape)

        # Concatenate features
        node_features = Concatenate(axis=-1)([app_features, morph_features, centroid_features])
        node_features = Dense(self.n_filters, name='dense_ne0')(node_features)
        node_features = BatchNormalization(axis=-1, name='bn_ne0')(node_features)
        node_features = Activation('relu', name='relu_ne0')(node_features)

        print('concated features shape: ', node_features.shape)

        # Apply graph convolution
        for i in range(self.n_layers):
            node_features = GCSConv(self.n_filters,
                                    activation=None, name='gcs{}'.format(i))([node_features, adj])
            node_features = BatchNormalization(axis=-1, name='bn_ne{}'.format(i+1))(node_features)
            node_features = Activation('relu', name='relu_ne{}'.format(i+1))(node_features)

        concat = Concatenate(axis=-1)([app_features, morph_features, node_features])
        node_features = Dense(self.embedding_dim, name='dense_nef')(concat)
        node_features = BatchNormalization(axis=-1, name='bn_nef')(node_features)
        node_features = Activation('relu', name='relu_nef')(node_features)

        inputs = [app_input, morph_input, centroid_input, adj_input]
        outputs = [node_features, centroid_input]

        self.neighborhood_encoder = Model(inputs=inputs, outputs=outputs, name='neighborhood_encoder')

    def _create_unmerge_embeddings_model(self):
        inputs = Input(shape=(self.appearance_shape[0], self.embedding_dim),
                       name='unmerge_embeddings_input')
        x = inputs
        x = Lambda(self._unmerge_embeddings,
                   name='unmerge_embeddings')(x)
        self.unmerge_embeddings_model = Model(inputs=inputs, outputs=x, name='unmerge_embeddings_model')

    def _create_unmerge_centroids_model(self):
        inputs = Input(shape=(self.centroid_shape[0], self.centroid_shape[-1]),
                       name='unmerge_centroids_input')
        x = inputs
        x = Lambda(self._unmerge_centroids,
                   name='unmerge_centroids')(x)
        self.unmerge_centroids_model = Model(inputs=inputs, outputs=x, name='unmerge_centroids_model')

    def _get_deltas(self, x):
        # Convert raw positions to deltas
        deltas = Lambda(lambda t: t[:,:,1:,:]-t[:,:,0:-1,:])(x)

        deltas = Lambda(lambda t: tf.pad(t, tf.constant([[0,0],[0,0],[1,0],[0,0]])))(deltas)

        return deltas

    def _get_deltas_across_frames(self, centroids):
        # Find deltas across frames
        centroid_current = Lambda(lambda t: t[:,:,0:-1,:])(centroids)
        centroid_future = Lambda(lambda t: t[:,:,1:,:])(centroids)
        centroid_current = Lambda(lambda t: tf.expand_dims(t, 2))(centroid_current)
        centroid_future = Lambda(lambda t: tf.expand_dims(t, 1))(centroid_future)
        deltas_across_frames = Subtract()([centroid_future, centroid_current])

        return deltas_across_frames

    def _create_training_branch(self):
        inputs = self.reshape_model.inputs

        x, centroids = self.neighborhood_encoder(self.reshape_model.outputs)

        # Reshape embeddings to add back temporal dimension
        x = self.unmerge_embeddings_model(x)
        centroids = self.unmerge_centroids_model(centroids)

        # Get current and future embeddings
        x_current = Lambda(lambda t: t[:,:,0:-1,:])(x)
        x_future = Lambda(lambda t: t[:,:,1:,:])(x)

        # Integrate temporal information for embeddings and compare
        x_current = self.embedding_temporal_merge_model(x_current)
        x = Lambda(self._comparison, name='training_embedding_comparison')([x_current, x_future])

        # Convert centroids to deltas
        deltas_current = self._get_deltas(centroids)
        deltas_future = self._get_deltas_across_frames(centroids)

        deltas_current = Activation(tf.math.abs)(deltas_current)
        deltas_future = Activation(tf.math.abs)(deltas_future)

        deltas_current = self.delta_encoder(deltas_current)
        deltas_future = self.delta_across_frames_encoder(deltas_future)

        deltas_current = Lambda(lambda t: t[:,:,0:-1,:])(deltas_current)
        deltas_current = self.delta_temporal_merge_model(deltas_current)
        deltas_current = Lambda(lambda t: tf.expand_dims(t, 2))(deltas_current)
        multiples = [1, 1, self.centroid_shape[0], 1, 1]
        deltas_current = Lambda(lambda t: tf.tile(t, multiples))(deltas_current)

        deltas = Concatenate(axis=-1)([deltas_current, deltas_future])

        outputs = [x, deltas]

        # Create submodel
        self.training_branch = Model(inputs=inputs, outputs=outputs, name='training_branch')

    def _create_inference_branch(self):
        # batch size, tracks
        current_embedding = Input(shape=(None, None, self.encoder_dim),
                                  name='current_embeddings')
        current_centroids = Input(shape=(None, None, self.centroid_shape[-1]),
                                  name='current_centroids')

        future_embedding = Input(shape=(None, 1, self.encoder_dim),
                                 name='future_embeddings')
        future_centroids = Input(shape=(None, 1, self.centroid_shape[-1]),
                                 name='future_centroids')
        inputs = [current_embedding, current_centroids, future_embedding, future_centroids]

        # Embeddings - Integrate temporal information
        x_current = self.embedding_temporal_merge_model(current_embedding)

        # Embeddings - Get final frame from current track
        x_current = Lambda(lambda t: t[:,:,-1:,:])(x_current)

        x = Lambda(self._comparison, name='inference_comparison')([x_current, future_embedding])

        # Centroids - Get deltas
        deltas_current = self._get_deltas(current_centroids)
        deltas_current = Activation(tf.math.abs)(deltas_current)

        deltas_current = self.delta_encoder(deltas_current)
        deltas_current = self.delta_temporal_merge_model(deltas_current)
        deltas_current = Lambda(lambda t: t[:,:,-1:,:])(deltas_current)

        # Centroids - Get deltas across frames
        centroid_current_end = Lambda(lambda t: t[:,:,-1:,:])(current_centroids)

        centroid_current_end = Lambda(lambda t: tf.expand_dims(t, 2))(centroid_current_end)
        centroid_future = Lambda(lambda t: tf.expand_dims(t, 1))(future_centroids)
        deltas_future = Subtract()([centroid_future, centroid_current_end])
        deltas_future = Activation(tf.math.abs)(deltas_future)
        deltas_future = self.delta_across_frames_encoder(deltas_future)

        deltas_current = Lambda(lambda t: tf.expand_dims(t, 2))(deltas_current)
        multiples = [1, 1, self.centroid_shape[0], 1, 1]

        def delta_reshape(args):
            c = args[0]
            f = args[1]
            multiples = [1, 1, tf.shape(f)[1], 1, 1]
            output = tf.tile(c, multiples)
            return output

        deltas_current = Lambda(delta_reshape)([deltas_current, future_centroids])

        deltas = Concatenate(axis=-1)([deltas_current, deltas_future])

        outputs = [x, deltas]

        self.inference_branch = Model(inputs=inputs, outputs=outputs, name='inference_branch')

    def _create_tracking_decoder(self):
        embedding_input = Input(shape=(None, None, None, 2*self.embedding_dim))
        deltas_input = Input(shape=(None, None, None, 2*self.encoder_dim))

        embedding = Concatenate(axis=-1)([embedding_input, deltas_input])

        embedding = Dense(self.n_filters, name='dense_td0')(embedding)
        embedding = BatchNormalization(axis=-1, name='bn_td0')(embedding)
        embedding = Activation('relu', name='relu_td0')(embedding)

        for i in range(self.n_layers):
            res = Dense(self.n_filters, name='dense_td{}'.format(i+1))(embedding)
            res = BatchNormalization(axis=-1, name='bn_td{}'.format(i+1))(res)
            res = Activation('relu', name='relu_td{}'.format(i+1))(res)
            embedding = Add()([embedding, res])

        embedding = Dense(3, name='dense_outembed')(embedding)

        # Add classification head
        output = Softmax(axis=-1, name='softmax_comparison')(embedding)
        self.tracking_decoder = Model(inputs=[embedding_input, deltas_input],
                                      outputs=output,
                                      name='tracking_decoder')

    def _create_models(self):
        # Create inputs
        training_inputs = self.reshape_model.input
        inference_inputs = self.inference_branch.input

        # Apply decoder
        training_output = self.tracking_decoder(self.training_branch.output)
        inference_output = self.tracking_decoder(self.inference_branch.output)

        # Name the training output layer
        training_output = Lambda(lambda t: t, name='temporal_adj_matrices')(training_output)

        self.training_model = Model(inputs=training_inputs, outputs=training_output)
        self.inference_model = Model(inputs=inference_inputs, outputs=inference_output)
