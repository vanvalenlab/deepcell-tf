from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras import backend as K
import numpy as np
from deepcell import train_model_siamese, rate_scheduler
from deepcell import siamese_model as the_model

def main():
    direc_data = '/data/npz_data/cells/unspecified_nuclear_data/nuclear_movie/'
    dataset = 'nuclear_movie_same'

    training_data = np.load('{}{}.npz'.format(direc_data, dataset))

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    lr_sched = rate_scheduler(lr=0.01, decay=0.99)
    in_shape = (14, 14, 1)
    model = the_model(input_shape=in_shape)#, n_features=1, reg=1e-5)

    train_model_siamese(model=model,
                        dataset='nuclear_movie_same',
                        optimizer=optimizer,
                        expt='',
                        it=0,
                        batch_size=1,
                        n_epoch=100,
                        direc_save='/data/models/cells/unspecified_nuclear_data/nuclear_movie',
                        direc_data='/data/npz_data/cells/unspecified_nuclear_data/nuclear_movie/',
                        lr_sched=lr_sched,
                        rotation_range=0,
                        flip=True,
                        shear=0,
                        class_weight=None)


if __name__=='__main__':
    main()
