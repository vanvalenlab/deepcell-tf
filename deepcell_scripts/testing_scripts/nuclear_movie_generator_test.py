from deepcell import train_model_siamese, rate_scheduler

def main():
    train_model_siamese(model=None,
                        dataset='nuclear_movie_same',
                        optimizer=None,
                        expt='',
                        it=0,
                        batch_size=1,
                        n_epoch=100,
                        direc_save='/data/training_data/trained_networks/',
                        direc_data='/data/training_data/training_data_npz/nuclear_movie/',
                        lr_sched=rate_scheduler(lr=0.01, decay=0.95),
                        rotation_range=0,
                        flip=True,
                        shear=0,
                        class_weight=None)


if __name__=='__main__':
    main()
