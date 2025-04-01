from tensorflow.keras.callbacks import ModelCheckpoint
from utils.data_generator import DataGenerator
from model.factory import ModelFactory
from training.callbacks import TrainValTensorBoard

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_gen = None
        self.val_gen = None
        
    def _init_generators(self):
        self.train_gen = DataGenerator(
            self.config.train_seismic_path,
            self.config.train_fault_path,
            range(self.config.train_samples),
            self.config.batch_size
        )
        self.val_gen = DataGenerator(
            self.config.val_seismic_path,
            self.config.val_fault_path,
            range(self.config.val_samples),
            self.config.batch_size,
            shuffle=False
        )
        
    def _init_callbacks(self):
        self.callbacks = [
            ModelCheckpoint(
                filepath=f"{self.config.checkpoint_dir}fseg-{{epoch:02d}}.hdf5",
                monitor="val_acc",
                save_best_only=False,
                mode="max"
            ),
            TrainValTensorBoard(log_dir=self.config.log_dir)
        ]
    
    def execute(self):
        self.model = ModelFactory.create_unet(self.config.input_shape)
        self._init_generators()
        self._init_callbacks()
        
        self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.config.epochs,
            callbacks=self.callbacks
        )
        self.model.save(self.config.model_weights_path)
