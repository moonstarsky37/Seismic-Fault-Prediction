import numpy as np
from utils.visualization import save_images
from data_loader import SeismicDataLoader
from model.factory import ModelFactory

class Predictor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.seismic_data = None
        self.predictions = None
        
    def _load_model(self):
        self.model = ModelFactory.create_unet(
            self.config.input_shape,
            self.config.model_weights_path
        )
        
    def _process_data(self, dat_index: int):
        raw_data = SeismicDataLoader.load(
            f"{self.config.val_seismic_path}{dat_index}.dat",
            self.config.input_shape[:-1]
        )
        self.seismic_data = SeismicDataLoader.normalize(
            SeismicDataLoader.preprocess_for_prediction(raw_data)
        )
    
    def execute(self, dat_index: int):
        self._load_model()
        self._process_data(dat_index)
        
        input_data = np.expand_dims(self.seismic_data, axis=[0, -1])
        self.predictions = self.model.predict(input_data)[0,...,0]
        
        save_images(
            self.seismic_data,
            self.predictions,
            self.config.output_dir,
            self.config.slices_to_visualize
        )
