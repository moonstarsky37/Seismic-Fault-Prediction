import numpy as np

class SeismicDataLoader:
    @staticmethod
    def load(file_path: str, shape: tuple) -> np.ndarray:
        data = np.fromfile(file_path, dtype=np.single)
        return np.reshape(data, shape)
    
    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        return (data - np.mean(data)) / np.std(data)
    
    @staticmethod
    def preprocess_for_prediction(data: np.ndarray) -> np.ndarray:
        return np.transpose(data)
