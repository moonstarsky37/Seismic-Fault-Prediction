from tensorflow.keras.optimizers import Adam
from model.unet import unet, cross_entropy_balanced

class ModelFactory:
    @staticmethod
    def create_unet(input_shape, weights_path=None):
        model = unet(input_size=input_shape)
        model.compile(optimizer=Adam(learning_rate=1e-4), 
                    loss=cross_entropy_balanced)
        if weights_path:
            model.load_weights(weights_path)
        return model
