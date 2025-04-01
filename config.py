from dataclasses import dataclass, field

@dataclass
class Config:
    # Data configuration
    train_seismic_path: str = "./data/train/seis/"
    train_fault_path: str = "./data/train/fault/"
    val_seismic_path: str = "./data/validation/seis/"
    val_fault_path: str = "./data/validation/fault/"
    model_weights_path: str = "models/ziyu-fseg-70.hdf5"
    output_dir: str = "./predict_output/"
    
    # Model parameters
    input_shape: tuple = (128, 128, 128, 1)
    batch_size: int = 1
    learning_rate: float = 1e-4
    epochs: int = 100
    slices_to_visualize: list = field(default_factory=lambda: [50, 64, 90])
    
    # Training config
    checkpoint_dir: str = "check1/"
    log_dir: str = "./logs/"
    train_samples: int = 200
    val_samples: int = 20
