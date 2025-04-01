import argparse
from config import Config
from training.trainer import Trainer
from prediction.predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="Seismic Fault Segmentation Pipeline")
    parser.add_argument('--mode', required=True, choices=['train', 'predict'],
                      help="Pipeline mode: 'train' or 'predict'")
    parser.add_argument('--data-index', type=int, default=2,
                      help="Index of seismic data file for prediction")
    args = parser.parse_args()
    
    config = Config()
    
    if args.mode == 'train':
        trainer = Trainer(config)
        trainer.execute()
    elif args.mode == 'predict':
        predictor = Predictor(config)
        predictor.execute(dat_index=args.data_index)

if __name__ == "__main__":
    main()
