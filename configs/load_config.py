import argparse
import json

def get_custom_args():

    config_file = "configs/8frames_default.json"

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Create a simple namespace object to store config values
    class Args:
        pass

    args = Args()

    args.train_yaml = config['train_yaml']
    args.val_yaml = config['val_yaml']
    args.test_yaml = config['test_yaml']
    args.num_devices = config['num_devices']
    args.batch_size = config['per_gpu_batch_size']
    args.max_epochs = config['num_train_epochs']
    args.learning_rate = config['learning_rate']
    args.num_workers = config['num_workers']
    args.max_seq_length = config['max_seq_length']
    args.scheduler = config['scheduler']
    args.adam_epsilon = config['adam_epsilon']
    args.max_num_frames = config['max_num_frames']
    args.seed = config['seed']

    return args