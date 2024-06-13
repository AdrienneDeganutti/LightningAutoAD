import argparse
import json

def get_custom_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    args.train_yaml = config['train_yaml']
    args.val_yaml = config['val_yaml']
    args.test_yaml = config['test_yaml']
    args.batch_size = config['per_gpu_batch_size']
    args.max_epochs = config['num_train_epochs']
    args.learning_rate = config['learning_rate']
    args.num_workers = config['num_workers']
    args.max_seq_length = config['max_seq_length']
    args.scheduler = config['scheduler']
    args.adam_epsilon = config['adam_epsilon']

    return args