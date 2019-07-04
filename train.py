import argparse
import json

from utils import set_seed, get_model_vocab, get_trainer, train_val_split_dataset, get_train_dataset


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='input/train.csv', help="Path of data")
    parser.add_argument("--config_path", type=str, default='config.json', help="Path of config")

    return parser


def main(args):
    with open(args.config_path, 'r') as json_file:
        config = json.load(json_file)

    seed = config['seed']
    validation_size = config['validation_size']
    n_epochs = config['n_epochs']
    linear_scheduler = config['linear_scheduler']
    train_batch_size = config['train_batch_size']
    train_batch_split = config['train_batch_split']
    test_batch_size = config['test_batch_size']
    save_last = config['save_last']
    save_best = config['save_best']
    neg_sample = config['neg_sample']
    train_data_path = config['train_data_path']
    validation_data_path = config['validation_data_path']


    set_seed(seed)
    train_val_split_dataset(args.data_path, train_data_path, validation_data_path, validation_size, seed)
    model, vocab = get_model_vocab(config)
    train_dataset = get_train_dataset(train_data_path, vocab, model.n_pos_embeddings, neg_sample)
    val_dataset = get_train_dataset(validation_data_path, vocab, model.n_pos_embeddings)

    if linear_scheduler:
        config['lr_decay'] = 1 / (n_epochs * (len(train_dataset) + train_batch_size - 1) // train_batch_size)
    trainer = get_trainer(config, model)

    trainer.train(train_data=train_dataset,
                  n_epochs=n_epochs,
                  train_batch_size=train_batch_size,
                  train_batch_split=train_batch_split,
                  test_data=val_dataset,
                  test_batch_size=test_batch_size,
                  save_last=save_last,
                  save_best=save_best)


if __name__ == "__main__":
    arg_parser = get_parser()
    args = arg_parser.parse_known_args()[0]
    main(args)
