import numpy as np
import torch
import argparse
from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split

from torch.utils.data import DataLoader
from nuscenes.prediction.models.backbone import Backbone
from nuscenes.prediction.models.mtp import MTP, MTPLoss
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
import torch.optim as optim
from nuscenes.prediction.load_data import MTPDataset
from nuscenes.prediction.load_data import CoverNetDataset
import json
import pickle

import datetime
import os

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
RUN_TIME = datetime.datetime.now()


def get_dataset(tokens, helper, args):
    if args.key == 'mtp':
        mtp_dataset = MTPDataset(tokens, helper)
        return train_mtpdataset
    elif args.key == 'covernet':
        covernet_dataset = CoverNetDataset(tokens, helper)
        return covernet_dataset


def get_model(args):
    """Return a model"""
    backbone = Backbone(args.backbone, args.freeze_bottom)
    if args.key == 'mtp':
        model = MTP(backbone, args.num_modes)
    elif args.key == 'covernet':
        model = CoverNet(backbone, args.num_modes)

    model = model.to(device)
    return model


def get_loss_fn(args):
    """Return a loss function"""
    if args.key == 'mtp':
        return MTPLoss(args.num_modes, 1, 5)
    elif args.key == 'covernet':

        NUM_SECONDS_INTO_FUTURE = 6

        # Epsilon is the amount of coverage in the set,
        # i.e. a real world trajectory is at most 8 meters from a trajectory in this set
        # We released the set for epsilon = 2, 4, 8. Consult the paper for more information
        # on how this set was created

        if args.num_modes == 64:
            PATH_TO_EPSILON_8_SET = args.data_root + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
            trajectories = pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
        elif args.num_modes == 415:
            PATH_TO_EPSILON_4_SET = args.data_root + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_4.pkl"
            trajectories = pickle.load(open(PATH_TO_EPSILON_4_SET, 'rb'))
        elif args.num_modes == 2206:
            PATH_TO_EPSILON_2_SET = args.data_root + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_2.pkl"
            trajectories = pickle.load(open(PATH_TO_EPSILON_2_SET, 'rb'))
        else:
            raise Exception('Invalid number of modes')

        lattice = torch.zeros(args.num_modes, NUM_SECONDS_INTO_FUTURE*2, 2)
        for i in range(args.num_modes):
            lattice[i] = torch.Tensor(trajectories[i])

        return ConstantLatticeLoss(lattice)


def store_weights(model, optimizer, epoch, args):
    """Store model and optimizer weights in EXPERIMENT_DIR/weights"""
    weights_dir = os.path.join(args.experiment_dir, 'weights')

    new_weights_path = os.path.join(
        weights_dir,
        f'{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}_{args.key}_weights after_epoch {epoch}.pt')
    torch.save(model.state_dict(), new_weights_path)
    print("Stored weights at", new_weights_path)

    new_optimizer_path = os.path.join(
        weights_dir,
        f'{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}_{args.key}_optimizer after_epoch {epoch}.pt')

    torch.save(optimizer.state_dict(), new_optimizer_path)
    print("Stored optimizer at", new_optimizer_path)


def store_results(results, fname, args):
    """Store training or evaluation results in EXPERIMENT_DIR"""
    with open(os.path.join(args.experiment_dir, fname), 'w') as json_file:
        json.dump(results, json_file)
    print("Stored results at", fname)


def run_epoch(model, optimizer, dataloader, loss_function, epoch, phase, args):
    """
    Run one epoch of either training or validation

    phase: 'train' or 'validate'
    """
    assert phase in ['train', 'validate']

    if phase == 'train':
        model.train()
    else:
        model.eval()

    print(f"{phase.capitalize()}, epoch {epoch}:", datetime.datetime.now())
    print(f"{len(dataloader)} batches")

    start = datetime.datetime.now()

    running_loss = 0.0

    for i, (img, agent_state_vector, ground_truth, _, _) in enumerate(dataloader):
        if (i % args.print_every_batches == 0) and (i > 0):
            print(f"Running loss after {i} batches: {(running_loss / i):4f}",
                  datetime.datetime.now())

        optimizer.zero_grad()

        img = img.to(device).float()
        agent_state_vector = agent_state_vector.to(device).float()
        ground_truth = ground_truth.to(device).float()

        with torch.set_grad_enabled(phase == 'train'):
            prediction = model(img, agent_state_vector)
            loss = loss_function(prediction, ground_truth)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)

    elapsed_mins = (datetime.datetime.now() - start) / datetime.timedelta(minutes=1)

    print(f"{phase.capitalize()} epoch {epoch} completed at", datetime.datetime.now())
    print(f"Elapsed minutes: {elapsed_mins}")
    print(f"{phase.capitalize()} loss after epoch {epoch}: {epoch_loss:.4f}")

    return {
        'epoch': epoch,
        'loss': epoch_loss,
        'elapsed_mins': elapsed_mins
    }


def train_epochs(train_dataloader,
                 val_dataloader,
                 args):
    """
    Run several epochs of training and evaluation for a model, optionally
    loading weights.

    key: model key
    n_epochs: how many more epochs to train (if starting from scratch, this is
        total epochs)
    previously_completed_epochs: if continuing to train a model. Default: 0
    """
    print()

    # load model
    model = get_model(args)
    loss_function = get_loss_fn(args)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # prepare for storing results
    all_train_results = {}
    all_validation_results = {}
    train_results_fname = f'train_results_{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}.json'
    val_results_fname = f'val_results_{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}.json'

    # optionally load model weights
    if args.load_weights_path:
        model.load_state_dict(
            torch.load(os.path.join(args.experiment_dir, 'weights', args.load_weights_path)))
        print("Loaded model weights from",
              os.path.join(args.experiment_dir, 'weights', args.load_weights_path))

    # optionally load optimizer weights
    if args.load_optimizer_path:
        optimizer.load_state_dict(
            torch.load(os.path.join(args.experiment_dir, 'weights', args.load_optimizer_path)))
        print("Loaded optimizer weights from",
              os.path.join(args.experiment_dir, 'weights', args.load_optimizer_path))

    for i in range(args.n_epochs):
        epoch = args.previously_completed_epochs + i

        print("="*15)
        print(f"Epoch: {epoch} (model: {args.key})")

        # train
        train_results = run_epoch(model, optimizer, train_dataloader,
                                  loss_function, epoch, phase='train',
                                  args=args)
        all_train_results[args.key+'_'+str(epoch)] = train_results

        # validate
        val_results = run_epoch(model, optimizer, val_dataloader,
                                loss_function, epoch, phase='validate',
                                args=args)
        all_validation_results[args.key+'_'+str(epoch)] = val_results

        # store weights
        store_weights(model, optimizer, epoch, args)

        # store results
        store_results(all_train_results, train_results_fname, args)
        store_results(all_validation_results, val_results_fname, args)


def main(args):
    print("Args:")
    print(vars(args))
    
    print("Device:")
    print(device)

    # prepare output directories
    if not os.path.exists(args.experiment_dir):
        os.mkdir(args.experiment_dir)

    if not os.path.exists(os.path.join(args.experiment_dir, 'weights')):
        os.mkdir(os.path.join(args.experiment_dir, 'weights'))

    # store the arguments for reference
    config_fname = f'config_for_runtime_{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}.json'
    with open(
            os.path.join(args.experiment_dir, config_fname), 'w') as json_file:
        json.dump(vars(args), json_file)

    # load data
    nusc = NuScenes(version=args.version, dataroot=args.data_root)
    helper = PredictHelper(nusc)
    train_tokens = get_prediction_challenge_split(args.train_split_name,
                                                  dataroot=args.data_root)
    val_tokens = get_prediction_challenge_split(args.val_split_name,
                                                dataroot=args.data_root)

    # apply downsampling
    train_tokens = np.random.choice(
        train_tokens,
        int(len(train_tokens) / args.train_downsample_factor),
        replace=False)
    val_tokens = np.random.choice(
        val_tokens,
        int(len(val_tokens) / args.val_downsample_factor),
        replace=False)

    # create data loaders
    train_dataset = get_dataset(train_tokens, helper, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True)

    val_dataset = get_dataset(val_tokens, helper, args)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                num_workers=args.num_workers, shuffle=False)

    # run training
    train_epochs(train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader,
                 args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model.')

    # model arguments
    parser.add_argument('--num_modes',
                        help='Number of trajectory options to be predicted.',
                        type=int,
                        default=1)
    parser.add_argument('--key', help='Key to identify model architecture. E.g. mtp, covernet')
    parser.add_argument('--backbone',
                        help='Which backbone vision model to use. resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x4d_ssl, resnext101_32x4d_swsl, simclr',
                        default='resnet50')
    parser.add_argument('--freeze_bottom', dest='freeze_bottom',
                        help='Freeze the bottom layers of the backbone network, allowing only the top layers to be fine-tuned',
                        action='store_true')
    parser.add_argument('--no_freeze_bottom', dest='freeze_bottom',
                        help='Allow all params of the backbone to be fine-tuned',
                        action='store_false')
    parser.set_defaults(freeze_bottom=True)

    # training arguments
    parser.add_argument('--n_epochs',
                        help='How many (more) epochs to run training.',
                        type=int,
                        default=15)
    parser.add_argument('--print_every_batches',
                        help='How often to print an update during training',
                        type=int,
                        default=50)
    parser.add_argument('--batch_size',
                        help='Batch size for training and validation data loaders.',
                        type=int,
                        default=16)
    parser.add_argument('--num_workers',
                        help='Num workers for data loader',
                        type=int,
                        default=0)
    parser.add_argument('--previously_completed_epochs',
                        help='Starting epoch. If training from scratch, this is 0.',
                        type=int,
                        default=0)
    parser.add_argument('--load_weights_path', help='Name of model weights file in the directory EXPERIMENT_DIR/weights', default=None)
    parser.add_argument('--load_optimizer_path', help='Name of optimizer state file in the directory EXPERIMENT_DIR/weights', default=None)

    # data arguments
    parser.add_argument('--train_downsample_factor',
                        help='Divide training data size by this factor.',
                        type=float,
                        default=5)
    parser.add_argument('--val_downsample_factor',
                        help='Divide validation data size by this factor.',
                        type=float,
                        default=5)
    parser.add_argument('--version', help='nuScenes version number.', default='v1.0-trainval')
    parser.add_argument('--data_root', help='Directory storing NuScenes data.', default='/home/jupyter/data/sets/nuscenes')
    parser.add_argument('--train_split_name', help='Data split to run train on.', default='train')
    parser.add_argument('--val_split_name', help='Data split to run validation on.', default='train_val')
    parser.add_argument('--experiment_dir', help='Where to store the results. E.g. /home/jupyter/experiments/04')

    args = parser.parse_args()
    main(args)
