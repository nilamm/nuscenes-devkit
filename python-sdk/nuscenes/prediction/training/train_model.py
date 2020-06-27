import numpy as np
import torch

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
RUN_TIME = datetime.datetime.now()

## HYPERPARAMETERS ##

# model hyperparams
NUM_MODES = 2206
EXPERIMENT_DIR = '/home/jupyter/experiments/04'
KEY = 'covernet'  # mtp, covernet
BACKBONE = 'resnet50'  # resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x4d_ssl, resnext101_32x4d_swsl, simclr
PRINT_EVERY_BATCHES = 50

N_EPOCHS = 15 # how many (more) epochs to run
PREVIOUSLY_COMPLETED_EPOCHS = 0  # Starting epoch (default: 0)

# load weights from previous training,
# from directory: EXPERIMENT_DIR/weights
# (can be None)
LOAD_WEIGHTS_PATH = None
LOAD_OPTIMIZER_PATH = None

# data hyperparams
TRAIN_DOWNSAMPLE_FACTOR = 5
VAL_DOWNSAMPLE_FACTOR = 5
VERSION = 'v1.0-trainval'  # v1.0-mini, v1.0-trainval
DATA_ROOT = '/home/jupyter/data/sets/nuscenes'  # wherever the data is stored
TRAIN_SPLIT_NAME = 'train'  # 'mini_train', 'mini_val', 'train', 'train_val', 'val'
VAL_SPLIT_NAME = 'train_val'

# prepare output directories
if not os.path.exists(EXPERIMENT_DIR):
    os.mkdir(EXPERIMENT_DIR)

if not os.path.exists(os.path.join(EXPERIMENT_DIR, 'weights')):
    os.mkdir(os.path.join(EXPERIMENT_DIR, 'weights'))

# store config for later reference
config = {
    'notes': '',  # optional: any notes to describe this run
    'NUM_MODES': NUM_MODES,
    'EXPERIMENT_DIR': EXPERIMENT_DIR,
    'KEY': KEY,
    'BACKBONE': BACKBONE,
    'N_EPOCHS': N_EPOCHS,
    'PREVIOUSLY_COMPLETED_EPOCHS': PREVIOUSLY_COMPLETED_EPOCHS,
    'LOAD_WEIGHTS_PATH': LOAD_WEIGHTS_PATH,
    'LOAD_OPTIMIZER_PATH': LOAD_OPTIMIZER_PATH,
    'TRAIN_DOWNSAMPLE_FACTOR': TRAIN_DOWNSAMPLE_FACTOR,
    'VAL_DOWNSAMPLE_FACTOR': VAL_DOWNSAMPLE_FACTOR,
    'VERSION': VERSION,
    'DATA_ROOT': DATA_ROOT,
    'TRAIN_SPLIT_NAME': TRAIN_SPLIT_NAME,
    'VAL_SPLIT_NAME': VAL_SPLIT_NAME
}
config_fname = f'config_for_runtime_{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}.json'
with open(os.path.join(EXPERIMENT_DIR, config_fname), 'w') as json_file:
    json.dump(config, json_file)


## HELPER FUNCTIONS ##
def get_dataset(tokens, helper, key):
    if key == 'mtp':
        mtp_dataset = MTPDataset(tokens, helper)
        return train_mtpdataset
    elif key == 'covernet':
        covernet_dataset = CoverNetDataset(tokens, helper)
        return covernet_dataset
        
def get_dataloader(dataset, batch_size=16, num_workers=0, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def get_model(key, backbone='resnet18'):
    """Return a model"""
    backbone = Backbone(backbone)
    if key == 'mtp':
        model = MTP(backbone, NUM_MODES)
    elif key == 'covernet':
        model = CoverNet(backbone, NUM_MODES)
    
    model = model.to(device)
    return model
    

def get_loss_fn(key):
    """Return a loss function"""
    if key == 'mtp':
        return MTPLoss(NUM_MODES, 1, 5)
    elif key == 'covernet':
        
        NUM_SECONDS_INTO_FUTURE = 6
        
        # Epsilon is the amount of coverage in the set, 
        # i.e. a real world trajectory is at most 8 meters from a trajectory in this set
        # We released the set for epsilon = 2, 4, 8. Consult the paper for more information
        # on how this set was created

        if NUM_MODES == 64:
            PATH_TO_EPSILON_8_SET = DATA_ROOT + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
            trajectories = pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
        elif NUM_MODES == 415:
            PATH_TO_EPSILON_4_SET = DATA_ROOT + "nuscenes-prediction-challenge-trajectory-sets/epsilon_4.pkl"
            trajectories = pickle.load(open(PATH_TO_EPSILON_4_SET, 'rb'))
        elif NUM_MODES == 2206:
            PATH_TO_EPSILON_2_SET = DATA_ROOT + "nuscenes-prediction-challenge-trajectory-sets/epsilon_2.pkl"
            trajectories = pickle.load(open(PATH_TO_EPSILON_2_SET, 'rb'))
        else:
            raise Exception('Invalid number of modes')

        lattice = torch.zeros(NUM_MODES, NUM_SECONDS_INTO_FUTURE*2, 2)
        for i in range(NUM_MODES):
            lattice[i] = torch.Tensor(trajectories[i])

        return ConstantLatticeLoss(lattice)

def store_weights(model, optimizer, epoch, key):
    """Store model and optimizer weights in EXPERIMENT_DIR/weights"""
    weights_dir = os.path.join(EXPERIMENT_DIR, 'weights')

    new_weights_path = os.path.join(
        weights_dir,
        f'{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}_{key}_weights after_epoch {epoch}.pt')
    torch.save(model.state_dict(), new_weights_path)
    print("Stored weights at", new_weights_path)

    new_optimizer_path = os.path.join(
        weights_dir,
        f'{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}_{key}_optimizer after_epoch {epoch}.pt')

    torch.save(optimizer.state_dict(), new_optimizer_path)
    print("Stored optimizer at", new_optimizer_path)


def store_results(results, fname):
    """Store training or evaluation results in EXPERIMENT_DIR"""
    with open(os.path.join(EXPERIMENT_DIR, fname), 'w') as json_file:
        json.dump(results, json_file)
    print("Stored results at", fname)


def run_epoch(model, optimizer, dataloader, loss_function, epoch, phase):
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
        if (i % PRINT_EVERY_BATCHES == 0) and (i > 0):
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


def train_epochs(key,
                 backbone,
                 n_epochs,
                 train_dataloader,
                 val_dataloader,
                 previously_completed_epochs=0,
                 load_weights_path=None,
                 load_optimizer_path=None):
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
    model = get_model(key, backbone)
    loss_function = get_loss_fn(key)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # prepare for storing results
    all_train_results = {}
    all_validation_results = {}
    train_results_fname = f'train_results_{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}.json'
    val_results_fname = f'val_results_{RUN_TIME:%Y-%m-%d %Hh%Mm%Ss}.json'
    
    # optionally load model weights
    if load_weights_path:
        model.load_state_dict(
            torch.load(os.path.join(EXPERIMENT_DIR, 'weights', load_weights_path)))
        print("Loaded model weights from",
              os.path.join(EXPERIMENT_DIR, 'weights', load_weights_path))

    # optionally load optimizer weights
    if load_optimizer_path:
        optimizer.load_state_dict(
            torch.load(os.path.join(EXPERIMENT_DIR, 'weights', load_optimizer_path)))
        print("Loaded optimizer weights from",
              os.path.join(EXPERIMENT_DIR, 'weights', load_optimizer_path))

    for i in range(n_epochs):
        epoch = previously_completed_epochs + i

        print("="*15)
        print(f"Epoch: {epoch} (model: {key})")

        # train
        train_results = run_epoch(model, optimizer, train_dataloader,
                                  loss_function, epoch, phase='train')
        all_train_results[key+'_'+str(epoch)] = train_results

        # validate
        val_results = run_epoch(model, optimizer, val_dataloader,
                                loss_function, epoch, phase='validate')
        all_validation_results[key+'_'+str(epoch)] = val_results

        # store weights
        store_weights(model, optimizer, epoch, key)

        # store results
        store_results(all_train_results, train_results_fname)
        store_results(all_validation_results, val_results_fname)

        
# load data
nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT)
helper = PredictHelper(nusc)
train_tokens = get_prediction_challenge_split(TRAIN_SPLIT_NAME, dataroot=DATA_ROOT)
val_tokens = get_prediction_challenge_split(VAL_SPLIT_NAME, dataroot=DATA_ROOT)


# apply downsampling
train_tokens = np.random.choice(train_tokens,
                                int(len(train_tokens) / TRAIN_DOWNSAMPLE_FACTOR),
                                replace=False)
val_tokens = np.random.choice(val_tokens,
                              int(len(val_tokens) / VAL_DOWNSAMPLE_FACTOR),
                              replace=False)


# create data loaders
train_dataset = get_dataset(train_tokens, helper, KEY)
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True)

val_dataset = get_dataset(val_tokens, helper, KEY)
val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=0, shuffle=False)


## RUN TRAINING ##

train_epochs(key=KEY,
             backbone=BACKBONE,
             n_epochs=N_EPOCHS,
             previously_completed_epochs=PREVIOUSLY_COMPLETED_EPOCHS,
             train_dataloader=train_dataloader,
             val_dataloader=val_dataloader,
             load_weights_path=LOAD_WEIGHTS_PATH,
             load_optimizer_path=LOAD_OPTIMIZER_PATH
            )
