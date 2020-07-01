""" Script for running MTP model on a given nuscenes-split. """

import json
import os
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.load_data import MTPDataset, CoverNetDataset
from nuscenes.prediction.models.backbone import Backbone
from nuscenes.prediction.models.mtp import MTP
from nuscenes.prediction.models.covernet import CoverNet
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import datetime
import pickle
import copy

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

## HYPERPARAMETERS ##

# inference hyperparams
EXPERIMENT_DIR = '/home/jupyter/experiments/09-simclr-freeze'
WEIGHTS = '2020-06-29 03h59m26s_covernet_weights after_epoch 5.pt'
NUM_MODES = 2206
TOP_K = 25
N_STEPS = 12  # 12 = 6 seconds * 2 frames/seconds
BACKBONE = 'simclr'
FREEZE = True

# data hyperparams
VERSION = 'v1.0-trainval'  # v1.0-mini, v1.0-trainval
DATA_ROOT = '/home/jupyter/data/sets/nuscenes'  # wherever the data is stored
SPLIT_NAME = 'val'

KEY = 'covernet'

## PREPARE DATA ##

# load data
nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT)
helper = PredictHelper(nusc)
data_tokens = get_prediction_challenge_split(SPLIT_NAME, dataroot=DATA_ROOT)
dataset = CoverNetDataset(data_tokens, helper)
dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)
print(f"Loaded split {SPLIT_NAME}, length {len(dataset)}, in {len(dataloader)} batches.")


## PREPARE MODEL ##

#TO DO (sj3003): Can we move this to a common place, seems to be used in a couple of other places as well
def get_model(key, backbone, num_modes):
    backbone = Backbone(backbone, FREEZE)
    if key == 'mtp':
        model = MTP(backbone, num_modes)
    elif key == 'covernet':
        model = CoverNet(backbone, num_modes)

    model = model.to(device)    
    return model

def get_fixed_trajectory_set(num_modes):
    if num_modes == 64:
        PATH_TO_EPSILON_8_SET = DATA_ROOT + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
        trajectories = pickle.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
    elif num_modes == 415:
        PATH_TO_EPSILON_4_SET = DATA_ROOT + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_4.pkl"
        trajectories = pickle.load(open(PATH_TO_EPSILON_4_SET, 'rb'))
    elif num_modes == 2206:
        PATH_TO_EPSILON_2_SET = DATA_ROOT + "/nuscenes-prediction-challenge-trajectory-sets/epsilon_2.pkl"
        trajectories = pickle.load(open(PATH_TO_EPSILON_2_SET, 'rb'))
    else:
        raise Exception('Invalid number of modes')
        
    return trajectories
      
        
model = get_model(KEY, BACKBONE, NUM_MODES)

if WEIGHTS is not None:
    model.load_state_dict(
        torch.load(os.path.join(EXPERIMENT_DIR, 'weights', WEIGHTS)))

model.eval()

def get_predictions(key):

    all_predictions = []
    
    if key == 'covernet':
        trajectories = np.array(get_fixed_trajectory_set(NUM_MODES))

    for j, (img, agent_state_vector, _, instance_tokens, sample_tokens) in enumerate(dataloader):
        if (j % 20 == 0) and (j > 0):
            print(f"Completed batch {j}", datetime.datetime.now())
        img = img.to(device).float()
        agent_state_vector = agent_state_vector.to(device).float()

        with torch.no_grad():
            model_pred = model(img, agent_state_vector)
            model_pred = model_pred.cpu().detach().numpy()
        
            #model outputs are x, y positions
            if key == 'mtp':
                for i in range(model_pred.size(0)):
                    pred = model_pred[i].cpu().detach().numpy()
                    instance_token = instance_tokens[i]
                    sample_token = sample_tokens[i]

                    # collect the predicted trajectories and correspondings probabilities
                    trajectories = []
                    probs = []
                    for j in range(NUM_MODES):
                        trajectories.append(pred[j*(N_STEPS*2+1):(j+1)*(N_STEPS*2+1)-1].reshape(-1, 2))
                        probs.append(pred[(j+1)*(N_STEPS*2+1)-1].item())

                    all_predictions.append(Prediction(
                        instance_token,
                        sample_token,
                        np.array(trajectories),
                        F.softmax(torch.Tensor(probs), dim=0).numpy()).serialize())

            #model outputs are logits corresponding to a pre-built fixed trajectory set
            elif key == 'covernet':
#                 trajectories = np.array(get_fixed_trajectory_set(NUM_MODES))
                for i in range(model_pred.shape[0]):
                    trajectories_cpy = copy.deepcopy(trajectories)
#                     logits = model_pred[i].cpu().detach().numpy()
                    logits = model_pred[i]
                    instance_token = instance_tokens[i]
                    sample_token = sample_tokens[i]

                    trajectories_cpy = trajectories[np.argsort(logits)[::-1]]
                    trajectories_cpy = trajectories_cpy[:TOP_K]
                    logits = logits[np.argsort(logits)[::-1]]
                    logits = logits[:TOP_K]

                    all_predictions.append(Prediction(
                        instance_token,
                        sample_token,
                        trajectories_cpy,
                        F.softmax(torch.Tensor(logits), dim=0).numpy()).serialize())
            
    return all_predictions

predictions = get_predictions(KEY)
json.dump(predictions,
          open(os.path.join(EXPERIMENT_DIR, f'{KEY}_preds_{datetime.datetime.now():%Y-%m-%d %Hh%Mm%Ss}.json'), "w"))