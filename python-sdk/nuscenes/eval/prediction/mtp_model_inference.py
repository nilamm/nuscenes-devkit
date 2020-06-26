""" Script for running MTP model on a given nuscenes-split. """

import json
import os
from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.eval.prediction.data_classes import Prediction
from nuscenes.prediction.load_data import MTPDataset
from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import datetime

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## HYPERPARAMETERS ##

# inference hyperparams
EXPERIMENT_DIR = '/home/jupyter/experiments/01'
WEIGHTS = None
NUM_MODES = 2
N_STEPS = 12  # 12 = 6 seconds * 2 frames/seconds

# data hyperparams
VERSION = 'v1.0-trainval'  # v1.0-mini, v1.0-trainval
DATA_ROOT = '/home/jupyter/data/sets/nuscenes'  # wherever the data is stored
SPLIT_NAME = 'val'
OUTPUT_DIR = '.'

## PREPARE DATA ##

# load data
nusc = NuScenes(version=VERSION, dataroot=DATA_ROOT)
helper = PredictHelper(nusc)
data_tokens = get_prediction_challenge_split(SPLIT_NAME, dataroot=DATA_ROOT)
dataset = MTPDataset(data_tokens, helper)
dataloader = DataLoader(dataset, batch_size=16, num_workers=1, shuffle=False)


## PREPARE MODEL ##

backbone = ResNetBackbone('resnet18')  # TODO: this could vary
model = MTP(backbone, NUM_MODES)
model = model.to(device)

if WEIGHTS is not None:
    model.load_state_dict(
        torch.load(os.path.join(EXPERIMENT_DIR, 'weights', WEIGHTS)))

model.eval()

all_predictions = []
for img, agent_state_vector, instance_tokens, sample_tokens in dataloader:
    img = img.to(device).float()
    agent_state_vector = agent_state_vector.to(device).float()

    with torch.no_grad():
        model_pred = model(img, agent_state_vector)

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


json.dump(all_predictions,
          open(os.path.join(OUTPUT_DIR, f'mtp_preds_{datetime.datetime.now():%Y-%m-%d %Hh%Mm%Ss}.json'), "w"))
