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
import argparse

np.random.seed(0)
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


#TO DO (sj3003): Can we move this to a common place, seems to be used in a couple of other places as well
def get_model(args):
    backbone = Backbone(args.backbone, args.freeze_bottom)
    if key == 'mtp':
        model = MTP(backbone, args.num_modes)
    elif key == 'covernet':
        model = CoverNet(backbone, args.num_modes)

    model = model.to(device)
    return model


def get_fixed_trajectory_set(args):
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

    return trajectories


def get_predictions(args, dataloader):

    all_predictions = []

    if args.key == 'covernet':
        trajectories = np.array(get_fixed_trajectory_set(args))

    for j, (img, agent_state_vector, _, instance_tokens, sample_tokens) in enumerate(dataloader):
        if (j % 20 == 0) and (j > 0):
            print(f"Completed batch {j}", datetime.datetime.now())
        img = img.to(device).float()
        agent_state_vector = agent_state_vector.to(device).float()

        with torch.no_grad():
            model_pred = model(img, agent_state_vector)
            model_pred = model_pred.cpu().detach().numpy()

            # model outputs are x, y positions
            if args.key == 'mtp':
                for i in range(model_pred.shape[0]):
                    pred = model_pred[i]
                    instance_token = instance_tokens[i]
                    sample_token = sample_tokens[i]

                    # collect the predicted trajectories and correspondings probabilities
                    trajectories = []
                    logits = []
                    for j in range(args.num_modes):
                        traj_idx_start = j*(args.n_steps*2+1)
                        traj_idx_end = (j+1)*(args.n_steps*2+1)-1
                        trajectories.append(pred[traj_idx_start:traj_idx_end].reshape(-1, 2))
                        logits.append(pred[traj_idx_end].item())

                    # TODO: may want to limit to top TOP_K, like with covernet

                    all_predictions.append(Prediction(
                        instance_token,
                        sample_token,
                        np.array(trajectories),
                        F.softmax(torch.Tensor(logits), dim=0).numpy()).serialize())

            # model outputs are logits corresponding to a pre-built fixed trajectory set
            elif args.key == 'covernet':
                for i in range(model_pred.shape[0]):
                    trajectories_cpy = copy.deepcopy(trajectories)
                    logits = model_pred[i]
                    instance_token = instance_tokens[i]
                    sample_token = sample_tokens[i]

                    trajectories_cpy = trajectories[np.argsort(logits)[::-1]]
                    trajectories_cpy = trajectories_cpy[:args.top_k]
                    logits = logits[np.argsort(logits)[::-1]]
                    logits = logits[:args.top_k]

                    all_predictions.append(Prediction(
                        instance_token,
                        sample_token,
                        trajectories_cpy,
                        F.softmax(torch.Tensor(logits), dim=0).numpy()).serialize())

    return all_predictions


def main(args):
    print("Running with args:")
    print(vars(args))

    print("Device:")
    print(device)

    # load data
    nusc = NuScenes(version=args.version, dataroot=args.data_root)
    helper = PredictHelper(nusc)
    data_tokens = get_prediction_challenge_split(args.split_name, dataroot=args.data_root)
    dataset = CoverNetDataset(data_tokens, helper)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)
    print(f"Loaded split {args.split_name}, length {len(dataset)}, in {len(dataloader)} batches.")

    # prepare model
    model = get_model(args)
    model.load_state_dict(
        torch.load(os.path.join(args.experiment_dir, 'weights', args.weights)))

    model.eval()

    predictions = get_predictions(args, dataloader)
    json.dump(predictions,
              open(os.path.join(args.experiment_dir, f'{args.key}_preds_{datetime.datetime.now():%Y-%m-%d %Hh%Mm%Ss}.json'), "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run inference with a trained model.')

    parser.add_argument('--experiment_dir',
                        help='Directory with model training results')
    parser.add_argument('--weights',
                        help='Path to model weights, in directory [experiment_dir]/weights')
    parser.add_argument('--key',
                        help='Key to identify model architecture. E.g. mtp, covernet')

    # model arguments
    parser.add_argument('--num_modes', type=int)
    parser.add_argument('--backbone', help='Model backbone')
    parser.add_argument('--n_steps',
                        type=int,
                        help='Number of trajectory steps to predict',
                        default=12)
    parser.add_argument('--top_k',
                        type=int,
                        default=25,
                        help="Only write predictions to disk for the TOP_K most probable trajectories.")
    parser.add_argument('--freeze_bottom', dest='freeze_bottom',
                        help='Freeze the bottom layers of the backbone network, allowing only the top layers to be fine-tuned',
                        action='store_true')
    parser.add_argument('--no_freeze_bottom', dest='freeze_bottom',
                        help='Allow all params of the backbone to be fine-tuned',
                        action='store_false')
    parser.set_defaults(freeze_bottom=True)

    # data arguments
    parser.add_argument('--version', help='nuScenes version number.', default='v1.0-trainval')
    parser.add_argument('--data_root', help='Directory storing NuScenes data.', default='/home/jupyter/data/sets/nuscenes')
    parser.add_argument('--split_name', help='Data split to run inference on.', default='val')

    args = parser.parse_args()
    main(args)
