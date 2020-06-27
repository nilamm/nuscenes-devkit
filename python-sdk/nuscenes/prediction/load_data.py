import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

SECONDS_OF_HISTORY = 2
SECONDS_TO_PREDICT = 6


#starting off with the same Dataset as MTP
class CoverNetDataset(Dataset):
    def __init__(self,
                 instance_sample_tokens,
                 helper):
        self.instance_sample_tokens = instance_sample_tokens
        self.helper = helper

        self.static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=SECONDS_OF_HISTORY)
        self.mtp_input_representation = InputRepresentation(
            self.static_layer_rasterizer,
            self.agent_rasterizer,
            Rasterizer())

        self.transform_fn = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    def __len__(self):
        return len(self.instance_sample_tokens)
    def __getitem__(self, idx):
        instance_token, sample_token = self.instance_sample_tokens[idx].split("_")

        image = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
        image = np.reshape(image, (3,500,500))
        image = image/255
        image_tensor = torch.from_numpy(image)
        image_tensor = self.transform_fn(image_tensor)

        agent_state_vector = np.array([self.helper.get_velocity_for_agent(instance_token, sample_token),
                                            self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                            self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)])

        ground_truth = self.helper.get_future_for_agent(instance_token, sample_token, seconds=SECONDS_TO_PREDICT, in_agent_frame=True)
        
        return (image_tensor, agent_state_vector,
                ground_truth.reshape((1, 12, 2)),
                instance_token, sample_token)

    

class MTPDataset(Dataset):
    def __init__(self,
                 instance_sample_tokens,
                 helper):
        self.instance_sample_tokens = instance_sample_tokens
        self.helper = helper

        self.static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=SECONDS_OF_HISTORY)
        self.mtp_input_representation = InputRepresentation(
            self.static_layer_rasterizer,
            self.agent_rasterizer,
            Rasterizer())

        self.transform_fn = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.instance_sample_tokens)

    def __getitem__(self, idx):
        instance_token, sample_token = self.instance_sample_tokens[idx].split("_")

        image = self.mtp_input_representation.make_input_representation(instance_token, sample_token)
        image = np.reshape(image, (3,500,500))
        image = image/255
        image_tensor = torch.from_numpy(image)
        image_tensor = self.transform_fn(image_tensor)

        agent_state_vector = np.array([self.helper.get_velocity_for_agent(instance_token, sample_token),
                                            self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                            self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)])

        # For MTP: "the targets are of shape [batch_size, 1, n_timesteps, 2]"
        # where n_timesteps = 2 * seconds predicted = 12
        ground_truth = self.helper.get_future_for_agent(instance_token, sample_token, seconds=SECONDS_TO_PREDICT, in_agent_frame=True)


        return (image_tensor, agent_state_vector,
                ground_truth.reshape((1, 12, 2)),
                instance_token, sample_token)
