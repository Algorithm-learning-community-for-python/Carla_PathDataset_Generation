import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os

import pickle as pkl
from torchvision import transforms

import argparse
from tqdm import tqdm

p_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([23.0582], [27.3226]),
    transforms.Lambda(lambda x: F.log_softmax(x.reshape(-1), dim=0).reshape(x.shape[1:]))
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([23.0582], [27.3226])
])


def generateDistanceMaskFromColorMap(src, scene_size=(64, 64)):
    img = cv2.imread(src)
    img = cv2.resize(img, scene_size)
    raw_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(raw_image, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    raw_image = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    raw_map_image = cv2.resize(raw_image.astype(np.float32), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
    raw_map_image[raw_map_image < 0] = 0  # Uniform on drivable area
    raw_map_image = raw_map_image.max() - raw_map_image  # Invert values so that non-drivable area has smaller values

    image = img_transform(raw_image)
    prior = p_transform(raw_map_image)

    return image, prior


def get_agent_mask(agent_past, agent_future, agent_translation):
    map_width = 50
    map_height = 50

    num_agents = len(agent_past)
    future_agent_masks = [True] * num_agents

    past_agents_traj = [[[0., 0.]] * 4] * num_agents
    future_agents_traj = [[[0., 0.]] * 6] * num_agents

    past_agents_traj = np.array(past_agents_traj)
    future_agents_traj = np.array(future_agents_traj)

    past_agents_traj_len = [4] * num_agents
    future_agents_traj_len = [6] * num_agents

    decode_start_vel = [[0., 0.]] * num_agents
    decode_start_pos = [[0., 0.]] * num_agents

    for idx, path in enumerate(zip(agent_past, agent_future)):
        past = path[0]
        future = path[1]
        pose = agent_translation[idx]
        # print(pose)

        # agent filtering
        side_length = map_width // 2
        if len(past) == 0 or len(future) == 0 \
                or np.max(pose) > side_length or np.min(pose) < -side_length:
            future_agent_masks[idx] = False

        # agent trajectory
        if len(past) < 4:
            past_agents_traj_len[idx] = len(past)
        for i, point in enumerate(past[:4]):
            past_agents_traj[idx, i] = point

        if len(future) < 6:
            future_agents_traj_len[idx] = len(future)
        for i, point in enumerate(future[:6]):
            future_agents_traj[idx, i] = point

        # vel, pose
        if len(future) != 0 and not isinstance(agent_translation[idx], int):
            # print(agent_translation[idx])
            decode_start_vel[idx] = (future[0] - agent_translation[idx]) / 0.5
        decode_start_pos[idx] = agent_translation[idx]

    return past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
           future_agent_masks, decode_start_vel, decode_start_pos

def dataProcessing(traj_path, map_path, traj_list, map_list, idx = 0, virtual=False):

    scene_id = idx

    # map mask & prior mask
    whole_map_path = map_path + map_list[idx][1] + '/' + map_list[idx][0]
    map_image, prior = generateDistanceMaskFromColorMap(whole_map_path, scene_size=(64, 64))

    # agent mask
    whole_path = traj_path + traj_list[idx][1] + '/' + traj_list[idx][0]
    with open(whole_path, 'rb') as f:
        raw_path = pkl.load(f)
    agent_past = raw_path["agent_pasts"]
    agent_future = raw_path["agent_futures"][:,1:]
    agent_translation = raw_path["agent_futures"][:,0]
    # print(np.shape(raw_path["agent_futures"]))
    past_agents_traj, past_agents_traj_len, future_agents_traj, future_agents_traj_len, \
    future_agent_masks, decode_start_vel, decode_start_pos = get_agent_mask(agent_past, agent_future, agent_translation)

    episode = None
    episode = [past_agents_traj, past_agents_traj_len, future_agents_traj,
                future_agents_traj_len, future_agent_masks,
                np.array(decode_start_vel), np.array(decode_start_pos),
                map_image, prior, scene_id]

    return episode


def dataGeneration(traj_path, map_path, traj_list, map_list):
    episodes = []
    N = len(traj_list)

    print("{} number of samples".format(N))
    # count the number of curved agents

    # original data
    for idx in tqdm(range(N), desc='load data'):
        episode = dataProcessing(traj_path, map_path, traj_list, map_list, idx)
        if sum(episode[4]) > 0:
            episodes.append(episode)

    print("--- generation finished ---")

    return episodes


parser = argparse.ArgumentParser(description='load details')
parser.add_argument('--traj_path', type=str, help='path of trajectory', default='./data_pdh/path/')
parser.add_argument('--map_path', type=str, help='path of map', default='./data_pdh/map/')
parser.add_argument('--result_path', type=str, help='path for results', default='./')

args = parser.parse_args()

if __name__ == "__main__":
    TRAJ_PATH = args.traj_path
    MAP_PATH = args.map_path
    result_path = args.result_path

    extension = ".pkl"
    right_list = sorted([(name, 'right') for name in os.listdir(TRAJ_PATH + 'right/') if name.lower().endswith(extension)])
    left_list = sorted([(name, 'left') for name in os.listdir(TRAJ_PATH + 'left/') if name.lower().endswith(extension)])
    fair_length = max(len(right_list), len(left_list))
    for_list = sorted([(name, 'forward') for name in os.listdir(TRAJ_PATH + 'forward/') if name.lower().endswith(extension)])[:fair_length]
    stop_list = sorted([(name, 'stop') for name in os.listdir(TRAJ_PATH + 'stop/') if name.lower().endswith(extension)])[:fair_length]
    traj_list = sorted(for_list + right_list + left_list + stop_list)

    extension = ".jpg"
    right_map_list = sorted([(name, 'right') for name in os.listdir(MAP_PATH + 'right/') if name.lower().endswith(extension)])
    left_map_list = sorted([(name, 'left') for name in os.listdir(MAP_PATH + 'left/') if name.lower().endswith(extension)])
    # fair_length = max(len(right_list), len(left_list))
    for_map_list = sorted([(name, 'forward') for name in os.listdir(MAP_PATH + 'forward/') if name.lower().endswith(extension)])[:fair_length]
    stop_map_list = sorted([(name, 'stop') for name in os.listdir(MAP_PATH + 'stop/') if name.lower().endswith(extension)])[:fair_length]
    map_list = sorted(right_map_list + left_map_list + for_map_list + stop_map_list)

    # test
    episode = dataProcessing(TRAJ_PATH, MAP_PATH, traj_list, map_list)
    print("test 100: {}".format(episode))
    print("Generation start...")

    # main
    parsed_data = dataGeneration(TRAJ_PATH, MAP_PATH, traj_list, map_list)

    print("Number of Data: {}".format(len(parsed_data)))

    filename = result_path + 'carla_' + str(len(parsed_data))
    with open(filename + '.pickle', 'wb') as f:
        pkl.dump(parsed_data, f, pkl.HIGHEST_PROTOCOL)

    print("--- finished ---")
    print("number of data: {}".format(len(parsed_data)))


