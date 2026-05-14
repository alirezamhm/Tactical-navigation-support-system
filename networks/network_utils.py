import numpy as np
import torch.nn as nn
import torch


def weighted_huber_loss(weights, **kwargs):
    huber = nn.HuberLoss(reduction='none', **kwargs)
    def loss_fn(input, target):
        loss = huber(input, target) 
        w = torch.tensor(weights, device=input.device).view(1, -1, *([1] * (input.dim() - 2)))
        return (loss * w).mean()
    return loss_fn

loss_fns = {
    "mse": nn.MSELoss,
    "huber": nn.HuberLoss,
    "bce": nn.BCEWithLogitsLoss,
    "mse_sum": lambda **kwargs: lambda input, target: nn.MSELoss(**kwargs)(
        input.sum(dim=tuple(range(1, input.dim()))),
        target.sum(dim=tuple(range(1, target.dim())))
    ),
    "weighted_huber": lambda weights, **kwargs: weighted_huber_loss(weights, **kwargs),
}

def get_loss_fn(loss_name, *args, **kwargs):
    if loss_name in loss_fns:
        return loss_fns[loss_name](*args, **kwargs)
    raise ValueError(f"Unknown loss: {loss_name}")

def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def extract_swath_observation(swath_map, cur_grid_y, win_width, win_height):
    
    cropped_window = np.zeros((win_height, win_width))

    swath_coords = np.argwhere(swath_map[cur_grid_y:] == 1)
    horizontal_mid = int(np.mean(swath_coords[:, 1]))

    x_low_map = horizontal_mid - win_width // 2
    x_high_map = horizontal_mid + win_width // 2
    if x_low_map < 0:
        x_gap = abs(x_low_map)
        x_low_map += x_gap
        x_high_map += x_gap

    elif x_high_map > swath_map.shape[1]:
        x_gap = x_high_map - swath_map.shape[1]
        x_low_map -= x_gap
        x_high_map -= x_gap
    
    x_low_win = 0
    x_high_win = win_width

    y_low_map = cur_grid_y
    y_low_win = 0
    assert y_low_map >= 0, print("cropping y low bound negative! ", y_low_map)

    y_high_map = y_low_map + win_height
    y_high_win = win_height
    assert y_high_map <= swath_map.shape[0], print("y_high_map exceeds environment height", y_high_map)

    assert (y_high_map - y_low_map) == (y_high_win - y_low_win), print("y-dim not same size!", y_low_map, y_high_map, y_low_win, y_high_win)
    assert (x_high_map - x_low_map) == (x_high_win - x_low_win), print("x-dim not same size!", x_low_map, x_high_map, x_low_win, x_high_win)

    cropped_window[y_low_win:y_high_win, x_low_win:x_high_win] = swath_map[y_low_map:y_high_map, x_low_map:x_high_map]
    return cropped_window, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win


def extract_observation_window(grid_map, win_width, win_height, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win):
    
    if len(grid_map.shape) == 2:
        cropped_window = np.zeros((win_height, win_width))
    elif len(grid_map.shape) == 3:
        cropped_window = np.zeros((win_height, win_width, grid_map.shape[2]))

    cropped_window[y_low_win:y_high_win, x_low_win:x_high_win] = grid_map[y_low_map:y_high_map, x_low_map:x_high_map]
    return cropped_window