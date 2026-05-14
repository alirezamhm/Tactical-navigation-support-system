import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import os
import numpy as np
import random
import gc
import pickle
import lightning as L
from networks.network_utils import extract_swath_observation, extract_observation_window
import torch.distributed as dist


class OccupancyDataset(Dataset):
    def __init__(
        self,
        concentrations=[0.5],
        trial_range=[0, 2],
        data_dir=None,
        local_window_height=80,
        local_window_width=80,
        max_swath_val=9,
        min_swath_val=0,
        max_thickness=1,
        max_velocity=1,
        max_cost=1,
        num_data_per_step=3,
        shuffle=False,
        skip_steps=4,
    ):

        self.concentrations = concentrations
        self.trial_range = trial_range
        self.local_window_height = local_window_height
        self.local_window_width = local_window_width
        
        self.max_thickness = max_thickness
        self.max_velocity = max_velocity
        self.max_cost = max_cost

        self.data_per_trial = 0
        self.data_per_conc = 0
        self.total_len = 0

        self.inputs = []
        self.occ_labels = []
        self.cost_labels = []
        for conc in concentrations:
            print(f"Loading data for concentration {conc}")
            save_trial = 0
            for i, trial_idx in enumerate(range(trial_range[0], trial_range[1])):
                print(f"\ttrial {trial_idx}", end="\r")
                trial_path = os.path.join(data_dir, f"{int(conc*100)}", f"t{trial_idx}")

                occ_observations = np.load(os.path.join(trial_path, "occ.npz"))['data'] 

                with open(os.path.join(trial_path, "planner.pkl"), "rb") as file:
                    pk_dict = pickle.load(file)

                footprint_observations = pk_dict["footprint"] 

                swath_observations = pk_dict["swath"] 

                cur_grids = np.array(pk_dict['cur_grids']) 
                
                metrics = np.array(pk_dict['metrics'])

                max_step = occ_observations.shape[0]
                max_t = max_step - num_data_per_step
                for t in range(0, max_t, skip_steps):
                    cur_grid_x, cur_grid_y = cur_grids[t]

                    occ_observation = occ_observations[t] 

                    footprint_idx = footprint_observations[t] 
                    footprint_observation = np.zeros((occ_observation.shape[:2]))
                    
                    footprint_observation[footprint_idx[:, 0], footprint_idx[:, 1]] = 1 

                    if cur_grid_y + self.local_window_height > occ_observation.shape[0]:
                        break

                    max_k = min(max_swath_val, max_step - 1 - t)
                    k_steps = random.sample(range(min_swath_val, max_k + 1), num_data_per_step)

                    for k in k_steps:
                        swath_idx = swath_observations[t + k]
                        swath_observation = np.zeros((occ_observation.shape[:2]))
                        swath_observation[swath_idx[:, 0], swath_idx[:, 1]] = 1 
                        
                        occ_observation_label = occ_observations[t + k]
                        cost = metrics[t+k][0] - metrics[t][0]
                        (
                            swath_obs,
                            x_low_map,
                            x_high_map,
                            y_low_map,
                            y_high_map,
                            x_low_win,
                            x_high_win,
                            y_low_win,
                            y_high_win,
                        ) = extract_swath_observation(
                            swath_map=swath_observation,
                            cur_grid_y=cur_grid_y,
                            win_height=self.local_window_height,
                            win_width=self.local_window_height,
                        )
                        swath_obs = np.expand_dims(swath_obs, axis=-1)

                        occ_map_obs = extract_observation_window(
                            occ_observation,
                            self.local_window_width,
                            self.local_window_height,
                            x_low_map,
                            x_high_map,
                            y_low_map,
                            y_high_map,
                            x_low_win,
                            x_high_win,
                            y_low_win,
                            y_high_win,
                        )
                        
                        footprint = extract_observation_window(
                            footprint_observation,
                            self.local_window_width,
                            self.local_window_height,
                            x_low_map,
                            x_high_map,
                            y_low_map,
                            y_high_map,
                            x_low_win,
                            x_high_win,
                            y_low_win,
                            y_high_win,
                        )
                        footprint = np.expand_dims(footprint, axis=-1)
                        
                        step_input =  np.transpose(
                                np.concatenate([occ_map_obs, footprint, swath_obs], axis=-1),
                                (2, 0, 1)
                            )

                        occ_label = extract_observation_window(
                            occ_observation_label,
                            self.local_window_width,
                            self.local_window_height,
                            x_low_map,
                            x_high_map,
                            y_low_map,
                            y_high_map,
                            x_low_win,
                            x_high_win,
                            y_low_win,
                            y_high_win,
                        )
                        occ_label = np.transpose(occ_label, (2, 0, 1))
                        

                        self.inputs.append(step_input)
                        self.occ_labels.append(occ_label)
                        self.cost_labels.append(cost)
                        
                if trial_idx == trial_range[0] and self.data_per_trial == 0:
                    self.data_per_trial = len(self.inputs)
                    self.data_per_conc = (trial_range[1] - trial_range[0]) * len(self.inputs)

                del (
                    occ_observations,
                    footprint_observations,
                    swath_observations,
                    cur_grids,
                    metrics,
                )
                gc.collect()

        self.inputs = np.array(self.inputs)
        self.occ_labels = np.array(self.occ_labels)
        self.cost_labels = np.array(self.cost_labels)
        self.total_len = self.inputs.shape[0]
        if shuffle:
            self.shuffle()

        print("\nData per trial: ", self.data_per_trial)
        print("Total Length: ", self.total_len)

    def shuffle(self):
        """
        Shuffle the dataset in place.
        """
        indices = np.arange(self.total_len)
        np.random.shuffle(indices)
        self.inputs = self.inputs[indices]
        self.occ_labels = self.occ_labels[indices]
        self.cost_labels = self.cost_labels[indices]

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):

        input = self.inputs[index] 
        occ_label = self.occ_labels[index]
        cost_label = self.cost_labels[index]

        input = self.normalize_channels(input)
        occ_label = self.normalize_channels(occ_label)
        cost_label = cost_label / self.max_cost
        
        input = torch.Tensor(input)
        occ_label = torch.Tensor(occ_label)
        cost_label = torch.Tensor([cost_label])

        return input, (occ_label, cost_label)

    def normalize_channels(self, input):
        """
        Normalize the thickness and velocity channels of the input tensor by the max values.
        Note that this function assumes the second channel is thickness and the third and fourth channels are velocity.
        Args:
            input (torch.Tensor): The input tensor of shape (C, H, W), where C is the number of channels.
        Returns:
            torch.Tensor: The normalized input tensor.
        """
        input[1] = input[1] / self.max_thickness
        input[2] = input[2] / self.max_velocity
        input[3] = input[3] / self.max_velocity
        return input
    

def preprocess_save(data_dir, output_dir, trial_range, chunk_size, concentration, dataset_kwargs={}):
    os.makedirs(output_dir, exist_ok=True)

    trial_start, trial_end = trial_range
    for chunk_start in range(trial_start, trial_end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, trial_end)
        print(f"Processing trials {chunk_start} to {chunk_end}")

        dataset = OccupancyDataset(
            data_dir=data_dir,
            trial_range=(chunk_start, chunk_end),
            concentrations=[concentration],
            **dataset_kwargs
        )

        inputs, occ_labels, cost_labels = [], [], []

        for input_tensor, (occ_label_tensor, cost_label_tensor) in dataset:
            inputs.append(input_tensor.numpy())
            occ_labels.append(occ_label_tensor.numpy())
            cost_labels.append(cost_label_tensor.numpy())

        inputs_np = np.stack(inputs)
        occ_labels_np = np.stack(occ_labels)
        cost_labels_np = np.stack(cost_labels)

        prefix = f"trials_{chunk_start}_{chunk_end}"
        np.save(os.path.join(output_dir, f"{prefix}_inputs.npy"), inputs_np)
        np.save(os.path.join(output_dir, f"{prefix}_occ_labels.npy"), occ_labels_np)
        np.save(os.path.join(output_dir, f"{prefix}_cost_labels.npy"), cost_labels_np)

        
        print(f"Saved chunk {chunk_start}-{chunk_end}:")
        print(f"  inputs shape:     {inputs_np.shape}")
        print(f"  occ_labels shape: {occ_labels_np.shape}")
        print(f"  cost_labels shape:{cost_labels_np.shape}")
        
        gc.collect()  

class ChunkedWrapper(IterableDataset):
    def __init__(self, concentrations, data_dir, trial_range, chunk_size, num_chunks, stage, cost_log=False):
        super().__init__()
        self.concentrations = concentrations
        self.data_dir = data_dir
        self.trial_range = trial_range
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.stage = stage
        self.cost_log = cost_log
        
    def __iter__(self):
        print('starting chunk')
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f'found rank {rank}')
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        print(f'wod size {world_size}')
        worker = get_worker_info()
        num_w = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        print(f"[Rank {rank}/{world_size}] Worker {wid}/{num_w} started\n")

        trials = list(range(self.trial_range[0], self.trial_range[1], self.chunk_size*self.num_chunks))
        
        # share across ranks
        rank_trials = trials[rank::world_size]
        
        # share across workers
        worker_trials = rank_trials[wid::num_w]

        for start in worker_trials:
            inputs, occ_labels, cost_labels = [], [], []
            for conc in self.concentrations:
                for i in range(self.num_chunks):
                    chunk_start = start + i * self.chunk_size
                    chunk_end = min(chunk_start + self.chunk_size, self.trial_range[1])
                    if chunk_start >= chunk_end:
                        continue
                    prefix = os.path.join(self.data_dir, str(conc), f"trials_{chunk_start}_{chunk_end}")
                    print(f"[Rank {rank}/{world_size}] Loading chunk {prefix}\n")
                    inputs.append(np.load(f"{prefix}_inputs.npy"))
                    occ_labels.append(np.load(f"{prefix}_occ_labels.npy"))
                    cost_labels.append(np.load(f"{prefix}_cost_labels.npy"))
            inputs = np.concatenate(inputs, axis=0)
            occ_labels = np.concatenate(occ_labels, axis=0)
            cost_labels = np.concatenate(cost_labels, axis=0)
            
            original_len = inputs.shape[0]
            
            if self.stage == "cls":
                # for classification, we only care about whether cost is 0 or not
                cost_labels = (cost_labels > 0).astype(float)
            elif self.stage == "reg":
                # for regression, we only care about the non-zero cost data
                pos_idx = np.where(cost_labels > 0)[0]
                if len(pos_idx) == 0:
                    pos_idx = np.random.choice(np.arange(original_len), size=1, replace=True)
                
                inputs = inputs[pos_idx]
                occ_labels = occ_labels[pos_idx]
                cost_labels = cost_labels[pos_idx]
                
                if inputs.shape[0] < original_len:
                    # upsample the positive samples to match the original length
                    pad_idx = np.random.choice(np.arange(inputs.shape[0]), size=original_len - inputs.shape[0], replace=True)
                    inputs = np.concatenate([inputs, inputs[pad_idx]], axis=0)
                    occ_labels = np.concatenate([occ_labels, occ_labels[pad_idx]], axis=0)
                    cost_labels = np.concatenate([cost_labels, cost_labels[pad_idx]], axis=0)
                
                if self.cost_log:
                    cost_labels = np.log(cost_labels)
                    
            
            # Shuffle the data before yielding
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs = inputs[indices]
            occ_labels = occ_labels[indices]
            cost_labels = cost_labels[indices]
            for i in range(inputs.shape[0]):
                yield torch.Tensor(inputs[i]), (torch.Tensor(occ_labels[i]), torch.Tensor(cost_labels[i]))
            
            # cleanup
            torch.cuda.empty_cache()
            
            
class OccupancyDataModule(L.LightningDataModule):
    def __init__(self, concentrations, data_dir, train_trial_range, val_trial_range, test_trial_range, chunk_size, num_chunks, batch_size, num_workers, stage, cost_log=False):
        super().__init__()
        self.batch_size, self.num_workers = batch_size, num_workers
        common_args = {
            'concentrations': concentrations,
            'data_dir': data_dir,
            'chunk_size': chunk_size,
            'num_chunks': num_chunks,
            'stage': stage,
            'cost_log': cost_log,
        }
        self.train_args = {**common_args, 'trial_range': train_trial_range}
        self.val_args = {**common_args, 'trial_range': val_trial_range}
        self.test_args = {**common_args, 'trial_range': test_trial_range}

    def train_dataloader(self):
        ds = ChunkedWrapper(**self.train_args)
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)
        
    def val_dataloader(self):
        ds = ChunkedWrapper(**self.val_args)
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=True,
                          pin_memory=True)
        
    def test_dataloader(self):
        ds = ChunkedWrapper(**self.test_args)
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
        
        
