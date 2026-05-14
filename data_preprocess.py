from networks.dataset import preprocess_save

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess and save dataset')
    parser.add_argument('--data_dir', type=str, default='./data/trials', help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='./data/processed', help='Directory to save preprocessed data')
    parser.add_argument('--trial_range', type=int, nargs=2, required=True, help="Trial range as two integers, e.g., 0 50.")
    parser.add_argument('--chunksize', type=int, default=25, help='Chunk size for data loading')
    parser.add_argument("--conc", type=int, choices=[20, 30, 40, 50], required=True, help="Concentration, value must be from [20, 30, 40, 50].")
    parser.add_argument('-wh', '--window_height', type=int, default=80, help='Height of the observation window')
    parser.add_argument('-ww', '--window_width', type=int, default=80, help='Width of the observation window')
   
    args = parser.parse_args()

    MAX_THICKNESS = 1
    MAX_VELOCITY = 30
    MAX_COST = 2e13
    
    dataset_kwargs = {
        'pre_load': True,
        'max_thickness': MAX_THICKNESS,
        'max_velocity': MAX_VELOCITY,
        'max_cost': MAX_COST,
        'local_window_height': args.window_height,
        'local_window_width': args.window_width,
        'min_swath_val': 0,
        'max_swath_val': 9,
        'num_data_per_step': 3,
        'skip_steps': 3,
        'cache_data': False,
    }

    preprocess_save(data_dir=args.data_dir,
                    output_dir=args.save_dir,
                    trial_range=args.trial_range,
                    chunk_size=args.chunksize,
                    concentration=args.conc / 100,
                    dataset_kwargs=dataset_kwargs,)
    
    print(f"Data preprocessed and saved to {args.save_dir}")