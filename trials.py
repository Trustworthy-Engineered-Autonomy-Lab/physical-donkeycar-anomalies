#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    trials.py (run_trials) [--model=<model>] [--js] [--type=(linear|categorical)] [--camera=(single|stereo)] [--meta=<key:value> ...] [--myconfig=<filename>] [--random_start] [--trials=<trials>] [--trial_name=<trial_name>]

Options:
    --trials=100            Number of trials to run
    --random_start          Start at from a random position on the track.
    -h --help               Show this screen.
    --js                    Use physical joystick.
    -f --file=<file>        A text file containing paths to tub files, one per line. Option may be used more than once.
    --meta=<key:value>      Key/Value strings describing meta data about this drive. 
                            Examples: noise:gaussian brightness_coeff:0.8 cmd_latency:100 friction_scale:1.2
                            Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""
from docopt import docopt
import os
import time
import logging
import random
import json

#
# import cv2 early to avoid issue with importing after tensorflow
# see https://github.com/opencv/opencv/issues/14884#issuecomment-599852128
#
try:
    import cv2
except:
    pass


import donkeycar as dk
from donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.parts.tub_v2 import TubWriter
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController, WebFpv, JoystickController
from donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.parts.behavior import BehaviorPart
from donkeycar.parts.file_watcher import FileWatcher
from donkeycar.parts.launch import AiLaunch
from donkeycar.parts.kinematics import NormalizeSteeringAngle, UnnormalizeSteeringAngle, TwoWheelSteeringThrottle
from donkeycar.parts.kinematics import Unicycle, InverseUnicycle, UnicycleUnnormalizeAngularVelocity
from donkeycar.parts.kinematics import Bicycle, InverseBicycle, BicycleUnnormalizeAngularVelocity
from donkeycar.parts.explode import ExplodeDict
from donkeycar.parts.transform import Lambda
from donkeycar.parts.pipe import Pipe
from donkeycar.utils import *
from manage import drive

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_random_points(n, start_pos_dir='trials'):
    """
    Generate or load n random unique starting positions (0-303).
    Saves to trials/start_positions.json if file doesn't exist.
    Loads from file if it exists.
    
    Args:
        n: Number of starting positions to generate
        start_pos_dir: Directory to save/load positions from (default: 'trials')
    
    Returns:
        List of n unique random starting positions
    """
    os.makedirs(start_pos_dir, exist_ok=True)
    start_pos_file = os.path.join(start_pos_dir, 'start_positions.json')
    
    # If file exists, load from it
    if os.path.exists(start_pos_file):
        print(f"Loading start positions from {start_pos_file}")
        with open(start_pos_file, 'r') as f:
            positions = json.load(f)
        if len(positions) >= n:
            return positions[:n]
        else:
            print(f"Existing file has {len(positions)} positions, but {n} requested. Re-generating...")
    
    # Generate new random positions
    print(f"Generating {n} random start positions...")
    used_starts = set()
    positions = []
    
    for i in range(n):
        if len(used_starts) == 304:
            print('Warning: Number of iterations has exceeded the number of start positions available (304).')
            break
        
        start_pos = random.randint(0, 303)
        while start_pos in used_starts:
            start_pos = random.randint(0, 303)
        used_starts.add(start_pos)
        positions.append(start_pos)
    
    # Save to file
    with open(start_pos_file, 'w') as f:
        json.dump(positions, f, indent=2)
    print(f"Saved {len(positions)} start positions to {start_pos_file}")
    
    return positions



if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config(myconfig=args['--myconfig'])

    if args['run_trials']:
        model_type = args['--type']
        camera_type = args['--camera']
        iterations = int(args['--trials'])
        trial_name = args['--trial_name']
        meta_list = args['--meta'] or []
        anomaly_type = meta_list[0].split(':')[0] if meta_list else 'normal'
        intensity_param = float(meta_list[0].split(':')[1]) if meta_list and ':' in meta_list[0] else 0.0
        anomaly_flag = [item.split(':')[0] for item in meta_list] if meta_list else []
        cfg.GYM_CONF['random_start'] = bool(args['--random_start'])

        # Need to make new directory for this trial and save all the info to this one.
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        session_dir = os.path.join('trials', anomaly_type, timestamp)
        runs_dir = os.path.join(session_dir, 'runs')
        os.makedirs(runs_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Trial session directory: {os.path.abspath(session_dir)}")
        print(f"Results being saved to: {os.path.abspath(runs_dir)}")
        print(f"Anomaly type: {anomaly_type}")
        print(f"Anomalies: {anomaly_flag}")
        print(f"{'='*60}\n")

        # Generate or load random start positions
        start_positions = generate_random_points(iterations)

        for i in range(iterations):
            cfg.GYM_CONF['start_pos'] = start_positions[i]
            log_dir = os.path.join(runs_dir, f'log_{i}.csv')
            drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
                model_type=model_type, camera_type=camera_type,
                meta=meta_list, folder_name=session_dir + '/', log_dir=log_dir,
                run_id=i, anomaly_type=anomaly_type,
                intensity_param=intensity_param, anomaly_flag=anomaly_flag, runs_dir=runs_dir)
    
    