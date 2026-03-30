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
    --meta=<key:value>      Key/Value strings describing describing a piece of meta data about this drive. Option may be used more than once.
    --myconfig=filename     Specify myconfig file to use. 
                            [default: myconfig.py]
"""
from docopt import docopt
import os
import time
import logging
import random

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
        anomaly_flag = len(meta_list)
        cfg.GYM_CONF['random_start'] = bool(args['--random_start'])

        # Need to make new directory for this trial and save all the info to this one.
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        session_dir = os.path.join('trials', anomaly_type, timestamp)
        runs_dir = os.path.join(session_dir, 'runs')
        os.makedirs(runs_dir, exist_ok=True)

        for i in range(iterations):
            # Choose one of 304 random starting positions.
            start_pos = random.randint(0, 303)
            cfg.GYM_CONF['start_pos'] = start_pos
            log_dir = os.path.join(runs_dir, f'log_{i}.csv')
            drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
                model_type=model_type, camera_type=camera_type,
                meta=[], folder_name=session_dir + '/', log_dir=log_dir,
                run_id=i, anomaly_type=anomaly_type,
                intensity_param=intensity_param, anomaly_flag=anomaly_flag, runs_dir=runs_dir)
    
    