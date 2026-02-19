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

#
# import cv2 early to avoid issue with importing after tensorflow
# see https://github.com/opencv/opencv/issues/14884#issuecomment-599852128
#
try:
    import cv2
except:
    pass


import donkeycar.donkeycar as dk
from donkeycar.donkeycar.parts.transform import TriggeredCallback, DelayedTrigger
from donkeycar.donkeycar.parts.tub_v2 import TubWriter
from donkeycar.donkeycar.parts.datastore import TubHandler
from donkeycar.donkeycar.parts.controller import LocalWebController, WebFpv, JoystickController
from donkeycar.donkeycar.parts.throttle_filter import ThrottleFilter
from donkeycar.donkeycar.parts.behavior import BehaviorPart
from donkeycar.donkeycar.parts.file_watcher import FileWatcher
from donkeycar.donkeycar.parts.launch import AiLaunch
from donkeycar.donkeycar.parts.kinematics import NormalizeSteeringAngle, UnnormalizeSteeringAngle, TwoWheelSteeringThrottle
from donkeycar.donkeycar.parts.kinematics import Unicycle, InverseUnicycle, UnicycleUnnormalizeAngularVelocity
from donkeycar.donkeycar.parts.kinematics import Bicycle, InverseBicycle, BicycleUnnormalizeAngularVelocity
from donkeycar.donkeycar.parts.explode import ExplodeDict
from donkeycar.donkeycar.parts.transform import Lambda
from donkeycar.donkeycar.parts.pipe import Pipe
from donkeycar.donkeycar.utils import *
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
        cfg.GYM_CONF['random_start'] = bool(args['--random_start'])

        # Need to make new directory for this trial and save all the info to this one.
        path = f'trials/{trial_name}'
        if os.path.exists(path):
            print('path already exists')
        else:
            os.makedirs(f'trials/{trial_name}')

        

        for i in range(iterations):
            trial_path = path + '_' + str(i)
            drive(cfg, model_path=args['--model'], use_joystick=args['--js'],
                model_type=model_type, camera_type=camera_type,
                meta=args['--meta'],folder_name=trial_path)
    
    