import os
import logging
import time
import random
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union, Optional
from itertools import product
from tqdm import tqdm
import gin


EXPERIMENTS_PATH = 'results'
SearchSpace = List[Union[str, int, float]]


class Experiment(ABC):
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.root = Path(config_path).parent
        gin.parse_config_file(self.config_path)

    @gin.configurable()
    def build(self,
              experiment_name: str,
              module: str,
              repeat: int,
              variables_dict: Dict[str, SearchSpace]):
        logging.info('Creating experiment instances ...')
        experiment_path = os.path.join(EXPERIMENTS_PATH, experiment_name)
        variables_dict['repeat'] = list(range(repeat))
        variable_names, variables = zip(*variables_dict.items())
        for instance_values in tqdm(product(*variables)):
            instance_variables = dict(zip(variable_names, instance_values))
            instance_name = ','.join(['%s=%.4g' % (name.split('.')[-1], value)
                                      if isinstance(value, float)
                                      else '%s=%s' % (name.split('.')[-1], str(value).replace(' ', '_'))
                                      for name, value in instance_variables.items()])
            instance_path = os.path.join(experiment_path, instance_name)
            Path(instance_path).mkdir(parents=True, exist_ok=False)
            instance_config_path = os.path.join(instance_path, 'config.gin')
            copy(self.config_path, instance_config_path)
            with open(instance_config_path, 'a') as cfg:
                for name, value in instance_variables.items():
                    value = f"'{value}'" if isinstance(value, str) else str(value)
                    cfg.write(f'{name} = {value}\n')
            command_file = os.path.join(instance_path, 'command')
            with open(command_file, 'w') as cmd:
                cmd.write(f'python -m {module} '
                          f'--config_path={instance_config_path} '
                          f'run >> {instance_path}/instance.log 2>&1')

    @abstractmethod
    def instance(self):
        ...

    @gin.configurable()
    def run(self, timer: Optional[int] = 0):
        time.sleep(random.uniform(0, timer))
        running_flag = os.path.join(self.root, '_RUNNING')
        success_flag = os.path.join(self.root, '_SUCCESS')
        if os.path.isfile(success_flag) or os.path.isfile(running_flag):
            return
        elif not os.path.isfile(running_flag):
            Path(running_flag).touch()

        try:
            self.instance()
        except Exception as e:
            Path(running_flag).unlink()
            raise e
        except KeyboardInterrupt:
            Path(running_flag).unlink()
            raise Exception('KeyboardInterrupt')

        Path(running_flag).unlink()
        Path(success_flag).touch()

    def build_experiment(self):
        if EXPERIMENTS_PATH in str(self.root):
            raise Exception('Cannot build ensemble from ensemble member configuration.')
        self.build()
