import os
import sys
import importlib
from general_utils.config_utils import load_config

def load_model(root, task, ckpt_path):
    model_config = load_config(os.path.join(ckpt_path, 'model.yaml'))

    cwd = os.getcwd()
    new_cwd = os.path.join(root, 'tasks', task)
    os.chdir(new_cwd)
    sys.path.append(new_cwd)
    get_model = getattr(importlib.import_module(f'tasks.{task}.models'), 'get_model')
    model = get_model(model_config, task)
    os.chdir(cwd)
    sys.path.remove(new_cwd)

    model.load_state_dict_from_path(os.path.join(ckpt_path, 'model.pt'))
    train_transform = model.make_train_transform()
    test_transform = model.make_test_transform()
    model.eval()
    return model, train_transform, test_transform

