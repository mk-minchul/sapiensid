import os
from typing import Union
import torch
from torch import device
from .utils import get_parameter_device, get_parameter_dtype, save_state_dict_and_config, load_state_dict_from_path
import time
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

class BaseModel(torch.nn.Module):
    """
    A base model class that provides a template for implementing models. It includes methods for
    loading, saving, and managing model configurations and states. This class is designed to be
    extended by specific model implementations.

    Attributes:
        config (object): Configuration object containing model settings.
        input_color_flip (bool): Whether to flip the color channels from BGR to RGB.
    """

    def __init__(self, config=None):
        """
        Initializes the BaseModel class.

        Parameters:
            config (object, optional): Configuration object containing model settings.
        """
        super(BaseModel, self).__init__()
        self.config = config
        if self.config.color_space == 'BGR':
            self.input_color_flip = True
            self._config_color_space = 'BGR'
            self.config.color_space = 'RGB'
        else:
            self.input_color_flip = False

    def forward(self, x):
        """
        Forward pass of the model. Needs to be implemented in subclass.

        Parameters:
            x (torch.Tensor): Input tensor.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError('forward must be implemented in subclass')

    @classmethod
    def from_config(cls, config) -> "BaseModel":
        """
        Creates an instance of this class from a configuration object. Needs to be implemented in subclass.

        Parameters:
            config (object): Configuration object.

        Returns:
            BaseModel: An instance of the subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError('from_config must be implemented in subclass')

    def make_train_transform(self):
        """
        Creates training data transformations. Needs to be implemented in subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError('make_train_transform must be implemented in subclass')

    def make_test_transform(self):
        """
        Creates testing data transformations. Needs to be implemented in subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError('make_test_transform must be implemented in subclass')

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        name: str = 'model.pt',
        rank: int = 0,
    ):
        """
        Saves the model's state_dict and configuration to the specified directory.

        Parameters:
            save_dir (Union[str, os.PathLike]): The directory to save the model.
            name (str, optional): The name of the file to save the model as. Default is 'model.pt'.
            rank (int, optional): The rank of the process (used in distributed training). Default is 0.
        """
        save_path = os.path.join(save_dir, name)
        if rank == 0:
            save_state_dict_and_config(self.state_dict(), self.config, save_path)

    def load_state_dict_from_path(self, pretrained_model_path, device='cpu'):
        if os.path.exists(pretrained_model_path.replace('.pt', '.tar')):
            # body model legacy
            pretrained_model_path = pretrained_model_path.replace('.pt', '.tar')
            state_dict = load_state_dict_from_path(pretrained_model_path)
            state_dict = state_dict['model_state_dict']
            state_dict = {'net.'+k: v for k, v in state_dict.items()}
        elif os.path.exists(pretrained_model_path.replace('model.pt', 'checkpoint.pth')):
            state_dict = load_state_dict_from_path(pretrained_model_path.replace('model.pt', 'checkpoint.pth'))
            state_dict = state_dict['model']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            state_dict = {'net.'+k: v for k, v in state_dict.items()}
        elif os.path.isfile(os.path.dirname(pretrained_model_path)):
            state_dict = load_state_dict_from_path(os.path.dirname(pretrained_model_path))
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            state_dict = {'net.'+k: v for k, v in state_dict.items()}
            state_dict = {k.replace('.module.', '.'): v for k, v in state_dict.items()}
        else:
            state_dict = load_state_dict_from_path(pretrained_model_path)

        if 'net.vit' in list(self.state_dict().keys())[-1] and 'pretrained_models' in pretrained_model_path:
            state_dict = {k.replace('net', 'net.vit'): v for k, v in state_dict.items()}
        
        if 'model_state_dict' in state_dict:
            # dq sapiens model
            state_dict = state_dict['model_state_dict']
            state_dict = {k:v for k, v in state_dict.items() if k.startswith('model.')}
            state_dict = {k[6:]: v for k, v in state_dict.items()}

        st_keys = list(state_dict.keys())
        self_keys = list(self.state_dict().keys())
        print('\n')
        print(f'{GREEN}Check{RESET}')
        print( f'{CYAN}compatible keys in state_dict {YELLOW}{len(set(st_keys).intersection(set(self_keys)))}{RESET}/{len(st_keys)}')
        print(f'{CYAN}missing keys in state_dict {RED}{len(set(self_keys) - set(st_keys))}{RESET}/{len(self_keys)}')
        if 'net.pos_embed' in state_dict:
            if state_dict['net.pos_embed'].shape != self.state_dict()['net.pos_embed'].shape:
                # resizing the input leads to different shape pos_embed
                print(f'{RED}pos_embed shape mismatch{RESET}. Adjusting shape')
                from models.pos_emb_utils import _interpolate_pos_embed, _pad_pos_embed
                pos_embed = state_dict['net.pos_embed']
                h, w = self.config.input_size[1], self.config.input_size[2]
                patch_size = self.net.patch_embed.proj.kernel_size[0]
                if self.config.pos_embded_interpolate_method == 'resize':
                    resized_pos_embed = _interpolate_pos_embed(pos_embed, w, h, patch_size=patch_size)
                elif self.config.pos_embded_interpolate_method == 'pad':
                    resized_pos_embed = _pad_pos_embed(pos_embed, w, h, patch_size=patch_size)
                else:
                    raise ValueError(f'Unknown pos_embded_interpolate_method: {self.config.pos_embded_interpolate_method}')
                state_dict['net.pos_embed'] = resized_pos_embed
        if 'net.feature.0.weight' in state_dict:
            if state_dict['net.feature.0.weight'].shape[1] != self.state_dict()['net.feature.0.weight'].shape[1]:
                # it could happen if there is no config.yaml and the model should be using wrapper..
                # then add config.yaml
                # resizing the input leads to different shape linear layer
                print(f'{RED}feature.0.weight shape mismatch{RESET}. Adjusting shape')
                st_weight = state_dict['net.feature.0.weight']
                diff = self.state_dict()['net.feature.0.weight'].shape[1] - st_weight.shape[1]
                new_weight = torch.cat([st_weight, st_weight[:, -diff:]], dim=1)
                state_dict['net.feature.0.weight'] = new_weight
        if self.device != 'cpu':
            for k, v in state_dict.items():
                state_dict[k] = v.to(device)
        result = self.load_state_dict(state_dict, strict=False)
        print(result)
        print(f"{GREEN}Loaded pretrained model from {pretrained_model_path}{RESET}")
        print('\n')
        time.sleep(1)
        return result


    @property
    def device(self) -> device:
        """
        Returns the device of the model's parameters.

        Returns:
            device: The device the model is on.
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        Returns the data type of the model's parameters.

        Returns:
            torch.dtype: The data type of the model.
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False) -> int:
        """
        Returns the number of parameters in the model, optionally filtering only trainable parameters.

        Parameters:
            only_trainable (bool, optional): Whether to count only trainable parameters. Default is False.

        Returns:
            int: The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

    def has_trainable_params(self):
        """
        Checks if the model has any trainable parameters.

        Returns:
            bool: True if the model has trainable parameters, False otherwise.
        """
        return any(p.requires_grad for p in self.parameters())
