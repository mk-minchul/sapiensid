from .kp19_preprocessor import KP19Preprocessor

def make_kp_preprocessor(config):
    kp_preproc_config = config.kp_preprocessor
    if kp_preproc_config.name == 'none':
        return None
    elif kp_preproc_config.name == 'kp19':
        return KP19Preprocessor(config)
    else:
        raise NotImplementedError(f'Preprocessor {kp_preproc_config.name} is not implemented')