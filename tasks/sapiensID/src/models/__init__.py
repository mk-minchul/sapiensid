

def get_model(model_config):

    if '/sapiens_id' in model_config.yaml_path:
        from .sapiens_id import load_model as load_sapiens_kprpe_model
        model = load_sapiens_kprpe_model(model_config)
        print('Loaded Sapeins ID model')
    else:
        raise NotImplementedError(f"Model {model_config.yaml_path} not implemented")
    if model_config.start_from:
        model.load_state_dict_from_path(model_config.start_from)

    if model_config.freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model
