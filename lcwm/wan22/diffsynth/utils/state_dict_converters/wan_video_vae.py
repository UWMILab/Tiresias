def convert_wan_vae_state_dict(state_dict):
    converted = {}
    if 'model_state' in state_dict:
        state_dict = state_dict['model_state']
    for name in state_dict:
        converted['model.' + name] = state_dict[name]
    return converted
