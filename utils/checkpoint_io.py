import os
import torch


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, method):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            print("epoch:", state_dict["epoch"])
            state_dict = state_dict[checkpoint_key]

        if "mokd" in method :
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
        return True
    else:
        print("There is no reference weights available for this model => Random weights will be used.")
        return False


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)


def load_pretrained_linear_weights(linear_classifier, ckpt_path, method):
    if not os.path.isfile(ckpt_path):
        print("Cannot find checkpoint at {}, Use random linear weights.".format(ckpt_path))
        return False
    print("Found checkpoint at {}".format(ckpt_path))
    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    linear_classifier.load_state_dict(state_dict, strict=True)
    return True


def restart_from_checkpoint(ckpt_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckpt_path):
        print("Checkpoint not founded in {}, train from random initialization".format(ckpt_path))
        return
    print("Found checkpoint at {}".format(ckpt_path))

    # open checkpoint file
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckpt_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckpt_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckpt_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckpt_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]