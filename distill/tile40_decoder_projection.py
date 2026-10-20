"""Initialize smaller local decoders while preserving concat channel coordinates."""
from dataclasses import replace

import torch

from distill.student import Student


def project_decoder(source, width):
    if source.config.stage != 'decoder' or width > source.config.width or width < 8 or width % 8:
        raise ValueError('A smaller decoder width in multiples of8 is required.')
    target = Student(replace(source.config, width=width))
    old_widths = [source.config.width*min(2**index, 2) for index in range(source.config.depth)]
    new_widths = [width*min(2**index, 2) for index in range(source.config.depth)]
    original = source.state_dict()
    state = {}
    for name, expected in target.state_dict().items():
        value = original[name]
        if value.shape == expected.shape:
            state[name] = value.clone()
            continue
        if value.ndim == 4:
            columns = torch.arange(expected.shape[1])
            if name.startswith('decoders.') and ('.skip.weight' in name or '.layers.1.weight' in name):
                index = int(name.split('.')[1])
                old_up = old_widths[-1-index]
                new_up, new_skip = new_widths[-1-index], new_widths[-2-index]
                columns = torch.cat([torch.arange(new_up), torch.arange(old_up, old_up+new_skip)])
            value = value[:expected.shape[0]].index_select(1, columns)
        elif value.ndim == 1:
            value = value[:expected.shape[0]]
        else:
            raise ValueError(f'Unsupported projection tensor: {name}')
        if value.shape != expected.shape:
            raise ValueError(f'Projection changed the decoder interface: {name}')
        state[name] = value.clone()
    target.load_state_dict(state)
    return target.eval()
