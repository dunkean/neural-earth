"""Project the native teacher with explicit channel, skip and attention indices.

Embeddings retain their original coordinates. Attention QKV outputs are
interleaved per channel; whole heads are selected together. Calibration scores
must come from training scenes, never evaluation worlds.
"""
from __future__ import annotations

import torch


def compressed_config(config, width, blocks):
    config = {key: value for key, value in dict(config).items() if not key.startswith('_')}
    if config['layers_per_block'] != 3 or not 1 <= blocks <= 3:
        raise ValueError('Compression requires the pinned three-block teacher.')
    if not 0 < width <= config['model_channels']:
        raise ValueError('Compressed width must be positive and no larger than teacher.')
    options = config.get('block_kwargs') or {}
    if options.get('conv_type', 'default') != 'default' or options.get('expansion_factor', 1) != 1:
        raise ValueError('Only the pinned default, unexpanded MP convolutions are supported.')
    head = options.get('channels_per_head', 64)
    if config.get('midblock_attention', True) and width*max(config['model_channel_mults']) % head:
        raise ValueError('Bottleneck width must preserve complete attention heads.')
    return config | dict(model_channels=width, layers_per_block=blocks,
        emb_channels=config.get('emb_channels') or config['model_channels']*max(config['model_channel_mults']),
        noise_emb_dims=config['model_channels'] if config.get('noise_emb_dims') is None else config['noise_emb_dims'])


def block_source(name, blocks):
    parts = name.split('.')
    if len(parts) > 1 and parts[0] in ('enc', 'dec') and '_block' in parts[1]:
        prefix, index = parts[1].rsplit('_block', 1)
        retained = ({1: (0,), 2: (0, 2), 3: (0, 1, 2)} if parts[0] == 'enc' else
                    {1: (0, 3), 2: (0, 2, 3), 3: (0, 1, 2, 3)})[blocks]
        parts[1] = f'{prefix}_block{retained[int(index)]}'
    return '.'.join(parts)


def select_channels(old, new, head_size=64, scores=None):
    if not 0 < new <= old:
        raise ValueError('Invalid projection dimensions.')
    if scores is not None and (scores.shape != (old,) or not torch.isfinite(scores).all()):
        raise ValueError('Calibration scores must be finite, one per teacher channel.')
    if old == new:
        return torch.arange(old)
    if old % head_size == 0 and new % head_size == 0:
        old_heads, new_heads = old//head_size, new//head_size
        if scores is None:
            heads = torch.linspace(0, old_heads-1, new_heads).round().long()
        else:
            heads = torch.argsort(scores.reshape(old_heads, head_size).mean(1), descending=True, stable=True)[:new_heads]
        return (heads.sort().values[:, None]*head_size+torch.arange(head_size)).flatten()
    if scores is None:
        return torch.linspace(0, old-1, new).round().long()
    return torch.argsort(scores, descending=True, stable=True)[:new].sort().values


def project_state(teacher, student, scores=None):
    """Return exact teacher subsets and a recoverable axis/source mapping.

    No shape-based prefix slicing: decoder inputs consist of two separate
    channel groups, and QKV has a third interleaved axis.
    """
    scores = scores or {}
    source = teacher.state_dict()
    target = student.state_dict()
    blocks = student.config.layers_per_block
    options = student.config.block_kwargs or {}
    head_size = options.get('channels_per_head', 64)
    axes, channel_sets = {}, {}

    def channels(old, new):
        key = (old, new)
        if key not in channel_sets:
            channel_sets[key] = select_channels(old, new, head_size, scores.get(str(old)))
        return channel_sets[key]

    def register(name, output, inputs):
        axes[name+'.weight'] = (output, inputs)

    def block_axes(name, new_block, old_block, current, old_input):
        out = channels(old_block.out_channels, new_block.out_channels)
        if new_block.conv_skip is not None:
            if old_block.conv_skip is None:
                raise ValueError('Compression unexpectedly introduced a skip projection.')
            register(name+'.conv_skip', out, current)
        register(name+'.conv_res0', out, out if new_block.mode == 'enc' else current)
        register(name+'.conv_res1', out, out)
        if new_block.emb_linear is not None:
            register(name+'.emb_linear', out, torch.arange(old_block.emb_linear.weight.shape[1]))
        if new_block.num_heads:
            if not old_block.num_heads or out.numel() % head_size:
                raise ValueError('Attention projection lost complete teacher heads.')
            # Layout is (head, channel_within_head, QKV), not (Q, K, V).
            qkv = (out[:, None]*3+torch.arange(3)).flatten()
            register(name+'.attn_qkv', qkv, out)
            register(name+'.attn_proj', out, out)
        elif old_block.num_heads:
            raise ValueError('Compression cannot silently remove teacher attention.')
        return out, old_block.out_channels

    current = torch.arange(teacher.config.in_channels+1)
    old_input = current.numel()
    skips = []
    for name, block in student.enc.items():
        source_name = block_source('enc.'+name, blocks)
        old = teacher.enc[source_name.split('.')[1]]
        if name.endswith('_conv'):
            output = channels(old.weight.shape[0], block.weight.shape[0])
            register('enc.'+name, output, current)
            current, old_input = output, old.weight.shape[0]
        else:
            current, old_input = block_axes('enc.'+name, block, old, current, old_input)
        skips.append((current, old_input))
    for name, block in student.dec.items():
        old = teacher.dec[block_source('dec.'+name, blocks).split('.')[1]]
        if '_block' in name:
            skip, old_skip = skips.pop()
            joined = torch.cat((current, skip+old_input))
            if old.conv_res0.weight.shape[1] != old_input+old_skip:
                raise ValueError('Teacher decoder skip correspondence changed.')
            current = joined
        current, old_input = block_axes('dec.'+name, block, old, current, old_input)
    if skips:
        raise ValueError('Not all native skip connections were projected.')
    register('out_conv', torch.arange(teacher.config.out_channels), current)
    result, mapping = {}, {}
    for name, expected in target.items():
        parent = block_source(name, blocks)
        value = source[parent]
        plan = axes.get(name)
        if plan is not None:
            output, inputs = plan
            value = value.index_select(0, output.to(value.device)).index_select(1, inputs.to(value.device))
        if value.shape != expected.shape:
            raise ValueError(f'Unmapped native compression dimension: {name}: {value.shape} != {expected.shape}')
        result[name] = value.detach().clone()
        mapping[name] = dict(source=parent, output=plan[0].tolist() if plan else None,
                             inputs=plan[1].tolist() if plan else None)
    return result, mapping
