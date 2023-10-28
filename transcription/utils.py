import sys
from functools import reduce
from torch.nn.modules.module import _addindent
from torch.utils.data.sampler import Sampler
from matplotlib import pyplot as plt
import torch as th

def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    '''
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        # file.flush()
    '''
    print(string)
    return string, count

class CustomSampler(Sampler):
    def __init__(self, indexes):
        self.indexes = indexes
        self.n_batch = len(indexes)

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.n_batch

def cycle(iterable, set_epoch=False):
    epoch = 0
    while True:
        if set_epoch:
            iterable.sampler.set_epoch(epoch)
        for item in iterable:
            yield item
        epoch += 1

def draw_model_outs(frame_out, vel_out):
    batch_size = frame_out.shape[0] // 2
    fig, axes = plt.subplots(4, batch_size, figsize=(batch_size*10, 16))
    for n in range(batch_size):
        axes[0, n].imshow(th.argmax(frame_out[n], -1).cpu().detach().numpy().T, aspect='auto', origin='lower',
                    vmin=0, vmax=5, interpolation='nearest')
        axes[1, n].imshow(th.argmax(frame_out[n+batch_size], -1).cpu().detach().numpy().T, aspect='auto', origin='lower',
                    vmin=0, vmax=5, interpolation='nearest')
        axes[2, n].imshow(vel_out[n].cpu().detach().numpy().T, aspect='auto', origin='lower',
                    interpolation='nearest')
        axes[3, n].imshow(vel_out[n+batch_size].cpu().detach().numpy().T, aspect='auto', origin='lower',
                    interpolation='nearest')
    return fig