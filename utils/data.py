__all__ = ('ROOT', 'BASES', 'GROUPS', 'REPLICA_COUNT', 'SEQUENCE_LENGTH',
           'path', 'load', 'load_sequence', 'average_fitness',
           'predefined_cross_validation')

import pathlib

import numpy
import tables


ROOT = (pathlib.Path(__file__) / '../../data').resolve()

BASES = numpy.asarray([b'ACGT']).view('S1')

GROUPS = ['training', 'validation', 'testing']

REPLICA_COUNT = 6

SEQUENCE_LENGTH = 72


def path(id_):
    dirs = [p for p in ROOT.glob(f'{id_:02}*') if p.is_dir()]
    if not dirs:
        raise ValueError(f'no matching path: {id_:02}')
    if len(dirs) > 1:
        raise ValueError(f'ambiguous path matching: {id_:02}')
    return dirs[0]


def load(group, entry):
    try:
        group_id = GROUPS.index(group)
    except:
        raise RuntimeError(f'invalid group: {group}')
    with tables.open_file(path(0) / 'data.h5', 'r') as file:
        group_table = file.get_node('/', 'group').read()
        filter_ = group_table == group_id
        data_table = file.get_node('/', entry).read()
        return data_table[filter_, ...]


def load_sequence(group, one_hot=True):
    sequences = load(group, 'sequence')
    shape = sequences.shape
    data = sequences[..., None] == BASES
    if one_hot:
        return data.reshape(shape[0], shape[1] * BASES.size)
    return numpy.where(data.reshape(-1, BASES.size))[1].reshape(shape)


def average_fitness(array):
    return array.mean(axis=1)


def predefined_cross_validation():
    file = tables.open_file(path(1) / '00.cv.h5', 'r')
    folds = file.get_node('/', 'cv_train').read()
    for train_idx in folds:
        yield train_idx


def stacking_splits():
    file = tables.open_file(path(1) / '01.stack.h5', 'r')
    folds = file.get_node('/', 'stack_train').read()
    for train_idx in folds:
        yield train_idx
