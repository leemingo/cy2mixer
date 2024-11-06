import networkx as nx
import numpy as np
import torch
import pickle
import random
import os
import json
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.convert import to_networkx


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


def _make_cycle_adj_speed_nosl(original_adj, data):
        Xgraph = to_networkx(data, to_undirected=True)
        num_g_cycle = (
            Xgraph.number_of_edges()
            - Xgraph.number_of_nodes()
            + nx.number_connected_components(Xgraph)
        )
        node_each_cycle = nx.cycle_basis(Xgraph)

        if num_g_cycle > 0:
            if len(node_each_cycle) != num_g_cycle:
                raise ValueError(
                    f"Number of cycles mismatch: local {len(node_each_cycle)}, total {num_g_cycle}"
                )

            cycle_adj = np.zeros(original_adj.shape)
            for nodes in node_each_cycle:
                for i in nodes:
                    cycle_adj[i, nodes] = 1
                cycle_adj[nodes, nodes] = 0
        else:
            node_each_cycle, cycle_adj = [], []

        return node_each_cycle, cycle_adj


def make_cy2c(
    data, max_node, cy2c_self=False, cy2c_same_attr=False, cy2c_trans=False
):
    v1, v2 = data.edge_index
    list_adj = torch.zeros((max_node, max_node))
    list_adj[v1, v2] = 1
    # list_feature = data.x

    node_each_cycle, cycle_adj = _make_cycle_adj_speed_nosl(list_adj, data)

    if len(cycle_adj) > 0:
        stacked_adjs = np.stack((list_adj, cycle_adj), axis=0)
    else:
        cycle_adj = np.zeros((1, list_adj.shape[0], list_adj.shape[1]))
        stacked_adjs = np.concatenate((list_adj[np.newaxis], cycle_adj), axis=0)

    edge_index = data.edge_index
    check_num = torch.sum(
        edge_index[0] - np.where(stacked_adjs[0] == 1)[0]
    ) + torch.sum(edge_index[1] - np.where(stacked_adjs[0] == 1)[1])
    if check_num != 0:
        print("error")
        return False

    cycle_index = torch.stack(
        (
            torch.LongTensor(np.where(stacked_adjs[1] != 0)[0]),
            torch.LongTensor(np.where(stacked_adjs[1] != 0)[1]),
        ),
        dim=0,
    )

    if cy2c_self:
        cycle_index, _ = remove_self_loops(
            cycle_index
        )  # Remove if self loops already exist
        cycle_index, _ = add_self_loops(cycle_index)
        cycle_attr = torch.ones(cycle_index.shape[1]).long()
    else:
        cycle_attr = torch.ones(cycle_index.shape[1]).long()

    if cy2c_same_attr:
        pos_edge_attr = torch.ones(edge_index.shape[1]).long()
    else:
        pos_edge_attr = torch.zeros(edge_index.shape[1]).long()

    if cy2c_trans:
        cycle_index, _ = remove_self_loops(cycle_index)
        old_length = cycle_index.shape[1]
        cycle_index, _ = add_self_loops(cycle_index)
        new_length = cycle_index.shape[1]
        cycle_attr = torch.ones(new_length).long()
        if new_length > old_length:
            cycle_attr[-(new_length - old_length) :] = 0

    return cycle_index, cycle_attr, pos_edge_attr, node_each_cycle
