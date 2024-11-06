import numpy as np
# In[3]:
import torch.nn as nn
import torch_geometric.transforms as T
import torch
import networkx as nx
from torch_geometric.utils import degree, get_laplacian, to_scipy_sparse_matrix,to_undirected, to_dense_adj, scatter
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import degree


def make_cycle_adj_speed_nosl(raw_list_adj,data):
    
    #original_adj=np.array(list(g.get_adjacency()))
    original_adj = np.array(raw_list_adj)
    Xgraph = to_networkx(data,to_undirected= True)
    num_g_cycle=Xgraph.number_of_edges() - Xgraph.number_of_nodes() + nx.number_connected_components(Xgraph)
    node_each_cycle=nx.cycle_basis(Xgraph)
    if num_g_cycle >0 : 
  
        if len(node_each_cycle) != num_g_cycle:
            print('Error in the number of cycles in graph')
            print('local cycle',len(node_each_cycle), 'total cycle',num_g_cycle)
            
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=np.zeros((original_adj.shape[0],original_adj.shape[1]))
        for nodes in node_each_cycle:
            #start = time.time()
            #N_V=len(nodes)                
            for i in nodes:
                SUM_CYCLE_ADJ[i,nodes]=1   
            SUM_CYCLE_ADJ[nodes,nodes]=0
            #print('3. time',time.time()-start)    
    else:
        node_each_cycle=[]
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        CAL_SUB_ADJ=[]
        SUM_CYCLE_ADJ=[]
    return node_each_cycle, SUB_ADJ, RAW_SUB_ADJ, CAL_SUB_ADJ, SUM_CYCLE_ADJ

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def data_load(dataset, normalize=False,option=0):
    
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())
        print('max_degree',max_degree)
        
        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    elif normalize:
        if option < 1 :
            mean = torch.mean(dataset.data.x, axis=0)
            std = torch.std(dataset.data.x, axis=0)
            dataset.data.x -= mean
            dataset.data.x /= std
        elif option >0:
            mean = torch.mean(dataset.data.x, axis=0)
            std = torch.std(dataset.data.x, axis=0)
            mean[option:]=0
            std[option:]=1
            dataset.data.x -= mean
            dataset.data.x /= std
        
    return dataset



def max_node_dataset(dataset):
    aa=[]
    for i in range(len(dataset)):
        data=dataset[i]
        aa.append(data.num_nodes)

    max_node=np.max(aa) 
    return max_node


def make_NEWDATA(dataset,max_node):
        SUB_ADJ=[]
        RAW_SUB_ADJ=[]
        NEWDATA=[]
        # for i in range(len(dataset)):
        for i in range(1):
            data=dataset[i]
            v1=data.edge_index[0,:]
            v2=data.edge_index[1,:]
            #print(torch.max(v1))
            adj = torch.zeros((max_node,max_node))
            adj[v1,v2]=1
            adj=adj.numpy()
            (adj==adj.T).all()
            list_feature=(data.x)
            list_adj=(adj)       

            #print(dataset[i])
            _, _, _, _, sum_sub_adj = make_cycle_adj_speed_nosl(list_adj,data)

            # if i % 100 == 0:
            #     print(i)

            #_sub_adj=np.array(sub_adj)

            if len(sum_sub_adj)>0:    
                new_adj=np.stack((list_adj,sum_sub_adj),0)
            else :
                sum_sub_adj=np.zeros((1, list_adj.shape[0], list_adj.shape[1]))
                new_adj=np.concatenate((list_adj.reshape(1, list_adj.shape[0], list_adj.shape[1]),sum_sub_adj),0)

            #SUB_ADJ.append(new_adj)
            SUB_ADJ=new_adj
            data=dataset[i]
            check1=torch.sum(data.edge_index[0]-np.where(SUB_ADJ[0]==1)[0])+torch.sum(data.edge_index[1]-np.where(SUB_ADJ[0]==1)[1])
            if check1 != 0 :
                print('error')

            data.cycle_index=torch.stack((torch.LongTensor(np.where(SUB_ADJ[1]!=0)[0]), torch.LongTensor(np.where(SUB_ADJ[1]!=0)[1])),1).T.contiguous()
            #data.cycle_attr = torch.FloatTensor(SUB_ADJ[1][np.where(SUB_ADJ[1]!=0)[0],np.where(SUB_ADJ[1]!=0)[1]]) 
            NEWDATA.append(data)
        cycle_index = torch.stack((torch.LongTensor(np.where(SUB_ADJ[1]!=0)[0]), torch.LongTensor(np.where(SUB_ADJ[1]!=0)[1])),1).T.contiguous()
        # return NEWDATA
        return cycle_index

cfg = {
    'hid': 16,
    'pe_dim' :28,
    'rwse_dim' : 20,
    'emb_dim' : 36
}

def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs