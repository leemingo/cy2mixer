import numpy as np
# In[3]:
import torch.nn as nn
import torch_geometric.transforms as T
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx
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
    