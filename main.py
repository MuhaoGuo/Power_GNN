import torch
import networkx as nx
from torch.nn import Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv, GraphConv, GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter
import math
import copy


class VCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, output_channels):
        super(VCN, self).__init__()
        torch.manual_seed(12345)
        self.conv0 = VmagConv(in_channels=dataset.num_node_features, out_channels=dataset.num_node_features) # in_chan = out_chan

        self.conv1 = VmagConv(dataset.num_node_features, dataset.num_node_features)

        self.conv2 = VmagConv(dataset.num_node_features, dataset.num_node_features)

        self.conv3 = VmagConv(dataset.num_node_features, dataset.num_node_features)

        # Layer to map voltage to VSI i.e., single value i.e., out feature for each node/ entire graph.
        self.lin = Linear(dataset.num_node_features, output_channels)

    def forward(self, x, edge_index, batch, edge_attr):
        # 1. Obtain node embeddings
        x = self.conv0(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

def train(model, train_loader):
     model.train()
     for data in train_loader:
         # data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
         out = model(data.x, data.edge_index, data.batch, data.edge_attr)  # Perform a single forward pass.
         loss = criterion(out, data.y.unsqueeze(1))  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

class VmagConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(VmagConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.lin = torch.nn.Linear(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    # TODO: Reset the parameters?
    def reset_parameters(self): # GCNConv
        self.glorot(self.weight)
        self.zeros(self.bias)

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, F_in]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, F_e]
        num_nodes = int(edge_index.max()) + 1

        # get the h2 and h3 features and save them before performing the convolution step.
        # TODO: This is the update value for h1 feature only. we should keep h2
        # and h3 constant according to the conv update equation we derived.
        self.x_original = copy.deepcopy(x)

        # TODO: NOTE this is matrix multiplication... this means it will change the meaning of eqs in propagate function.
        # Hence I will try the element wise multiplication to preserve the meaning for now but I have try both to see
        # which works.
        # TODO: I can now use this and apply linear or mlp on this stacked VmagConv layers to finally predict VSI.
        # Intuition is that PV curve and VSI share some commanlity and it is not so complex and hence the linear or mlp
        # would capture the VSI using the voltage update feature as input.
        # x = torch.matmul(x,self.weight) # matrix multiplication
        x = x*self.weight # element wise multiplication

        # The updated value of h0 after message-passing/fixed-point step i.e., voltage equivalent feature.
        out = self.propagate(edge_index, x, edge_attr, num_nodes)

        out += self.bias

        return out

    def propagate(self, edge_index, x, edge_attr, num_nodes):

        # Compute neighboring terms for each branch message
        neighbor_terms = self.message(edge_index, x, edge_attr)

        # Aggregate the neighboring terms for each node in the network
        aggregated_neighbor_terms = self.aggregate(neighbor_terms, edge_index[1], dim_size=num_nodes)

        # Updte the features of the each node with both "self" and "neighbor" terms. We will calculate self inside the
        # update function.
        out = self.update(num_nodes, x, aggregated_neighbor_terms)

        return out

    def aggregate(self, inputs, index, dim_size):
        return scatter(inputs, index, -1, dim_size=dim_size, reduce=self.aggr)

    def message(self, edge_index, x, edge_attr):
        # compute the neighbor node terms in the conv eq.
        row, col = edge_index[0], edge_index[1]
        # NOTE: Do not lift the output dimensions of VmagConv or VangConv because if we uplift the feature tensors
        # then we lose the meaning of the tensor. If we uplift, Maybe concat + other things may or may not work.
        # Basically we do not want matrix multiplication between different features to get uplifted features when we
        # multiply weights of neural nets. Then the next GNN layer loses the meaning completely :/
        neighbor_terms = edge_attr.view(1,-1).flatten() * x[col,0] * x[col,2]
        return neighbor_terms

    def update(self, num_nodes, x, aggregated_neighbor_terms):
        # compute the self node terms in the conv eq.
        self_node_ids = [node_id for node_id in range(num_nodes)]
        h2 = x[self_node_ids,1]
        h1h3 = x[self_node_ids,0]*x[self_node_ids,2]
        self_terms = h2/(h1h3)

        out = self_terms + aggregated_neighbor_terms
        # This is the update value for h1 feature only. we should keep h2
        # and h3 constant according to the conv update equation we derived.

        return out

if __name__ == "__main__":
    # Edge connectivities.
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    # Node feature information.
    x = torch.tensor([[-1, 2, 1], [-4, 3, -4], [1, 3, -1]], dtype=torch.float)

    # output label for the graph
    y = torch.tensor([[1]], dtype=torch.float)

    # Edge feature information.
    edge_at = torch.tensor([[22], [22], [33], [33]], dtype=torch.float)

    # make the data object for pygeo.
    data = Data(x=x, edge_index=edge_index, edge_attr= edge_at,  y=y)
    print("data", data)
    G = to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])
    print(f"edge info = {G.edges(data=True)}")
    print(f" node info = {G.nodes(data=True)}")
    # batch wise data object
    train_loader = DataLoader([data, data, data], batch_size=1, shuffle=True)

    # model of VCN
    model = VCN(data, hidden_channels=None, output_channels=1)
    # TODO: Why wont it show the dimensions in print? because it is not located in the nn folder of pygeo?
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.SmoothL1Loss()

    for epoch in range(1, 201):
        train(model, train_loader)
    k = 1


#_______________________________________________________________________________________________________________
        # non_self_edges_len = edge_weight.shape[0]
        #
        # # Add self loop
        # fill_value = 1 # TODO: This is fine but double check later.
        # edge_index, tmp_edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)
        # assert tmp_edge_weight is not None
        # edge_weight = tmp_edge_weight
        # self_edges_len = edge_weight.shape[0] - non_self_edges_len
        #
        # edge_weight[:non_self_edges_len] = t_1
        #
        # # Write code here to update edge_weight in "self_term" locations to have the custom self term from eq.
        # self_edge_node_ids = edge_index[non_self_edges_len:]
        # h2 = x[self_edge_node_ids,1]
        # h1h3 = x[self_edge_node_ids,0]*x[self_edge_node_ids,2]
        # self_terms = (edge_weight[non_self_edges_len:]*h2)/(h1h3)
        # edge_weight[non_self_edges_len:] = self_terms
        #
        # # propagate_type: (x: Tensor, edge_weight: OptTensor)
        # out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
        #                      size=None)

        # return out
        #_______________________________________________________________________________________________________________