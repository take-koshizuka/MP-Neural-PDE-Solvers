import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_geometric.data import Data
from equations.PDEs import *
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm, GCNConv

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)

def L_out(L_in, padding, dilation, kernel_size, stride):
    return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1



class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + time_window + 1 + n_variables, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + n_variables, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class Linear_MP_PDE_Solver(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pde: PDE,
                 time_window: int = 25,
                 hidden_features: int = 128,
                 decoder_channels: int = 8,
                 eq_variables: dict = {}
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(Linear_MP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.time_window = time_window
        self.eq_variables = eq_variables
        self.decoder_channels = decoder_channels
        self.gnn_layers = GCNConv(in_channels=self.hidden_features,out_channels=self.hidden_features)

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + 2 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        if(self.time_window==20):
            Lout1 = L_out(self.hidden_features, 0, 1, 15, 4)
            Lout2 = L_out(Lout1, 0, 1, 10, 1)
            self.output_mlp = nn.Sequential(nn.Conv1d(1, self.decoder_channels, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(self.decoder_channels, 1, 10, stride=1),
                                            Swish(),
                                            nn.Linear(Lout2, self.time_window)
                                            )
        if (self.time_window == 25):
            Lout1 = L_out(self.hidden_features, 0, 1, 16, 3)
            Lout2 = L_out(Lout1, 0, 1, 14, 1)
            self.output_mlp = nn.Sequential(nn.Conv1d(1, self.decoder_channels, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(self.decoder_channels, 1, 14, stride=1),
                                            Swish(),
                                            nn.Linear(Lout2, self.time_window)
                                            )
            

            
        if(self.time_window==50):
            Lout1 = L_out(self.hidden_features, 0, 1, 12, 2)
            Lout2 = L_out(Lout1, 0, 1, 10, 1)
            self.output_mlp = nn.Sequential(nn.Conv1d(1, self.decoder_channels, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(self.decoder_channels, 1, 10, stride=1),
                                            Swish(),
                                            nn.Linear(Lout2, self.time_window)
                                            )

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_x = pos[:, 1][:, None] / self.pde.L
        pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        variables = pos_t    # time is treated as equation variable
        if "alpha" in self.eq_variables.keys():
            variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        if "beta" in self.eq_variables.keys():
            variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        if "gamma" in self.eq_variables.keys():
            variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        if "bc_left" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_left), -1)
        if "bc_right" in self.eq_variables.keys():
            variables = torch.cat((variables, data.bc_right), -1)
        if "c" in self.eq_variables.keys():
            variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)

        # Encoder and processor (message passing)
        node_input = torch.cat((u, pos_x, variables), -1)
        h0 = self.embedding_mlp(node_input)
        edge_weight = torch.tensor([ torch.norm(pos_x[i] - pos_x[j]) for [i,j] in edge_index.T ], device=edge_index.device)
        h = self.gnn_layers(h0, edge_index=edge_index, edge_weight=edge_weight)

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.hidden_features) * self.pde.dt).to(h.device)
        # dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        H = h0 + dt * h
        out = self.output_mlp(H[:, None]).squeeze(1)
        return out