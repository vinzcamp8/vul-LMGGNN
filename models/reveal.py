import torch.nn as nn
import torch

from torch_geometric.nn.conv import GatedGraphConv
from torch_geometric.data import Batch
from models.utils import GlobalAddPool

my_mlp_hidden_dim = 256
my_out_channels = 200

# Reveal model
class Reveal(nn.Module):
    def __init__(self, num_layers=1, MLP_hidden_dim=my_mlp_hidden_dim, need_node_emb=False):
        super().__init__()
        MLP_internal_dim = int(MLP_hidden_dim / 2)
        input_dim = my_out_channels 
        self.input_size = input_dim
        self.hidden_dim = input_dim
        # GGNN
        self.GGNN = GatedGraphConv(out_channels=input_dim, num_layers=5)
        self.readout = GlobalAddPool()
        self.dropout_p = 0.2
        self.need_node_emb = need_node_emb

        # MLP
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=MLP_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features=MLP_hidden_dim, out_features=MLP_internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=MLP_internal_dim, out_features=MLP_hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        ) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=MLP_hidden_dim, out_features=2),
        )

        # Linear layer to reduce the embedding size
        self.linear_layer = nn.Linear(769, 200)

    @property
    def key_layer(self):
        return self.GGNN

    def extract_feature(self, x):
        out = self.layer1(x)
        for layer in self.feature:
            out = layer(out)
        return out

    def embed_graph(self, x, edge_index, batch):
        node_emb = self.GGNN(x, edge_index)
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        graph_emb = self.readout(node_emb, batch)  # [batch_size, embedding_dim]
        return graph_emb, node_emb

    def arguments_read(self, *args, **kwargs):
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=x.device)
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=args[0].device)
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        return x, edge_index, batch


    def forward(self, *args, **kwargs):
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        if x.size(1) > 200:
            x = self.reduce_embedding(x)
        graph_emb, node_emb = self.embed_graph(x, edge_index, batch)
        feature_emb = self.extract_feature(graph_emb)
        probs = self.classifier(feature_emb) # [batch_size, 2]

        if self.need_node_emb:
            return probs, node_emb
        else:
            return probs
        
    def reduce_embedding(self, x):
        reduced_embedding = self.linear_layer(x)
        return reduced_embedding