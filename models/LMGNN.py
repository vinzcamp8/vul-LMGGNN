import torch as th
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Conv, encode_input
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import numpy as np
class BertGGCN(nn.Module):
    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(BertGGCN, self).__init__()
        self.k = 0.1
        
        self.ggnn = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        self.nb_class = 2
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(device)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, self.nb_class).to(device)
        self.device = device
        # self.conv.apply(init_weights)
        # print('=== LMGNN.py - __init__:', self.bert_model.config.hidden_size, '===')
        # print('=== LMGNN.py - __init__:', self.feat_dim, '===')

    def forward(self, data):
        # the DataLoader format
        # DataBatch(x=[1640, 101], edge_index=[2, 933], y=[8], func=[8], batch=[1640], ptr=[9])
        
        # print('=== LMGNN.py - forward:', data, '===')

        # if self.training:
        #     self.update_nodes(data) ## FORSE SI PUÒ FAR FUNNZIONARE USANDO data.func PER OTTENERE IL CODICE

        data.x = self.reduce_embedding(data)
        # print('=== LMGNN.py - forward 2:', data, '===')

        x, edge_index, text = data.x, data.edge_index, data.func
        x = self.ggnn(x, edge_index)
        x = self.conv(x, data.x)

        input_ids, attention_mask = encode_input(text, self.tokenizer)
        cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))

        pred = (x + 1e-10) * self.k + cls_logit * (1 - self.k)
        pred = th.log(pred)

        return pred

    ## FORSE SI PUÒ FAR FUNNZIONARE USANDO data.func PER OTTENERE IL CODICE
    def update_nodes(self, data):
        print('===', type(data), '===')
        print('===', data, '===')
        print('===', type(data.x), '===')
        print('===', data.x, '===')
        print('===', type(data.x.size(0)), '===')
        print('===', data.x.size(0), '===')
        # for n_id, node in data.x.items():
        for n_id in range(data.x.size(0)):  # Iterate over the number of nodes
            node = data.x[n_id]  # Get the feature vector for the node
            # Get node's code
            node_code = node.get_code()
            tokenized_code = self.tokenizer(node_code, True)

            input_ids, attention_mask = encode_input(tokenized_code, self.tokenizer_bert)
            cls_feats = self.bert_model(input_ids.to("cuda"), attention_mask.to("cuda"))[0][:, 0]
            # cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]

            source_embedding = np.mean(cls_feats.cpu().detach().numpy(), 0)
            # The node representation is the concatenation of label and source embeddings
            embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
            # print(node.label, node.properties.properties.get("METHOD_FULL_NAME"))
            data.x = embedding
    
    def reduce_embedding(self, data):
        linear_layer = nn.Linear(data.x.size(1), 101).to(self.device)
        reduced_embedding = linear_layer(data.x)
        return reduced_embedding

    def save(self, path):
        print(path)
        torch.save(self.state_dict(), path)
        print("save!!!!!!")

    def load(self, path):
        self.load_state_dict(torch.load(path))

