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
        self.k = 0.5
        
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

    def forward(self, data):
        
#         if self.training:
        #     self.update_nodes(data) ## FORSE SI PUÒ FAR FUNNZIONARE USANDO data.func PER OTTENERE IL CODICE
        
        # Reduce embeddings
        if torch.isnan(data.x).any():
            print(f"=== NaN detected in data.x - BertGCNN forward before reduce emb")
        print("=== BertGCNN forward before reduce emb - data.x min/max: ", data.x.min().item(), data.x.max().item())

            
        data.x = self.reduce_embedding(data)
        
        if torch.isnan(data.x).any():
            print(f"=== NaN detected in data.x - BertGCNN forward after reduce emb")
        print("=== BertGCNN forward after reduce emb - data.x min/max: ", data.x.min().item(), data.x.max().item())
        
        if torch.isnan(data.edge_index).any():
            print(f"=== NaN detected in data.edge_index - BertGCNN forward")
        print("=== BertGCNN forward before reduce emb - data.edge_index min/max: ", data.edge_index.min().item(), data.edge_index.max().item())
        
        # Extract x, edge_index, and text
        x, edge_index, text = data.x, data.edge_index, data.func
        
#         print(f"=== data.edge_index in BertGCNN forward: {data.edge_index}")
#         print(f"=== data.func in BertGCNN forward: {text}")

        # Gated Graph Convolution

        x = self.ggnn(x, edge_index)
        if torch.isnan(x).any():
            print("=== x in BertGCNN forward - after ggnn layer - NaN found")
        print("=== in BertGCNN forward - after ggnn layer - x min/max: ", x.min().item(), x.max().item())

        x = self.conv(x, data.x)
        if torch.isnan(x).any():
            print("=== x in BertGCNN forward - after conv layer - NaN found")
        print("=== in BertGCNN forward - after conv layer - x min/max: ", x.min().item(), x.max().item())

        # Encode the input and get CodeBERT features
        input_ids, attention_mask = encode_input(text, self.tokenizer)
        cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))
        
        print("=== in BertGCNN forward x min/max: ", x.min().item(), x.max().item())
        print("=== in BertGCNN forward cls_logit min/max: ", cls_logit.min().item(), cls_logit.max().item())


        # Combine the predictions
        pred = (x + 1e-10) * self.k + cls_logit * (1 - self.k)
        print("=== in BertGCNN forward pred min/max: ", pred.min().item(), pred.max().item())

        # Ensure pred is strictly positive 
        pred = th.clamp(pred, min=1e-10)
        print("=== in BertGCNN forward pred after clamp min/max: ", pred.min().item(), pred.max().item())

        # Apply logarithm safely
        pred = th.log(pred)
        print("=== in BertGCNN forward pred after log min/max: ", pred.min().item(), pred.max().item())

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
            cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
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

