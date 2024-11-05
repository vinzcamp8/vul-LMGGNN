import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from models.layers import Conv, encode_input
from torch_geometric.nn.conv import GatedGraphConv
from transformers import AutoModel, AutoTokenizer
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import numpy as np
from utils.functions import tokenizer

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
        
        # print(f"=== BertGCNN forward - data: {data}")
        # print(f"=== BertGCNN forward - data.x: {data.x}")
        # print(f"=== BertGCNN forward - data.edge_index: {data.edge_index}")
        # print(f"=== BertGCNN forward - data.y: {data.y}")
        # print(f"=== BertGCNN forward - data.func: {data.func}")
        # print(f"=== BertGCNN forward - data.batch: {data.batch}")
        # print(f"=== BertGCNN forward - data.ptr: {data.ptr}")
        

        if self.training:
            self.update_nodes(data) ## FORSE SI PUÒ FAR FUNNZIONARE 
        
        # Reduce embeddings            
        data.x = self.reduce_embedding(data)
        
        # Extract x, edge_index, and text
        x, edge_index, text = data.x, data.edge_index, data.func
        
#         print(f"=== data.edge_index in BertGCNN forward: {data.edge_index}")
#         print(f"=== data.func in BertGCNN forward: {text}")

        # Gated Graph Convolution
        x = self.ggnn(x, edge_index)
        x = self.conv(x, data.x)

        # Encode the input and get CodeBERT features
        input_ids, attention_mask = encode_input(text, self.tokenizer)
        cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
        cls_logit = self.classifier(cls_feats.to(self.device))


        # Combine the predictions
        pred = (x + 1e-10) * self.k + cls_logit * (1 - self.k)

        # Ensure pred is strictly positive 
        pred = th.clamp(pred, min=1e-10)

        # Apply logarithm safely
        pred = th.log(pred)

        return pred

    ## FORSE SI PUÒ FAR FUNNZIONARE USANDO data.func PER OTTENERE IL CODICE
    # provare a mettere il padding anche nei types a nei codes di ogni data loader
    # in questo modo le dimensioni sono fisse (205 nodi max)
    # e posso fare l'embedding updates in modo corretto in update_nodes
    # magari mettere un valore flag per indicare che il nodo è un padding
    # e non fare l'update_nodes ma lasciare il padding
    def update_nodes(self, data):
        embeddings = []
        i = 0
        print(f"=== UPDATE NODES data: {data}")

        
        for types_list, codes_list in zip(data.types, data.codes):
            for node_type, node_code in zip(types_list, codes_list):
                i = i + 1
                if node_type is None or node_code is None:
                    embedding = np.zeros(self.bert_model.config.hidden_size + 1)
                    embeddings.append(embedding)
                    continue
                try:
                    tokenized_code = tokenizer(node_code, True)
                    input_ids, attention_mask = encode_input(tokenized_code, self.tokenizer)

                    cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
                    source_embedding = np.mean(cls_feats.cpu().detach().numpy(), 0)
                    # The node representation is the concatenation of label and source embeddings
                    embedding = np.concatenate((np.array([node_type]), source_embedding), axis=0)
                    embeddings.append(embedding)

                    # Delete the tensors from GPU and free up memory
                    del input_ids, attention_mask, cls_feats
                    if th.cuda.is_available():
                        th.cuda.empty_cache()
                except Exception as e:
                    # Handle any exceptions and save the node only with its type
                    print(f"forward - Error processing node {i}: {e}")
                    print(f"Node code python type: {type(node_code)}")
                    print(f"Node code: {node_code}")
                    # ERRORE QUI DICE CHE NON È STR ANCHE SE LO È
                    # forward - Error processing node 1: text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
                    # Node code python type: <class 'str'>
                    # Node code: table_row_block
                    source_embedding = np.zeros(self.bert_model.config.hidden_size)
                    embedding = np.concatenate((np.array([node_type]), source_embedding), axis=0)
                    embeddings.append(embedding)
                    continue

        data.x = th.from_numpy(np.array(embeddings)).float()
    
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


