import numpy as np
import torch
from torch_geometric.data import Data
from utils.functions import tokenizer
from utils.functions import log as logger
from gensim.models.keyedvectors import Word2VecKeyedVectors
from models.layers import encode_input
from transformers import RobertaTokenizer, RobertaModel

class NodesEmbedding:
    def __init__(self, nodes_dim: int):
        # self.w2v_keyed_vectors = w2v_keyed_vectors
        # self.kv_size = w2v_keyed_vectors.vector_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_bert = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base").to(self.device)
        # self.bert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.nodes_dim = nodes_dim

        assert self.nodes_dim >= 0

        # Buffer for embeddings with padding
        self.target = torch.zeros(self.nodes_dim, self.bert_model.config.hidden_size + 1).float()

    def __call__(self, nodes):
        embedded_nodes = self.embed_nodes(nodes)
        nodes_tensor = torch.from_numpy(embedded_nodes).float()

        self.target[:nodes_tensor.size(0), :] = nodes_tensor ### PROBLEMA QUI CON LE DIMENSIONI
# self.target[:nodes_tensor.size(0)] = nodes_tensor
# RuntimeError: expand(torch.FloatTensor{[72, 769]}, size=[72]): the number of sizes provided (1) must be greater or equal to the number of dimensions in the tensor (2)

        return self.target

    def embed_nodes(self, nodes):
        embeddings = []

        for n_id, node in nodes.items():
            try:
                # Get node's code
                node_code = node.get_code()
                tokenized_code = tokenizer(node_code, True)

                # Tokenize the code and handle potential errors
                input_ids, attention_mask = encode_input(tokenized_code, self.tokenizer_bert)
                if input_ids is None or len(input_ids) == 0:  # Check if the encoding was successful
                    print(f"Skipping node {n_id}: Tokenizer returned empty input_ids")
                    continue

                # Get embeddings using the BERT model
                cls_feats = self.bert_model(input_ids.to(self.device), attention_mask.to(self.device))[0][:, 0]
                source_embedding = np.mean(cls_feats.cpu().detach().numpy(), 0)

                # Concatenate the node type with the source embeddings
                embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
                embeddings.append(embedding)

                # Delete the tensors from GPU and free up memory
                del input_ids, attention_mask, cls_feats
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Optional: consider removing if it impacts performance

            except Exception as e:
                # Handle any exceptions and save the node only with its type
                print(f"Error processing node {n_id}: {e}")
                print(f"Node code: {node_code}")
                source_embedding = np.zeros(self.bert_model.config.hidden_size)
                embedding = np.concatenate((np.array([node.type]), source_embedding), axis=0)
                embeddings.append(embedding)
                continue
                
        return np.array(embeddings)


    # fromTokenToVectors
    # This is the original Word2Vec model usage.
    # Although we keep this part of the code, we are not using it.
    def get_vectors(self, tokenized_code, node):
        vectors = []

        for token in tokenized_code:
            if token in self.w2v_keyed_vectors.vocab:
                vectors.append(self.w2v_keyed_vectors[token])
            else:
                # print(node.label, token, node.get_code(), tokenized_code)
                vectors.append(np.zeros(self.kv_size))
                if node.label not in ["Identifier", "Literal", "MethodParameterIn", "MethodParameterOut"]:
                    msg = f"No vector for TOKEN {token} in {node.get_code()}."
                    logger.log_warning('embeddings', msg)

        return vectors


class GraphsEmbedding:
    def __init__(self, edge_type):
        self.edge_type = edge_type

    def __call__(self, nodes):
        connections = self.nodes_connectivity(nodes)

        return torch.tensor(connections).long()

    # nodesToGraphConnectivity
    def nodes_connectivity(self, nodes):
        # nodes are ordered by line and column
        coo = [[], []]
        
        count1 = 0
        count2 = 0

        for node_idx, (node_id, node) in enumerate(nodes.items()):
            if node_idx != node.order:
                raise Exception("Something wrong with the order")

            for e_id, edge in node.edges.items():
#                 print(f"=== Nodes connectivity - edge.type: {edge.type} self.edge_type:{self.edge_type}")
#                 if edge.type != self.edge_type:
#                     continue

                if edge.node_in in nodes and edge.node_in != node_id:
                    coo[0].append(nodes[edge.node_in].order)
                    coo[1].append(node_idx)
                    count1 += 1

                if edge.node_out in nodes and edge.node_out != node_id:
                    coo[0].append(node_idx)
                    coo[1].append(nodes[edge.node_out].order)
                    count2 += 1
                    
        print(f"Total edges: {count1} + {count2}")
            
        return coo


def nodes_to_input(nodes, target, nodes_dim, edge_type):
    nodes_embedding = NodesEmbedding(nodes_dim)
    graphs_embedding = GraphsEmbedding(edge_type)
    label = torch.tensor([target]).float()

    return Data(x=nodes_embedding(nodes), edge_index=graphs_embedding(nodes), y=label)