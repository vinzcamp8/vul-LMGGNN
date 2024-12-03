import json
import torch


class Config(object):
    def __init__(self, config, file_path="configs.json"):
        with open(file_path) as config_file:
            self._config = json.load(config_file)
            self._config = self._config.get(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_property(self, property_name, value):
        self._config[property_name] = value

    def get_property(self, property_name):
        return self._config.get(property_name)

    def get_device(self):
        return self.device

    def all(self):
        return self._config

    def update_from_args(self, args):
        if args.learning_rate is not None:
            self.set_property('learning_rate', args.learning_rate)
        if args.batch_size is not None:
            self.set_property('batch_size', args.batch_size)
        if args.epochs is not None:
            self.set_property('epochs', args.epochs)
        if args.weight_decay is not None:
            self.set_property('weight_decay', args.weight_decay)
        if args.patience is not None:
            self.set_property('patience', args.patience)
        if args.pred_lambda is not None:
            self.set_property('pred_lambda', args.pred_lambda)

class Create(Config):
    def __init__(self):
        super().__init__('create')

    @property
    def slice_size(self):
        return self.get_property('slice_size')

    @property
    def joern_cli_dir(self):
        return self.get_property('joern_cli_dir')


class Data(Config):
    def __init__(self, config):
        super().__init__(config)

    @property
    def cpg(self):
        return self.get_property('cpg')

    @property
    def raw(self):
        return self.get_property('raw')

    @property
    def input(self):
        return self.get_property('input')

    @property
    def input_w2v(self):
        return self.get_property('input_w2v')

    @property
    def model(self):
        return self.get_property('model')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')


class Paths(Data):
    def __init__(self):
        super().__init__('paths')

    @property
    def joern(self):
        return self.get_property('joern')


class Files(Data):
    def __init__(self):
        super().__init__('files')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')


class Embed(Config):
    def __init__(self):
        super().__init__('embed')

    @property
    def nodes_dim(self):
        return self.get_property('nodes_dim')

    @property
    def edge_type(self):
        return self.get_property('edge_type')
    
    @property
    def w2v_args(self):
        return self.get_property('word2vec_args')


class Process(Config):
    def __init__(self):
        super().__init__('process')

    @property
    def epochs(self):
        return self.get_property('epochs')

    @property
    def patience(self):
        return self.get_property('patience')

    @property
    def batch_size(self):
        return self.get_property('batch_size')

    @property
    def dataset_ratio(self):
        return self.get_property('dataset_ratio')

    @property
    def shuffle(self):
        return self.get_property('shuffle')


class BertGGNN(Config):
    def __init__(self):
        super().__init__('bertggnn')

    @property
    def learning_rate(self):
        return self.get_property('learning_rate')

    @property
    def weight_decay(self):
        return self.get_property('weight_decay')

    @property
    def pred_lambda(self):
        return self.get_property('pred_lambda')
    
    @property
    def w2v(self):
        return self.get_property('w2v')
    
    @property
    def model(self):
        return self.get_property('model')
