class Experiment:
    def __init__(self, config_dict):
        self.name = config_dict["name"]
        self.small = config_dict["small"]
        self.gpus = config_dict["gpus"]
        self.gamma = config_dict["gamma"]
        self.batch_size = config_dict["batch_size"]
        self.mixed_precision = config_dict["mixed_precision"]
        self.max_seq_len = config_dict["max_seq_len"]
        self.add_normalisation = config_dict["add_normalisation"]
        self.alternate_corr = config_dict["alternate_corr"]
        self.dataset_folder = config_dict["dataset_folder"]        
        self.restore_ckpt = config_dict["restore_ckpt"]
        self.train_or_test = config_dict["train_or_test"]
        self.model = config_dict["model"]
        self.dropout = config_dict["dropout"]
        self.dataset = config_dict["dataset"]
        if (config_dict['train_or_test'] == 'train'):
            self.stage = config_dict["stage"]
            self.validation = config_dict["validation"]
            self.lr = config_dict["lr"]
            self.num_steps = config_dict["num_steps"]
            self.iters = config_dict["iters"]
            self.wdecay = config_dict["wdecay"]
            self.epsilon = config_dict["epsilon"]
            self.clip = config_dict["clip"]
            self.dropout = config_dict["dropout"]
            self.add_noise = config_dict["add_noise"]
            self.beta_photo = config_dict["beta_photo"]
            self.beta_spatial = config_dict["beta_spatial"]
            self.beta_temporal = config_dict["beta_temporal"]
            
            
        elif (config_dict['train_or_test'] == 'test'):
            self.output_file =  config_dict["output_file"]
            self.iters = config_dict["iters"]
            
