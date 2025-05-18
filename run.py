
import time
import torch
import numpy as np
from train_eval import train, init_network,test
import wandb
from importlib import import_module
import argparse
#from memory_profiler import profile

parser = argparse.ArgumentParser(description='Encrypted Traffic Classification')
#parser.add_argument('--model', type=str, required=True, help='choose a model: TSCRNN, deeppacket, BiLSTM_Att, datanet')
parser.add_argument('--data', type=str, required=True, help='input dataset source')
#parser.add_argument('--test',  type=bool,default=False, required=True, help='True for Testing')
parser.add_argument('--test', type=bool, default=False, help='Train or test')

args = parser.parse_args()

def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}


def main():

    #dataset = "C:\\Users\\afif\\Documents\\Master\\Code\\benchmark_ntc\\Encrypted-Traffic-Classification-Models\\data\\ISCXVPN2016"
    dataset = "C:\\Users\\afif\\Documents\\Master\\Code\\benchmark_ntc\\Encrypted-Traffic-Classification-Models\\data\\" + args.data
    
    model_name = "Datanet"  
    from utils.utils_datanet import build_dataset, build_iterator, get_time_dif
    #model_name = args.model  
    # if 'deeppacket' in model_name:
    #     from utils.utils_deeppacket import build_dataset, build_iterator, get_time_dif
    # elif "BiLSTM" in model_name:
    #     from utils.utils_bilstm import build_dataset, build_iterator, get_time_dif
    # elif "TSCRNN" in model_name:
    #     from utils.utils_tscrnn import build_dataset, build_iterator, get_time_dif
    # elif "Datanet" in model_name:
    #     from utils.utils_datanet import build_dataset, build_iterator, get_time_dif
    # elif "MATEC" in model_name:
    #     from utils.utils_matec import build_dataset, build_iterator, get_time_dif
        
    x = import_module('models.' + model_name)

    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    print(len(train_data))
    train_iter = build_iterator(train_data, config)
    
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    pre_time_dif, average_pre_time = get_time_dif(start_time, test=0, data=args.data)
    print(f"Preprocess Time : {pre_time_dif:.10f} seconds")  # Show 6 decimal places
    print(f"Average prepocess time : {average_pre_time:.10f} seconds")

    
    model = x.Model(config).to(config.device)
    #init_network(model)
    
    if args.test == False:
        print(args.test)
        print(model.parameters)
        train(config, model, train_iter, dev_iter, test_iter, args.data)
    else:
        test(config, model, test_iter, data=args.data)
        wandb.log({"preprocess_time":  float(pre_time_dif)})
        wandb.log({"averagepreprocess_time":  float(average_pre_time)})
    
if __name__ == '__main__':
    main()