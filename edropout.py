from __future__ import print_function
import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR,ExponentialLR,CosineAnnealingLR
import time,os,sys
import numpy as np
import copy
from torchsummary import summary as summary2
import torchvision
from numpy import unique
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import distance

sys.path.insert(0, '..') # I know its a bad practice :(
from utils import dataloader,hsummary #import utils.dataloader 
import numpy as np
import nets.realresnet as realresnet

# from joblib import Parallel, delayed

def selection(population,new_population,cost_population,cost_newpopulation,popsize,dimsize):
    # Population
    cost_population = np.reshape(np.asarray(cost_population),(popsize,))
    cost_newpopulation = np.reshape(np.asarray(cost_newpopulation),(popsize,))

    population_to_go = np.zeros((popsize,dimsize),dtype='float64')

    # mask = (tf.math.less_equal(cost_population,cost_newpopulation))
    mask = cost_population<=cost_newpopulation
    mask_pop_cost = np.where(mask==True)[0]
    mask_new_pop_cost = np.where((np.logical_not(mask))==True)[0]

    population_to_go[mask_pop_cost,:]=population[mask_pop_cost,:]
    population_to_go[mask_new_pop_cost,:]=new_population[mask_new_pop_cost,:]

    # Cost 
    costs = np.zeros((popsize),dtype='float64')
    costs[mask_pop_cost]=cost_population[mask_pop_cost]
    costs[mask_new_pop_cost]=cost_newpopulation[mask_new_pop_cost]

    # Bests  
    best_solution = np.asarray(population_to_go[np.argmin(costs),:])
    best_cost=np.asarray(np.amin(costs))
    generation_cost=np.asarray(np.mean(costs))
    # print('costpop',cost_population)
    # print('costnewpop',cost_newpopulation)
    # print('costtogo',costs)
    return population_to_go, costs, best_solution, best_cost, generation_cost


def test(args, model, device, d_loader, criterion,isvalid):
    # total = 0 
    test_loss = 0
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    n_test = len(d_loader) # ignoring last batch
    with torch.no_grad():
        for data, target in d_loader:
            data, target = data.to(device), target.to(device)
            output, _, _ = model(data)
            #### with cross entropy
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            # total += target.size(0)
            correct += (predicted.eq(target).sum().item()/target.size(0)) # last batch might be smaller - so use batch size

            # top3
            _, predicted_args_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None) 
            vtarget = target.view(list(target.size())[0],1)
            target3 = vtarget.repeat(1,3)
            correct_top3 += (predicted_args_top3.eq(target3).sum().item()/target.size(0)) # last batch might be smaller - so use batch size
            # top5
            _, predicted_args_top5 = torch.topk(output, 5, dim=1, largest=True, sorted=True, out=None) 
            target5 = vtarget.repeat(1,5)
            correct_top5 += (predicted_args_top5.eq(target5).sum().item()/target.size(0)) # last batch might be smaller - so use batch size

    #### with cross entropy
    test_loss /= n_test
    accu = 100.*correct/n_test
    accu3 = 100.*correct_top3/n_test
    accu5 = 100.*correct_top5/n_test
    if isvalid:
        print('Validation set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%)'.format(test_loss, accu, accu3, accu5))
    else:
        print('Test set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%)'.format(test_loss, accu, accu3, accu5))
    print(20*'#')
    return test_loss, accu, accu3, accu5


def test_c(args, model, network_size, device, d_loader, states_reshaped,loss_func,isvalid):
    criterion = nn.CrossEntropyLoss()

    # backup_weights = weight_backer(model,network_size,)
    # # Sparsifiy the network
    sparsifier(model,network_size, states_reshaped)

    test_loss = 0
    energy_valid = 0
    total = 0
    correct = 0
    correct_top3 = 0
    correct_top5 = 0
    n_test = len(d_loader) #(len(d_loader.dataset)-len(d_loader[-1])) # ignoring last batch
    print('ntest',n_test)
    counter = 0
    # print(n_test_batch)
    with torch.no_grad():
        for data, target in d_loader:
            data, target = data.to(device), target.to(device)
            # Get network Energy with applied states
            output, valid_eng, _  = model(data) 

            if loss_func=='ce':
                #### with cross entropy
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = output.max(1)
                correct += (predicted.eq(target).sum().item()/target.size(0)) # last batch might be smaller - so use batch size
            elif loss_func=='log':
                ### With log
                output = F.log_softmax(output)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                predicted = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += (predicted.eq(target.view_as(predicted)).sum().item()/target.size(0))

            # top3
            _, predicted_args_top3 = torch.topk(output, 3, dim=1, largest=True, sorted=True, out=None) 
            vtarget = target.view(list(target.size())[0],1)
            target3 = vtarget.repeat(1,3)
            correct_top3 += (predicted_args_top3.eq(target3).sum().item()/target.size(0)) # last batch might be smaller - so use batch size
            # top5
            _, predicted_args_top5 = torch.topk(output, 5, dim=1, largest=True, sorted=True, out=None) 
            target5 = vtarget.repeat(1,5)
            correct_top5 += (predicted_args_top5.eq(target5).sum().item()/target.size(0)) # last batch might be smaller - so use batch size


            one_hot = torch.zeros(list(target.size())[0],n_classes).to(device)
            one_hot[torch.arange(list(target.size())[0]),target] = 1
            high_cost_target_one_hot = -10000*one_hot # high cost for target to remove it   [0 0 0 -10000 0 0] 


            net_eneg = torch.sum(torch.mul(one_hot,valid_eng),axis=1)
            other_net_eneg = high_cost_target_one_hot + valid_eng # mean over batches
            max_other_net_eneg = torch.max(other_net_eneg,1)

            eng_diff = net_eneg - max_other_net_eneg[0]
            energy_valid += (-1*eng_diff.mean(0))



    #### with cross entropy
    test_loss /= n_test
    accu = 100.*correct/n_test
    accu3 = 100.*correct_top3/n_test
    accu5 = 100.*correct_top5/n_test
    energy_valid = energy_valid/n_test
    if isvalid:
        print('Validation set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%,Evalid: ({:.2f})'.format(test_loss, accu, accu3, accu5, energy_valid))
    else:
        print('Test set: Average loss: {:.4f}, AccuracyT1: ({:.2f}%), AccuracyT3: ({:.2f}%),AccuracyT5: ({:.2f}%,Evalid: ({:.2f})'.format(test_loss, accu, accu3, accu5, energy_valid))
    print(20*'#')
    return test_loss, accu, accu3, accu5, energy_valid
  
 
  
def state_reshaper(device,NS,states,network_size,args):
    states_reshaped = {}
    for state_indx in range(NS):
        ## State reshpaer from vector to mask
        states_reshaped[str(state_indx)]={}
        st = 0 # state counter
        ## For Convolution Layers
        # base_key = 'Conv2d'
        for indx, val in enumerate(network_size[0]):
            # print(indx,val)
            con = states[state_indx, st:st+val[2][0]]
            # print(con,con.shape)  
            tmp = np.zeros((val[2]))
            for ii in range(len(con)):
                tmp[ii,:,:,:] = con[ii]
            states_reshaped[str(state_indx)][val[0]] = torch.from_numpy(tmp).float().to(device)
            # print(tmp.shape)
            # con = states[state_indx, st:st+network_size[base_key][indx][0]]
            # con = np.reshape(con,(network_size[base_key][indx][0],1,1))
            # con = np.tile(con,(args.batch_size,1,network_size[base_key][indx][1],network_size[base_key][indx][1]))
            st+=val[2][0]
            # key = base_key+'-'+str(indx+1)
            # states_reshaped[str(state_indx)][key]=torch.from_numpy(con).float().to(device)
        # base_key = 'fc'
        for indx, val in enumerate(network_size[1]):
            # print(indx,val,val[2][1])
            con = states[state_indx, st:st+val[2][1]]
            # print('this',con,con.shape)  
            tmp = np.zeros((val[2]))
            for ii in range(len(con)):
                tmp[:,ii] = con[ii]
            states_reshaped[str(state_indx)][val[0]] = torch.from_numpy(tmp).float().to(device)
            # con = states[state_indx, st:st+network_size[base_key][indx][0]]
            # con = np.reshape(con,(1,network_size[base_key][indx][0]))
            # con = np.tile(con,(args.batch_size,1)) # TODO: match shpae to layer shape
            st+=val[2][1]
            # key = base_key+'-'+str(indx+1)
            # states_reshaped[str(state_indx)][key]=torch.from_numpy(tmp).float().to(device)
        # print(states_reshaped[str(state_indx)][val[0]])
        # print(con)
    return states_reshaped

def evolution(states,NS,D,best_state):
    ### Evolve States
    # print('inpop',np.sum(states,axis=1))
    population = copy.deepcopy(states)
    candidates_indx = np.tile(np.arange(NS),(NS,1))
    candidates_indx = (np.reshape((candidates_indx[(candidates_indx!=np.reshape(np.arange(NS),(NS,1)))]),(NS,NS-1)))
    for i in range(NS):
        candidates_indx[i,:]=np.random.permutation(candidates_indx[i,:])
    parents = candidates_indx[:,:3]

    x1 = population[parents[:,0]]
    # x1 = best_state[0] # best rule

    x2 = population[parents[:,1]]
    x3 = population[parents[:,2]]


    Ff = 0.9
    F_mask = (np.random.rand(NS,D)>Ff) # smaller F more diversity 
    keeps_mask = (np.abs(1*x2-1*x3)==0)*F_mask #np.logical_not(F_mask) # genes to keep  - not of F-mask to keep
    ev = np.multiply(np.logical_not(F_mask),x1) + np.multiply(keeps_mask,(1-x1))
    
    crossover_rate = 0.9                # Recombination rate [0,1] - larger more diverisyt
    cr = (np.random.rand(NS,D)<crossover_rate)
    mut_keep = np.multiply(ev,cr)
    pop_keep = np.multiply(population,np.logical_not(cr))
    new_population = mut_keep + pop_keep


    return new_population


def ising_cost(states,NS,D, D_Conv2d,D_fc,entrop_sig,fcbias_sig,states_loss, interaction_sigma,network_size):
    cost_population = np.zeros((NS,1))
    #### Compute final energy
    cost_breakdown= np.zeros((NS,3))
    for state_indx in range(NS):

        const = states_loss[state_indx,0] # network energy loss

        cost_breakdown[state_indx,0] = 0 #-interaction
        cost_breakdown[state_indx,1] = 0 #-bias
        cost_breakdown[state_indx,2] = -10*const

        cost_population[state_indx,0] = np.sum(cost_breakdown[state_indx,2]) #-10*interaction-bias-100*const  # -interaction means encourage non-similar featuremaps

    # print('Cost breakdown:', np.mean(cost_breakdown,axis=0))
    cost_all_one = None
    return cost_population, cost_all_one


def weight_backer(model, network_size):
    backup_weights = {}
    # backup_weights['conv1'] = model._modules['conv1'].weight.data.clone()
    backup_weights['fc'] = model._modules['fc'].weight.data.clone()
    for lay in network_size[0][1:]: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
        lay_name = lay[0] # such as layer3.1.conv1
        layer_name = lay_name.split('.')[0]
        layer_indx = int(lay_name.split('.')[1])
        conv_name = lay_name.split('.')[2]
        backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.clone()         
    #     model._modules['layer1'][indx]._modules['conv1'].weight.data = torch.zeros(sh)
    return backup_weights
    

def weight_restore(model, network_size, backup_weights):
    with torch.no_grad():
        model._modules['fc'].weight.data.copy_(backup_weights['fc']) 
        for lay in network_size[0][1:]: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
            lay_name = lay[0] # such as layer3.1.conv1
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.copy_(backup_weights[lay_name]) 


def sparsifier(model,network_size, state_in):
    with torch.no_grad():
        # model._modules['conv1'].weight.data *= state_in['conv1']
        model._modules['fc'].weight.data *= state_in['fc']
        for lay in network_size[0][1:]: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
            # print(lay[0])
            lay_name = lay[0] # such as layer3.1.conv1
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            model._modules[layer_name][layer_indx]._modules[conv_name].weight.data *=state_in[lay_name]



def pnetwork_sensory(args,model,data,target,device,NS,stopcounter, states_reshaped,network_size,D,D_Conv2d, D_fc,n_classes,states):
    count_pop_states = np.sum(states,axis=1)
    states_loss = np.zeros((NS,1))
    backup_weights = weight_backer(model,network_size)
    ## One hotting target and allocating very low energy to target energy    
    one_hot = torch.zeros(args.batch_size,n_classes).to(device)
    one_hot[torch.arange(args.batch_size),target] = 1
    high_cost_target_one_hot = -10000*one_hot # high cost for target to remove it   [0 0 0 -10000 0 0] 

    ## Compute Network Loss
    ## Feed reshpaed state to the model for evaluation
    for state_indx in range(NS):
        # print(state_indx)
        sparsifier(model,network_size, states_reshaped[str(state_indx)])
        ## Get network Energy with applied states
        with torch.no_grad():
            o1, net_energy, net_signals  = model(data) 
        ## Restore network weights from backup
        weight_restore(model, network_size, backup_weights)

        ## get energy of other classes than target
        net_eneg = torch.sum(torch.mul(one_hot,net_energy),axis=1)
        other_net_eneg = high_cost_target_one_hot + net_energy # mean over batches
        max_other_net_eneg = torch.max(other_net_eneg,1)

        eng_diff = net_eneg - max_other_net_eneg[0]
        states_loss[state_indx,0] = (1*eng_diff.mean(0)) # + count_pop_states[state_indx] 

    interaction_sigma = None
    fcbias_sig = None
    entrop_sig = None
    return entrop_sig,fcbias_sig,states_loss,interaction_sigma,backup_weights


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_select(nnmodel,device,s_input_channel,n_classes):
    if nnmodel=='alexnet':
        model = alexnet.NetOrg(s_input_channel,n_classes).to(device)
    elif nnmodel=='lenet':
        model = lenet.NetOrg(s_input_channel,n_classes).to(device)    
    elif nnmodel=='googlenet':
        model = googlenet.NetOrg(s_input_channel,n_classes).to(device)       
        network_size = googlenet.networksize()
    elif nnmodel == 'resnext50':
        resnext_type = 'resnext50'
        model = resnext.ResNext(s_input_channel,n_classes,resnext_type).to(device) 
    elif nnmodel == 'resnet18':
        resnet_type = 'resnet18'
        model = realresnet.resnet18(s_input_channel,n_classes).to(device) 
    elif nnmodel == 'resnet34':
        resnet_type = 'resnet34'
        model = realresnet.resnet34(s_input_channel,n_classes).to(device) 
    elif nnmodel == 'resnet50':
        resnet_type = 'resnet50'
        model = realresnet.resnet50(s_input_channel,n_classes).to(device)         
    elif nnmodel == 'resnet101':
        resnet_type = 'resnet101'
        model = realresnet.resnet101(s_input_channel,n_classes).to(device)         
    elif nnmodel == 'resnet152':
        resnet_type = 'resnet152'
        model = realresnet.resnet152(s_input_channel,n_classes).to(device)         
    elif nnmodel == 'resnext50_32x4d':
        resnet_type = 'resnext50_32x4d'
        model = realresnet.resnext50_32x4d(s_input_channel,n_classes).to(device) 
    return model


def gradient_masker(model,best_stateI,nnmodel,device):
    if nnmodel=='alexnet':
        alexnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel=='lenet':
        lenet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel=='googlenet':
        googlenet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnext50':
        resnext50.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet20':
        resnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet18':
        realresnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet50':
        realresnet.gradient_mask(model,best_stateI,device,nnmodel)

def key_maker(model):
    # print(kk['linear.bias'].shape) # get size of weights
    # print(kk['linear.bias']) # call weights by name
    kk = model.state_dict()
    keys = [i for i in model.state_dict().keys()]
    # print(keys)
    # for indx, p in enumerate(model.parameters()):
    #     print(indx,p.shape)
    convs_list = []
    linear_list = []
    D_Conv2d = 0
    D_fc = 0
    for i in keys:
        if 'conv2' in i:
            name = '.'.join(i.split('.')[:-1])
            attr = i.split('.')[-1]
            li=[name,attr,kk[i].shape]
            convs_list.append(li)
            D_Conv2d+=kk[i].shape[0]
        elif 'fc' in i:
            name = ''.join(i.split('.')[:-1])
            attr = i.split('.')[-1]
            if attr == 'weight':
                li=[name,attr,kk[i].shape]
                linear_list.append(li)   
                D_fc+= kk[i].shape[1]
    D_fc+=n_classes
    D = D_Conv2d+D_fc #+n_classes
    network_size = [convs_list,linear_list]
    return network_size, D, D_Conv2d, D_fc


def kept_counter(network_size,final_state):
    # print(network_size)
    counter = 0
    total_conv = 0
    kept_conv = 0
    for indx, con in enumerate(network_size[0]):
        k_s = con[2][2] # kernel size
        kept_conv+= np.sum(final_state[0,counter:counter+con[2][0]])*k_s*k_s # kept parameters in current conv
        total_conv+= con[2][0]*k_s*k_s # total parameters in current conv
        counter += con[2][0]

    total_fc = network_size[1][0][2][1]*n_classes
    # print(final_state.shape)
    # print(final_state[0,counter:-n_classes])
    kept_fc = np.sum(final_state[0,counter:-n_classes])*n_classes
    kp = (kept_fc+kept_conv)/float(total_fc+total_conv)
    total_p = total_fc+total_conv
    print('Total parameters: ',total_p)
    print('Kept Conv:',kept_conv,'/',total_conv)
    print('Kept Fc:',kept_fc,'/',total_fc)
    print('Kept Percentile:', kp)
    
    return total_p,kept_conv,total_conv,kept_fc,total_fc, kp

def weights_initialization(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.xavier_uniform_(m.bias)

def energyloss(net_energy,target,n_classes,args,device):
    one_hot = torch.zeros(args.batch_size,n_classes).to(device)
    one_hot[torch.arange(args.batch_size),target] = 1
    high_cost_target_one_hot = -10000*one_hot # high cost for target to remove it   [0 0 0 -10000 0 0] 
    net_eneg = torch.sum(torch.mul(one_hot,net_energy),axis=1)
    other_net_eneg = high_cost_target_one_hot + net_energy # mean over batches
    max_other_net_eneg = torch.max(other_net_eneg,1)

    eng_diff = net_eneg - max_other_net_eneg[0]
    elloss = (-1*eng_diff.mean(0))
    return elloss

def train_ising(args,pretrained_name,pretrained_name_save,stopcounter, input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func):

    ### Pytorch Setup
    torch.manual_seed(args.seed)
    torch.cuda.is_available()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"  
    device = torch.device(dev) 
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    model = model_select(nnmodel,device,s_input_channel,n_classes)

    network_size, D, D_Conv2d, D_fc = key_maker(model)

    ### Load data
    train_loader, valid_loader = dataloader.traindata(kwargs, args, input_size, valid_percentage, dataset)
    ### Load pre-trained weights
    if args.pre_trained_f:
        pretrained_weights = torch.load(pretrained_name)
        model.load_state_dict(pretrained_weights)

    ### Optimizer
    if args.optimizer_m == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0.00001)
    elif args.optimizer_m == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.000125,momentum=0.9,nesterov=True)
    if args.scheduler_m == 'StepLR':
        scheduler = StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.gamma,last_epoch=-1)

    ## Result collection
    train_loss_collect = []
    train_accuracy_collect = []
    valid_loss_collect = []
    valid_accuracy_collect = []
    valid_accuracy_collect3 = []
    valid_accuracy_collect5 = []

    NS = args.NS
    states_rand = np.asarray(np.random.randint(0,2,(NS,D)),dtype='float64') # Population is 0/1
    states_init = np.asarray(np.ones((NS,D)),dtype='float64') # Population is 0/1


    states_init = states_rand
    n_batches = len(train_loader)
    states_rand_one = np.asarray(np.random.randint(0,2,(1,D)),dtype='float64') # Population is 0/1
    states_rand_one0 = states_rand_one[0,:]

    
    ## Starting ....
    state_converged = False
    epoch_best_so_far = []
    collect_cost_best_state = [] # large value of best cost for early masking
    collect_avg_state_cost = []
    kp_collect = []
    lr_collect = []
    valid_energy_collect = []
    best_valid_loss = 10e10

    ### Logs
    print(10*'#')
    print('Dimension of state vector: ',D)
    print('Pop size is: ',NS)
    print('Optimizer is: '+str(args.optimizer_m)+' and Scheduler is '+str(args.scheduler_m))
    print('Model: ',nnmodel,'  Device: ',device, '  Input channels: ',s_input_channel,'  N Classes: ',n_classes)
    print(10*'#')
    print('Starting training...')
    print(10*'#')

    for epoch in range(1, args.epochs + 1):
        counter = 0
        epoch_accuraacy = 0
        epoch_loss = 0


        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)


            if epoch>=threshold_early_state and batch_idx>10: 
                if mode == 'random':
                    best_state = states_rand_one0
                    avg_cost_states = 0
                    cost_best_state = 0
                    backup_weights = weight_backer(model,network_size)
                    state_converged = True
                    states = states_rand_one
                elif mode == 'ising':
                    ## Early state convergence
                    if np.mean(collect_cost_best_state[-10:-1]==collect_cost_best_state[-1]):
                        state_converged = True

            # Evolution State
            # state_converged = True
            if state_converged==False: # apply early masking
                if batch_idx % 1 == 0:
                    # print(batch_idx)
                    with torch.no_grad():
                        if batch_idx==0 and epoch==1:
                            states_reshaped = state_reshaper(device,NS,states_init,network_size,args)
                            entrop_sig, fcbias_sig, states_loss, weigts_mean_kerneled, backup_weights = pnetwork_sensory(args,model,data,target,device,NS,stopcounter,states_reshaped,network_size,D,D_Conv2d, D_fc,n_classes,states_init)
                            cost_states, cost_all_one = ising_cost(states_init,NS,D, D_Conv2d, D_fc,entrop_sig, fcbias_sig, states_loss, weigts_mean_kerneled,network_size)            
                            states = states_init
                            best_state = np.asarray(states[np.argmin(cost_states),:])

                            # print('init',cost_states)
                        new_states = evolution(states,NS,D,best_state)
                        states_reshaped = state_reshaper(device,NS,new_states,network_size,args)
                        entrop_sig, fcbias_sig, states_loss, weigts_mean_kerneled, backup_weights = pnetwork_sensory(args,model,data,target,device,NS,stopcounter,states_reshaped,network_size,D,D_Conv2d, D_fc,n_classes,new_states)
                        cost_new_states, cost_all_one = ising_cost(states,NS,D, D_Conv2d, D_fc,entrop_sig, fcbias_sig, states_loss, weigts_mean_kerneled,network_size)            
                        states, cost_states, best_state, cost_best_state, avg_cost_states = selection(states, new_states, cost_states, cost_new_states, NS, D)

            ## Collect evolution cost
            collect_avg_state_cost.append(avg_cost_states)
            collect_cost_best_state.append(cost_best_state)

            # Train procedure - Train for the best state
            # best_state = states_rand_one
            best_stateI = best_state
            optimizer.zero_grad()
            best_state2 = np.expand_dims(best_stateI,axis=0)
            best_state_reshaped = state_reshaper(device,1,best_state2,network_size,args)
            # Backup weights
            # backup_weights = weight_backer(model,network_size,)
            # Sparsifiy the network

            sparsifier(model,network_size, best_state_reshaped[str(0)])
            # Get network Energy with applied states
            # with torch.no_grad():
            output, _, _  = model(data) 


            if loss_func=='ce':
                #### with cross entropy
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, target)
                _, pred = output.max(1)
                correct = pred.eq(target.view_as(pred)).sum().item()
            elif loss_func=='log':
                ### With log
                output = F.log_softmax(output)
                loss = F.nll_loss(output, target, reduction='sum')  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()

            acc = 100.*(correct/np.float(args.batch_size))


            epoch_loss+=loss.item()
            epoch_accuraacy+=acc
            if batch_idx % args.log_interval == 0:
                print(nnmodel,dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                print('Accuracy: ', acc)
                print('Epoch: ',epoch)
                print('Population min cost: ',cost_best_state)
                print('Population avg cost: ',avg_cost_states)
                print('Learning rate: ',scheduler.get_lr()[0])
                print(np.sum(best_state),len(best_state))
                print('state',state_converged)
                print(10*'*')


            loss.backward() # computes gradients for ones with reques_grad=True
            gradient_masker(model,best_stateI,nnmodel,device)
            weight_restore(model, network_size, backup_weights)

            optimizer.step()
            optimizer.zero_grad() ######TODOm

            # Break
            counter+=1
            if counter==stopcounter or counter==(n_batches-1):
                epoch_loss = epoch_loss/counter
                epoch_accuracy = epoch_accuraacy/counter
                states_reshaped = state_reshaper(device,NS,states,network_size,args)

                break


        # Validation
        print('Validation at epoch %d is:'%(epoch))
        best_state3 = np.expand_dims(best_state,axis=0)
        best_state_reshaped = state_reshaper(device,1,best_state3,network_size,args)
        valid_loss, valid_accuracy,valid_accuracy3,valid_accuracy5, valid_energy = test_c(args, model, network_size, device, valid_loader, best_state_reshaped[str(0)],loss_func, isvalid=True)
        _,_,_,_,_, kp_valid = kept_counter(network_size, best_state3)

        scheduler.step()

        ## Result collection
        train_loss_collect.append(epoch_loss)
        train_accuracy_collect.append(epoch_accuracy)
        valid_loss_collect.append(valid_loss)
        valid_accuracy_collect.append(valid_accuracy)
        valid_accuracy_collect3.append(valid_accuracy3)
        valid_accuracy_collect5.append(valid_accuracy5)
        kp_collect.append(kp_valid)
        lr_collect.append(scheduler.get_lr()[0])
        epoch_best_so_far.append(cost_best_state) # cost best state at current epoch
        valid_energy_collect.append(valid_energy.data.cpu().numpy())
        ## Early Stopping weight save
        # if epoch>1 and scheduler.get_lr()[0]<0.0001:
        # if epoch>50: # give time to collect valid results
        # # # ### checkpoint weights
            # if valid_loss<=best_valid_loss: # valid_loss_collect[-2]>valid_loss_collect[-1] and valid_accuracy_collect[-1]>valid_accuracy_collect[-2]:
            #     torch.save(model.state_dict(), pretrained_name_save)
            #     best_valid_loss = valid_loss
            #     print('bestvalid',best_valid_loss)
        #     # if np.mean(valid_loss_collect[-5:-1])<=valid_loss and np.mean(valid_accuracy_collect[-5:-1])>=valid_accuracy:
        #     #     print('early stopping',np.mean(valid_loss_collect[-5:-1]),valid_loss, np.mean(valid_accuracy_collect[-5:-1]),valid_accuracy)
        #     #     break

]
    collect_epoch_avg_cost_pop = np.mean(np.reshape(collect_avg_state_cost, (-1, counter)),axis=1)
    collect_epoch_best_cost_pop = np.mean(np.reshape(collect_cost_best_state, (-1, counter)),axis=1)


    ## Saving results
    results = np.vstack((train_loss_collect,train_accuracy_collect,valid_loss_collect,valid_accuracy_collect,valid_accuracy_collect3,valid_accuracy_collect5,collect_epoch_avg_cost_pop,collect_epoch_best_cost_pop,kp_collect,lr_collect,valid_energy_collect)) # stack in order vertically
    results_evolutionary = np.vstack((collect_avg_state_cost,collect_cost_best_state)) # stack in order vertically

    ts = int(time.time())
    res_name = 'results/loss_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    res_name_evol = 'results/evolutionaryCost_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'

    if not os.path.exists('results'):
        os.makedirs('results')
    np.savetxt(res_name,results)
    np.savetxt(res_name_evol,results_evolutionary)
    ## Testing
    test_name = 'results/test_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    # np.savetxt(test_name,[test_loss,test_accuracy]) # Test results [test loss, test accuracy in percentage]

    ############## Test the model on test data ############## 
    ##
    # Load checkpoint
    # pretrained_weights = torch.load(pretrained_name_save)
    # model.load_state_dict(pretrained_weights)
    
    
    print(30*'*')
    print('Energy Prouning results')
    final_state = np.expand_dims(best_state, axis=0)
    final_state[0,int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one

    print('Ising state ',final_state)
    states_reshaped = state_reshaper(device,1,final_state,network_size,args)
    test_loader = dataloader.testdata(kwargs,args,input_size,dataset)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader, states_reshaped[str(0)],loss_func,isvalid=False)
    total_p,kept_conv,total_conv,kept_fc,total_fc, kp = kept_counter(network_size, final_state)
    report = ['TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)]

     ## Save model weights   
    # if args.save_model:
    #     torch.save(model.state_dict(), pretrained_name_save)


    ##
    print(30*'*')
    print('Energy Dropout But No Pruning results')
    final_state = np.asarray(int(D_Conv2d+D_fc)*[1]) # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)
    print(final_state.shape)
    states_reshaped = state_reshaper(device,1,final_state,network_size,args)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    total_p,kept_conv,total_conv,kept_fc,total_fc, kpND = kept_counter(network_size, final_state)
    report.extend(['Energy Dropout But No Pruning results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])

    ##
    print(30*'*')
    print('Random Pruning @50 results')

    D2=int(np.ceil(D/2))
    final_state = np.hstack([np.zeros((1,D2),dtype='float64'), np.ones((1,D-D2),dtype='float64')])[0]
    np.random.shuffle(final_state)
    final_state[int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)

    states_reshaped = state_reshaper(device,1,final_state,network_size,args)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    total_p,kept_conv,total_conv,kept_fc,total_fc, kpRD = kept_counter(network_size, final_state)
    print('Number of trainable parameters: ', count_parameters(model))
    print(test_name)
    report.extend(['Energy Dropout With Random Pruning @50 pruned results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    
    print(30*'*')
    print('Random Pruning @25 results')

    D2=int(np.ceil(D/4))
    final_state = np.hstack([np.zeros((1,D2),dtype='float64'), np.ones((1,D-D2),dtype='float64')])[0]
    np.random.shuffle(final_state)
    final_state[int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)

    states_reshaped = state_reshaper(device,1,final_state,network_size,args)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    total_p,kept_conv,total_conv,kept_fc,total_fc, kpRD = kept_counter(network_size, final_state)
    print('Number of trainable parameters: ', count_parameters(model))
    print(test_name)
    report.extend(['Energy Dropout With Random Pruning @25 pruned results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    
    print(30*'*')
    print('Random Pruning @75 results')

    D2=int(np.ceil(3*D/4))
    final_state = np.hstack([np.zeros((1,D2),dtype='float64'), np.ones((1,D-D2),dtype='float64')])[0]
    np.random.shuffle(final_state)
    final_state[int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
    final_state = np.expand_dims(final_state, axis=0)

    states_reshaped = state_reshaper(device,1,final_state,network_size,args)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5, valid_energy = test_c(args, model, network_size, device, test_loader,states_reshaped['0'],loss_func,isvalid=False)
    total_p,kept_conv,total_conv,kept_fc,total_fc, kpRD = kept_counter(network_size, final_state)
    print('Number of trainable parameters: ', count_parameters(model))
    print(test_name)
    report.extend(['Energy Dropout With Random Pruning @75 pruned results','TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5),'Total states: '+str(total_p), 'KeptConv: '+str(kept_conv),'TotalConv: '+str(total_conv),'KeptFC: '+str(kept_fc),'TotalFC: '+str(total_fc), 'TotalKept: '+str(kp), 'TestEnergy: '+str(valid_energy)])
    

    report.extend(['Number of trainable parameters: '+str(count_parameters(model))])
    np.savetxt(test_name, report, delimiter="\n", fmt="%s") # Test results [test loss, test accuracy in percentage]


    return None



def train(args,pretrained_name,pretrained_name_save,stopcounter, input_size, n_classes,s_input_channel,nnmodel,ts):

    # Pytorch Setup
    torch.manual_seed(args.seed)
    torch.cuda.is_available()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"  
    device = torch.device(dev) 
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    print(nnmodel,device,s_input_channel,n_classes)
    model = model_select(nnmodel,device,s_input_channel,n_classes)
    # weights_init = weights_initialization(model)
    print('################  Model Setup  ################')

    # load data and weights
    train_loader, valid_loader = dataloader.traindata(kwargs, args, input_size, valid_percentage, dataset)

    ## Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0.000125)
    scheduler = StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.gamma,last_epoch=-1)

    ## Result collection
    train_loss_collect = []
    train_accuracy_collect = []
    valid_loss_collect = []
    valid_accuracy_collect = []
    valid_accuracy_collect3 = []
    valid_accuracy_collect5 = []
    lr_collect= []
    # Initialization Step
    for epoch in range(1, args.epochs + 1):
        n_batches = len(train_loader)
        epoch_accuracy = 0
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output, logits, _ = model(data)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability       
            correct = pred.eq(target.view_as(pred)).sum().item()
            loss.backward()
            optimizer.step()

            acc = 100*(correct/np.float(args.batch_size))
            epoch_loss+=loss.item()
            epoch_accuracy+=acc
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                print('Accuracy: ', acc)
                print('Epoch: ',epoch)
                print('Learning rate: ',scheduler.get_lr())
                print(10*'*')

            # Break
            if batch_idx==stopcounter:
                print('Broke at batch indx ',batch_idx)
                break
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_accuracy = epoch_accuracy/(batch_idx+1)

        # Validation
        print('Validation at epoch %d is:'%(epoch))

        valid_loss, valid_accuracy,valid_accuracy3,valid_accuracy5 = test(args, model, device, valid_loader,criterion,isvalid=True)
        scheduler.step()

        ## Result collection
        train_loss_collect.append(epoch_loss)
        train_accuracy_collect.append(epoch_accuracy)
        valid_loss_collect.append(valid_loss)
        valid_accuracy_collect.append(valid_accuracy)
        valid_accuracy_collect3.append(valid_accuracy3)
        valid_accuracy_collect5.append(valid_accuracy5)
        lr_collect.append(scheduler.get_lr()[0])

        ## Early Stopping
        if epoch>50: # give time to collect valid results
        # #     # checkpoint weights
            if valid_loss_collect[-2]>valid_loss_collect[-1] and valid_accuracy_collect[-1]>valid_accuracy_collect[-2]:
                torch.save(model.state_dict(), pretrained_name_save)
        #     if np.mean(valid_loss_collect[-5:-1])<valid_loss and np.mean(valid_accuracy_collect[-5:-1])>valid_accuracy:
        #         print('early stopping',np.mean(valid_loss_collect[-5:-1]),valid_loss, np.mean(valid_accuracy_collect[-5:-1]),valid_accuracy)
        #         break



    ############## Test the model on test data ############## 
    # # Load checkpoint
    pretrained_weights = torch.load(pretrained_name_save)
    model.load_state_dict(pretrained_weights)
    ## Loading data
    test_loader = dataloader.testdata(kwargs,args,input_size,dataset)
    test_loss,test_accuracy,test_accuracy3,test_accuracy5 = test(args, model, device, test_loader,criterion,isvalid=False)
    ## Testing
    test_name = 'results/test_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    # np.savetxt(test_name,[test_loss,test_accuracy]) # Test results [test loss, test accuracy in percentage]
    # print('Number of trainable parameters: ', count_parameters(model))
    report = ['TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5)]

    ## Saving results
    results = np.vstack((train_loss_collect,train_accuracy_collect,valid_loss_collect,valid_accuracy_collect,valid_accuracy_collect3,valid_accuracy_collect5,lr_collect)) # stack in order vertically
    res_name = 'results/loss_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    if not os.path.exists('results'):
        os.makedirs('results')
    np.savetxt(res_name, results)

    report.extend(['Number of trainable parameters: '+str(count_parameters(model))])
    np.savetxt(test_name, report, delimiter="\n", fmt="%s") # Test results [test loss, test accuracy in percentage]
    print(test_name)
    ## Save model weights
    if args.save_model:
        torch.save(model.state_dict(), pretrained_name_save)

    return None


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1., metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--lr_stepsize', type=float, default=50, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-ls', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--save_model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('--NS', default=8, help='pop size')
    parser.add_argument('--stopcounter', default=10e31, help='stopcounter')
    parser.add_argument('--threshold_early_state', default=100, help='threshold_early_state')
    parser.add_argument('--pre_trained_f', default=False, help='load pretrained weights')
    parser.add_argument('--scheduler_m', default='StepLR', help='lr scheduler')
    parser.add_argument('--optimizer_m', default="Adadelta", help='optimizer model')


    args = parser.parse_args()
    argparse_dict = vars(args)
    loss_func = 'ce'

    global dataset, nnmodel
    global network_size
    global n_classes
    global s_input_channel

    dataset = 'fashion' #kuzushiji cifar10 cifar100 flowers
    nnmodel = 'resnet18' # resnet34 resnet50 resnet101
    mode = 'ising' #simple

    stopcounter = args.stopcounter #10#e10
    NS = args.NS # number of candidate states
    n_classes, s_input_channel = dataloader.dataset_specs(dataset)
    input_size = (s_input_channel,32,32)
    valid_percentage = 0.1 # out of 1

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    ts = int(time.time())
    pretrained_name_save = "weights/"+dataset+"_"+mode+"_"+nnmodel+"_"+str(ts)+".pt"
    pretrained_name = 'None' 
        

    torch.cuda.set_device(0)

    if mode=='ising':
        threshold_early_state = args.threshold_early_state
        train_ising(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size,threshold_early_state, args.NS)
    elif mode=='simple':
        train(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,ts)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size)
    elif mode=='random':
        threshold_early_state = 0
        args.NS = 1
        train_ising(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size)

    ## Save args
    argparse_dict_name = 'results/args_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.json'
    with open(argparse_dict_name, 'w') as fp:
        json.dump(argparse_dict, fp)
