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
# import matplotlib.pyplot as plt
import copy
# import tensorflow as tf
from torchsummary import summary as summary2
import torchvision
from numpy import unique
from scipy.stats import entropy as scipy_entropy
# from tqdm import tqdm
from scipy.spatial import distance

sys.path.insert(0, '..')
from utils import dataloader,hsummary #import utils.dataloader 
# from utils import progress_bar
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

            ### With log
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
    # test_loss /= n_test
    # accu = 100. * correct / n_test
    
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
  
    #### with cross entropy
    # n_test = total
    # test_loss /= n_test
    # accu = 100.*correct/n_test

            ### With log
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            # counter+=1
    # test_loss /= n_test
    # accu = 100. * correct / n_test


    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #     test_loss, correct, n_test, accu))
    # print(20*'#')
    # ## Restore network weights from backup
    # weight_restore(model, network_size, backup_weights)

  
def state_reshaper(device,NS,states,network_size,args):
    states_reshaped = {}
    for state_indx in range(NS):
        ## State reshpaer from vector to mask
        states_reshaped[str(state_indx)]={}
        st = 0 # state counter
        ## For Convolution Layers
        # base_key = 'Conv2d'
        for indx, val in enumerate(network_size[0]):
            # print(indx,val[2])
            con = states[state_indx, st:st+val[2][0]]
            if len(val[2])==4:
                tmp = con[:,np.newaxis,np.newaxis,np.newaxis]
                tmp = np.tile(tmp,(1,val[2][1],val[2][2],val[2][3]))
            elif len(val[2])==1:
                tmp = con
            else:
                raise('unknown tensor shape')
            # print('after',con,tmp.shape)  
            # print(str(state_indx),val[0])
            # tmp = np.zeros((val[2]))
            # for ii in range(len(con)):
            #     tmp[ii,:,:,:] = con[ii]
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
            # print(val[2])
            con = states[state_indx, st:st+val[2][1]]
            # print('this',con,con.shape) 
            # tmp = np.tile(con,) 
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
    # pp = np.asarray(np.ones((NS,D)),dtype='float64')
    # new_population[:int(NS/2),:] = pp[:int(NS/2),:]
    # print('newpop',np.sum(new_population,axis=1))

    return new_population


def ising_cost(states,NS,D, D_Conv2d,D_fc,entrop_sig,fcbias_sig,states_loss, interaction_sigma,network_size):
    cost_population = np.zeros((NS,1))
    #### Compute final energy
    cost_breakdown= np.zeros((NS,3))
    for state_indx in range(NS):
        # bias_conv = np.sum(np.multiply(states[state_indx,:D_Conv2d],entrop_sig[state_indx,:]))  # entropy of the featuremap
        # bias_fc = np.sum(np.multiply(states[state_indx,D_Conv2d:],fcbias_sig[state_indx,:]))
        # bias = bias_conv + bias_fc
        # dd = np.tile(states[state_indx,:],(D,1))
        # interaction = 0.5*(np.sum(dd*interaction_sigma[state_indx,:,:]*np.transpose(dd))) # /D to normalize  >> we do not need to normalize # Remove 0.5* because one side interaction we have in network sensory - later put gradient on the reverse connection
        # cost_breakdown[state_indx,0] = -interaction
        # cost_breakdown[state_indx,1] = -bias


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
    backup_weights['conv1'] = model._modules['conv1'].weight.data.clone()
    backup_weights['conv1bn1'] = model._modules['bn1'].weight.data.clone()

    backup_weights['fc'] = model._modules['fc'].weight.data.clone()
    for lay in network_size[0][1:]: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
        # print(lay)
        lay_name = lay[0] # such as layer3.1.conv1
        if 'conv' in lay_name:
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            conv_name = lay_name.split('.')[2]
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.clone()         
        #     model._modules['layer1'][indx]._modules['conv1'].weight.data = torch.zeros(sh)
        
            if conv_name=='conv1':
                backup_weights[lay_name+'bn1'] = model._modules[layer_name][layer_indx]._modules['bn1'].weight.data.clone()  

            elif conv_name=='conv2':
                backup_weights[lay_name+'bn2'] = model._modules[layer_name][layer_indx]._modules['bn2'].weight.data.clone()  
        elif 'downsample' in lay_name:
            layer_name = lay_name.split('.')[0]
            layer_indx = int(lay_name.split('.')[1])
            downs_indx = lay_name.split('.')[3]
            # print(model._modules[layer_name][layer_indx])
            backup_weights[lay_name] = model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data.clone()         

        else:
            raise('unknown layer')
    
    return backup_weights
    

def weight_restore(model, network_size, backup_weights):
    with torch.no_grad():
        # print(backup_weights['conv1'])
        # print(model._modules['conv1'].weight.data)
        # print('here')
        model._modules['conv1'].weight.data.copy_(backup_weights['conv1'])
        model._modules['bn1'].weight.data.copy_(backup_weights['conv1bn1'])
        model._modules['fc'].weight.data.copy_(backup_weights['fc']) 
        for lay in network_size[0][1:]: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
            # print(lay)
            lay_name = lay[0] # such as layer3.1.conv1
            if 'conv' in lay_name:    
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].weight.data.copy_(backup_weights[lay_name]) 
                if conv_name=='conv1':
                    model._modules[layer_name][layer_indx]._modules['bn1'].weight.data.copy_(backup_weights[lay_name+'bn1'])
                elif conv_name=='conv2':
                    model._modules[layer_name][layer_indx]._modules['bn2'].weight.data.copy_(backup_weights[lay_name+'bn2'])
            
            elif 'downsample' in lay_name:
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                downs_indx = lay_name.split('.')[3]
                # print(model._modules[layer_name][layer_indx])
                model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data.copy_(backup_weights[lay_name]) 

            else:
                raise('unknown layer')

def sparsifier(model,network_size, state_in):
    # print(model._modules)
    with torch.no_grad():
        model._modules['conv1'].weight.data *= state_in['conv1']
        model._modules['bn1'].weight.data *=  state_in['conv1'][:,0,0,0]
        # model._modules['bn1'].weight.data *= state_in['conv1'][0,]

        model._modules['fc'].weight.data *= state_in['fc']
        for lay in network_size[0][1:]: # [0]: is for convs - [1:]: Skipping conv1 - did it up there
            # print(lay)
            lay_name = lay[0] # such as layer3.1.conv1
            if 'conv' in lay_name:            
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                conv_name = lay_name.split('.')[2]
                model._modules[layer_name][layer_indx]._modules[conv_name].weight.data *=state_in[lay_name]

                if conv_name=='conv1':
                    model._modules[layer_name][layer_indx]._modules['bn1'].weight.data *=state_in[lay_name][:,0,0,0]

                elif conv_name=='conv2':
                    model._modules[layer_name][layer_indx]._modules['bn2'].weight.data *=state_in[lay_name][:,0,0,0]
            elif 'downsample' in lay_name:
                layer_name = lay_name.split('.')[0]
                layer_indx = int(lay_name.split('.')[1])
                downs_indx = lay_name.split('.')[3]
                model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data *=state_in[lay_name]
                # print(model._modules[layer_name][layer_indx]._modules['downsample'][int(downs_indx)].weight.data.shape,state_in[lay_name].shape)
            else:
                raise('unknown layer')
    return None


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





def network_sensory(args,model,data,target,device,NS,stopcounter, states_reshaped,network_size,D,D_Conv2d, D_fc,n_classes):
    states_loss = np.zeros((NS,1))
    entropy_signals = {}

    for i in range(len(network_size[0])):
        entropy_signals[network_size[0][i][0]] = np.zeros((NS, network_size[0][i][2][0]))
        # print(network_size[0][i],entropy_signals[network_size[0][i][0]].shape)
    interaction_sigma = np.zeros((NS,D,D))
    fcbias_sig = np.zeros((NS, D_fc))    

    # ## Get signals value wihtout sparsication
    # with torch.no_grad():
    #     _, _, net_signals  = model(data) 

    ## Backup weights
    backup_weights = weight_backer(model,network_size)
    # print(backup_weights.keys())


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

        # print(net_signals.keys())
        ## get energy of other classes than target
        net_eneg = torch.sum(torch.mul(one_hot,net_energy),axis=1)
        other_net_eneg = high_cost_target_one_hot + net_energy # mean over batches
        max_other_net_eneg = torch.max(other_net_eneg,1)

        eng_diff = net_eneg - max_other_net_eneg[0]
        states_loss[state_indx,0] += eng_diff.mean(0)

        ## Entropy of featuremaps
        counts_256_conv1 = {}
        for conv in network_size[0]:
            key = conv[0]
            # print(conv)
            size_fm = conv[2][0]

            mean_signal1 = net_signals[key].mean((0)) # mean over batches
            # print(mean_signal1.shape,size_fm)
            # print(mean_signal1.shape)
            mean_signal1 = mean_signal1.cpu()
            # print(mean_signal1)
            # mean_signal1 = np.nan_to_num(mean_signal1/mean_signal1.max())
            mean_signal1 = np.clip(mean_signal1,0,1) #/mean_signal1.max())

            # print(mean_signal1)
            quantized_sig1_x = np.asarray(255.*mean_signal1,'int')
            # print(quantized_sig1_x.shape)
            counts_256_conv1[key] = np.zeros((size_fm,256)) # probability distribution
            # print(key,net_signals[key].shape,conv)
            for fm in range(size_fm):
                # print(quantized_sig1_x.shape)
                u, counts = unique(quantized_sig1_x[fm,:,:], return_counts=True)
                # print(u,counts)
                # u, counts = unique(np.reshape(quantized_sig1_x[fm,:,:],(size_fm_out,size_fm_out)), return_counts=True)
                counts_256_conv1[key][fm,u] = counts #/(size_fm_out**2) # normalize - convert to probability
                entropy_signals[key][state_indx,fm]+= (-1*np.sum(np.nan_to_num(counts_256_conv1[key][fm,:]*np.log2(counts_256_conv1[key][fm,:]))))

        ## Interaction - KL divergence of featuremaps inside each layer to eliminate redundant ones
        prev = 0
        for conv in network_size[0]:
            key = conv[0]
            size_fm = conv[2][0]
            # print('conv',conv)
        # for key in entropy_signals.keys():
            # net_key_indx = int(key.split('-')[1])-1
            
            # size_fm = network_size[0][key][2][1] #[net_key_indx][0]
            
            # for i in range(0,size_fm):
            #     print('i',i)
            #     for j in range(i+1,size_fm):
            #         # sigma1 = counts_256_conv1[key][i,:]*np.log2((counts_256_conv1[key][i,:])/((counts_256_conv1[key][i,:]+counts_256_conv1[key][j,:])/2.))
            #         # sigma2 = counts_256_conv1[key][j,:]*np.log2((counts_256_conv1[key][j,:])/((counts_256_conv1[key][j,:]+counts_256_conv1[key][i,:])/2.))
            #         # sigma = np.sum(np.nan_to_num(sigma1+sigma2)) # sigma=0 : similar; sigma=1 : not similar ; using Jensen–Shannon divergence [ D_kl(p||(p+q)/2) + D_kl(q||(p+q)/2) ]
            #         sigma = distance.jensenshannon(counts_256_conv1[key][i,:],counts_256_conv1[key][j,:], 2.0)
            #         interaction_sigma[state_indx, prev+i, prev+j] = sigma 
            #         interaction_sigma[state_indx, prev+j, prev+i] = sigma  # Replcae with gradient in future
            #         # import matplotlib.pyplot as plt
                    # plt.plot(counts_256_conv1[key][i,:])
                    # plt.show()
            
                        
            def fun2(b,i,j):
                return distance.jensenshannon(b[i,:],b[j,:], 2.0)

            def fun(a,i,size_fm):
                return Parallel(n_jobs=36)(delayed(fun2)(a,i,j) for j in range(0,size_fm))
            # st = time.time()
            sigma = Parallel(n_jobs=36)(delayed(fun)(counts_256_conv1[key],i,size_fm) for i in range(0,size_fm))
            
            # interaction_sigma[state_indx, prev:prev+size_fm, prev:prev+size_fm] += np.triu(sigma,k=0) 
            # interaction_sigma[state_indx, prev+j, prev+i] = sigma  # Replcae with gradient in future                 
            interaction_sigma[state_indx, prev:prev+size_fm, prev:prev+size_fm] += sigma 
            
            
            
            
            prev += size_fm 

        ## Dense Layers Interaction Sigma
        prev = 0
        for fc in network_size[1]:
            key = fc[0]
            current_layer_size = fc[2][1]
            next_layer_size = fc[2][0]
            # print(key)
            # print(net_signals)
            dd = net_signals[key].unsqueeze_(-2)
            # print(dd.shape)
            gg = dd.repeat(1,next_layer_size,1)
            # next_layer_size = getattr(model,key2).weight.data
            rr = model._modules[key].weight.data.repeat(args.batch_size,1,1)

            res = torch.relu(rr*gg)
            ll = res.mean(0)              
            sti = D_Conv2d + prev
            fnsh = D_Conv2d + prev + current_layer_size # int(np.sum([network_size['fc'][k][0] for k in range(0,i+1)]))
            fnsh2 = D_Conv2d + prev + current_layer_size + next_layer_size # int(np.sum([network_size['fc'][k][0] for k in range(0,i+2)]))
            # print(sti,fnsh,fnsh2,ll.shape, interaction_sigma.shape)
            interaction_sigma[state_indx,sti:fnsh,fnsh:fnsh2] = ll.t().cpu()
            interaction_sigma[state_indx,fnsh:fnsh2,sti:fnsh] = ll.cpu()

            ### Bias Term
            # sti = int(np.sum([network_size['fc'][k][0] for k in range(0,i)]))
            # fnsh = int(np.sum([network_size['fc'][k][0] for k in range(0,i+1)]))
            # print(sti,fnsh,D_fc)
            b_sing = model._modules[key].weight.data.mean(0)
            # print(b_sing.shape)
            sti = prev
            fnsh = prev + current_layer_size # int(np.sum([network_size['fc'][k][0] for k in range(0,i+1)]))

            fcbias_sig[state_indx,sti:fnsh] = b_sing.cpu()
            prev+=current_layer_size



            # def funNS():
            #     return entropy_signals
            # # sigma = Parallel(n_jobs=36)(delayed(fun)(counts_256_conv1[key],i,size_fm) for i in range(0,size_fm))
            # entropy_signals = Parallel(n_jobs=36)(delayed(funNS)() for i in range(0,NS))


    stack_list = [entropy_signals[k] for k in entropy_signals.keys()]
    entrop_sig = np.hstack(stack_list) # stack all entropys

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
    else:
        raise TypeError("Check the model name")
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
    elif nnmodel == 'resnet34':
        realresnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet50':
        realresnet.gradient_mask(model,best_stateI,device,nnmodel)
    elif nnmodel == 'resnet101':
        realresnet.gradient_mask(model,best_stateI,device,nnmodel)
    else:
        raise TypeError("Check the model name")

def key_maker(model):
    # print(model.named_modules)
    # print(model._modules)
    # time.sleep(10)
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
        # print(i,kk[i].shape)
        if ('conv' in i) or ('downsample' in i and 'weight' in i):
            name = '.'.join(i.split('.')[:-1])
            attr = i.split('.')[-1]
            li=[name,attr,kk[i].shape]
            convs_list.append(li)
            D_Conv2d+=kk[i].shape[0]
            # print('got',li,kk[i].shape[0],len(kk[i].shape))
        elif 'fc' in i:
            name = ''.join(i.split('.')[:-1])
            attr = i.split('.')[-1]
            if attr == 'weight':
                li=[name,attr,kk[i].shape]
                linear_list.append(li)   
                D_fc+= kk[i].shape[1]
    # print(convs_list,linear_list)
                # print(D_fc)
    # D_fc+=n_classes
    D = D_Conv2d+D_fc #+n_classes
    network_size = [convs_list,linear_list]
    # print(network_size)
    # print(network_size)
    # print(D_Conv2d,D_fc)
    return network_size, D, D_Conv2d, D_fc


def kept_counter(network_size,final_state):
    # print(network_size)
    counter = 0
    total_conv = 0    # no buffer included
    # total_conv_with_buffer = 0
    kept_conv = 0
    # kept_conv_with_buffer = 0
    # cc = 0
    stride_s = 1
    for indx, con in enumerate(network_size[0]):
        # print(indx,con,len(con[2]))
        if len(con[2])==4:
            k_s = con[2][2] # kernel size
            f_size = con[2][0] # number of fmaps
            f_size_in = con[2][1] # number of fmaps in

            active_n = np.sum(final_state[0,counter:counter+con[2][0]])

            kept_conv+=(k_s*k_s*f_size_in*stride_s+1)*active_n
            total_conv+=(k_s*k_s*f_size_in*stride_s+1)*f_size
            # kept_conv += active_n*f_size_in*k_s*k_s + 2*(active_n)# kept parameters in current conv
            # kept_conv_with_buffer += active_n*f_size_in*k_s*k_s + 4*(active_n)
        elif len(con[2])==1:
            f_size = con[2][0] # number of fmaps

            active_n = np.sum(final_state[0,counter:counter+con[2][0]])
            
            kept_conv+= active_n
            total_conv+= f_size
            # print(active_n,f_size_in)
        else:
            raise('unknown size in counter')
    
        counter += f_size
        # print(k_s,f_size,f_size_in,active_n,active_n*f_size_in*k_s*k_s + 2*(active_n),active_n*f_size_in*k_s*k_s + 4*(active_n),f_size*f_size_in*k_s*k_s + 2*f_size,f_size*f_size_in*k_s*k_s + 4*f_size)
        # cc+=(k_s*k_s*f_size_in*stride_s+1)*f_size
        # print((k_s*k_s*f_size_in+1)*f_size,cc)
    total_fc = network_size[1][0][2][1]*n_classes
    # print(final_state.shape)
    # print(final_state[0,counter:-n_classes])
    kept_fc = np.sum(final_state[0,counter:])*n_classes
    kp = (kept_fc+kept_conv)/float(total_fc+total_conv)
    # kpwb = (kept_fc+kept_conv_with_buffer)/float(total_fc+total_conv_with_buffer) # with buffer

    total_p = total_fc+total_conv+n_classes # n_classes is bias of last layer with length n_classes
    # total_p_wbuffer = total_fc+total_conv_with_buffer+n_classes # n_classes is bias of last layer with length n_classes
    print('Total parameters: ',total_p)
    print('Kept Conv:',kept_conv,'/',total_conv)
    print('Kept Fc:',kept_fc,'/',total_fc)
    print('Kept Percentile:', kp)
    # print('\n')
    # print('Total parameters wbuffer: ',total_p_wbuffer)
    # print('Kept Conv: wbuffer',kept_conv_with_buffer,'/',total_conv_with_buffer)
    # print('Kept Fc:',kept_fc,'/',total_fc)
    # print('Kept Percentile wbuffer:', kpwb)

    # time.sleep(10)
    return total_p,kept_conv,total_conv,kept_fc,total_fc, kp

def weights_initialization(model):
    for m in model.modules():
        print(m)
        if isinstance(m, nn.Conv2d):
            print(type(m))
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

def train_ising(args,pretrained_name,pretrained_name_save,stopcounter, input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func, limiteddata):

    ### Pytorch Setup
    torch.manual_seed(args.seed)
    torch.cuda.is_available()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"  
    device = torch.device(dev) 
    kwargs = {'num_workers': 20, 'pin_memory': True} if use_cuda else {}
    model = model_select(nnmodel,device,s_input_channel,n_classes)

    network_size, D, D_Conv2d, D_fc = key_maker(model)
    # final_state = np.asarray(int(D_Conv2d+D_fc)*[1]) # output layer must be one
    # final_state = np.expand_dims(final_state, axis=0)
    # kept_counter(network_size,final_state)

    ### Load data
    train_loader, valid_loader = dataloader.traindata(kwargs, args, input_size, valid_percentage, dataset, limiteddata)
    n_batches = len(train_loader)

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
    ##TODO: add follwing schedulers
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # scheduler = StepLR(optimizer, step_size=args.lr_stepsize, gamma=args.gamma, last_epoch=-1)
    # scheduler = MultiStepLR(optimizer, milestones[50,100,150], gamma=args.gamma, last_epoch=-1)
    # scheduler = ExponentialLR(optimizer, gamma=0.97,last_epoch=-1)

    ## Result collection
    train_loss_collect = []
    train_accuracy_collect = []
    valid_loss_collect = []
    valid_accuracy_collect = []
    valid_accuracy_collect3 = []
    valid_accuracy_collect5 = []
    # Initialization Step
    # epoch = 0
    NS = args.NS
    states_rand = np.asarray(np.random.randint(0,2,(NS,D)),dtype='float64') # Population is 0/1
    states_init = np.asarray(np.ones((NS,D)),dtype='float64') # Population is 0/1
    # states_init[:int(NS/2),:] = states_rand[:int(NS/2),:]
    # states_init[1:,:] = states_rand[1:,:]

    states_init = states_rand

    states_rand_one = np.asarray(np.random.randint(0,2,(1,D)),dtype='float64') # Population is 0/1
    states_rand_one0 = states_rand_one[0,:]
    # print(states_init,np.sum(states_init,axis=1))
    
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
    etc = 0
    etime = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

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
                    # if np.mean(valid_energy_collect[-5:-1])<valid_energy_collect[-1] and state_converged == False:  # last best from previous epoch
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
                        # print('\n new ',batch_idx,new_states,cost_new_states, np.sum(new_states,axis=1))
                        states, cost_states, best_state, cost_best_state, avg_cost_states = selection(states, new_states, cost_states, cost_new_states, NS, D)
                        # print('\n selected',batch_idx,states,cost_states,np.sum(states,axis=1))

            ## Collect evolution cost
            collect_avg_state_cost.append(avg_cost_states)
            collect_cost_best_state.append(cost_best_state)

            # print(network_size)
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
            # if loss_func=='ce':
            #     criterion = nn.CrossEntropyLoss()
            #     loss = criterion(output, target)
            #     pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability       
            #     correct = pred.eq(target.view_as(pred)).sum().item()

            # loss = energyloss(net_eng,target,n_classes,args,device)
            ###################


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
            # weight_restore(model, network_size, backup_weights)

            # Compute Loss
            # loss = F.nll_loss(output, target) 

            # for indx, p in enumerate(model.parameters()):
            #     c = 0
            #     print(indx,p.grad.shape)
            

            # if state_converged==False: # Restore weights if state not converged
            # optimizer.step() # Updates the weights
            ## Restore network weights from backup
            # weight_restore(model, network_size, backup_weights)
            # backup_weights = weight_backer(model,network_size,)

            # for indx, p in enumerate(model.parameters()):
            #     print(indx,np.sum(p.grad.data.cpu().numpy()))
            # print(correct,loss.item())

            acc = 100.*(correct/np.float(args.batch_size))
            # print(correct,loss.item(),np.float(args.batch_size),acc)

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
                final_state = np.expand_dims(best_state, axis=0)
                final_state[0,int(D_Conv2d+D_fc-n_classes):] = 1 # output layer must be one
                kept_counter(network_size, final_state)

                print(10*'*')


            loss.backward() # computes gradients for ones with reques_grad=True
            gradient_masker(model,best_stateI,nnmodel,device)
            weight_restore(model, network_size, backup_weights)

            optimizer.step()
            optimizer.zero_grad() ######TODOm

           
            # output, _, _  = model(data) 
            # criterion = nn.CrossEntropyLoss()
            # loss = criterion(output, target)

            # # Compute Loss
            # # loss = F.nll_loss(output, target) 
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability       
            # correct = pred.eq(target.view_as(pred)).sum().item()
            # acc = correct/np.float(args.batch_size)
            # if batch_idx % args.log_interval == 0:
            #     print(nnmodel,dataset)
            #     print('2Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
            #     print('2Accuracy: ', acc)
            #     print('2Epoch: ',epoch)
            #     print('2Population min cost: ',cost_best_state)
            #     print('2Population avg cost: ',avg_cost_states)
            #     print('2Learning rate: ',scheduler.get_lr())
            #     print(10*'*')




            # Break
            counter+=1
            if counter==stopcounter or counter==(n_batches-1):
                # entrop_sig = entrop_sig/counter # Average over batches
                # states_loss = states_loss/counter # Average over batches
                # interaction_sigma = interaction_sigma/counter # Average over batches
                epoch_loss = epoch_loss/counter
                epoch_accuracy = epoch_accuraacy/counter
                states_reshaped = state_reshaper(device,NS,states,network_size,args)

                break

        etime+= time.time()-start_time 
        etc+=1
        print('time:',etime/etc)
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



    # collect_epoch_avg_cost_pop = [np.mean(i) for i in collect_avg_state_cost]
    # collect_epoch_best_cost_pop = [np.mean(i) for i in collect_cost_best_state]
    collect_epoch_avg_cost_pop = np.mean(np.reshape(collect_avg_state_cost, (-1, counter)),axis=1)
    collect_epoch_best_cost_pop = np.mean(np.reshape(collect_cost_best_state, (-1, counter)),axis=1)

    # print(collect_epoch_avg_cost_pop.shape)
    # print(collect_epoch_best_cost_pop.shape)

    ## Saving results
    results = np.vstack((train_loss_collect,train_accuracy_collect,valid_loss_collect,valid_accuracy_collect,valid_accuracy_collect3,valid_accuracy_collect5,collect_epoch_avg_cost_pop,collect_epoch_best_cost_pop,kp_collect,lr_collect,valid_energy_collect)) # stack in order vertically
    results_evolutionary = np.vstack((collect_avg_state_cost,collect_cost_best_state)) # stack in order vertically

    ts = int(time.time())
    res_name = 'resultstrain/loss_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    res_name_evol = 'resultstrain/evolutionaryCost_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'

    if not os.path.exists('resultstrain'):
        os.makedirs('resultstrain')
    np.savetxt(res_name,results)
    np.savetxt(res_name_evol,results_evolutionary)
    ## Testing
    test_name = 'resultstrain/test_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
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



def train(args,pretrained_name,pretrained_name_save,stopcounter, input_size, n_classes,s_input_channel,nnmodel,ts, limiteddata):

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
    # model.apply(weights_init)
    # pretrained_weights = torch.load(pretrained_name)
    # model.load_state_dict(pretrained_weights)

    # load data and weights
    train_loader, valid_loader = dataloader.traindata(kwargs, args, input_size, valid_percentage, dataset, limiteddata)

    ## Optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, rho=0.9, eps=1e-06, weight_decay=0.000125)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.000125,momentum=0.9,nesterov=True)
    # scheduler = ExponentialLR(optimizer, gamma=0.97,last_epoch=-1)

    # scheduler = MultiStepLR(optimizer, milestones=[2.,4.,6.], gamma=0.1, last_epoch=-1)
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
    etc = 0
    etime = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        n_batches = len(train_loader)
        epoch_accuracy = 0
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output, logits, _ = model(data) 
            # Compute Loss
            # loss = F.nll_loss(output, target) 
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability       
            correct = pred.eq(target.view_as(pred)).sum().item()
            # print(output,logits,loss,pred,correct,target)
            # print(10*'*')
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

            # optimizer.zero_grad()



            # Break
            if batch_idx==stopcounter:
                print('Broke at batch indx ',batch_idx)
                break
        epoch_loss = epoch_loss/(batch_idx+1)
        epoch_accuracy = epoch_accuracy/(batch_idx+1)

        etime+= time.time()-start_time 
        etc+=1
        print('time:',etime/etc)
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
        # if epoch>1 and scheduler.get_lr()[0]<0.0001:
        if epoch>5: # give time to collect valid results
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
    test_name = 'resultstrain/test_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    # np.savetxt(test_name,[test_loss,test_accuracy]) # Test results [test loss, test accuracy in percentage]
    # print('Number of trainable parameters: ', count_parameters(model))
    report = ['TestLoss: '+str(test_loss),'TestAcc: '+ str(test_accuracy),'TestAcc3: '+str(test_accuracy3),'TestAcc5: '+str(test_accuracy5)]

    ## Saving results
    results = np.vstack((train_loss_collect,train_accuracy_collect,valid_loss_collect,valid_accuracy_collect,valid_accuracy_collect3,valid_accuracy_collect5,lr_collect)) # stack in order vertically
    res_name = 'resultstrain/loss_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.txt'
    if not os.path.exists('resultstrain'):
        os.makedirs('resultstrain')
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
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 64)')
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

    dataset = 'flowers' #kuzushiji
    nnmodel = 'resnet34'
    mode = 'simple'
    limiteddata = False # @1000 samples

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
        train_ising(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func, limiteddata)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size,threshold_early_state, args.NS)
    elif mode=='simple':
        train(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,ts, limiteddata)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size)
    elif mode=='random':
        threshold_early_state = 0
        args.NS = 1
        train_ising(args,pretrained_name,pretrained_name_save,stopcounter,input_size, n_classes,s_input_channel,nnmodel,threshold_early_state,ts,loss_func, limiteddata)
        print(mode,dataset,nnmodel,args.lr,args.epochs,args.gamma,args.batch_size)

    ## Save args
    argparse_dict_name = 'resultstrain/args_'+nnmodel+'_'+dataset+'_'+mode+'_'+str(ts)+'.json'
    with open(argparse_dict_name, 'w') as fp:
        json.dump(argparse_dict, fp)
