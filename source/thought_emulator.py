import os
import sys
import math
import numpy
import warnings
warnings.filterwarnings('ignore')

#For safe imports
file_directory = os.getcwd()
parent_directory = os.path.dirname(file_directory)
sys.path.insert(False, parent_directory)

import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import tqdm
import logging
import inspect

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
from torch.utils.data import DataLoader

from data.data import DatasetHandler
from data.data import CoTDataCollator
from data.data import extractAnswer

from source.configurations import ThoughtEmulatorConfig
from source.utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor, createAccuracyPlot, createLossPlot
from source.gpt2_implicit import GPT2LMHeadImplicitModel

from source.teacher import Teacher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

class ThoughtEmulator(nn.Module):
    def __init__(self, config : ThoughtEmulatorConfig, teacher : Teacher):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size
        
        #We turn each layer into a verticle set of layers.
        self.verticle_model = nn.ModuleList([nn.Sequential(
             nn.Linear(2*hidden_size, 4*hidden_size),
             nn.ReLU(),
             nn.Linear(4*hidden_size, hidden_size),
             ) for _ in range(num_layers)])

        self.mixture_components = nn.Embedding(config.mixture_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, \
                batch_first=False, dropout=0, bidirectional=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size*2, hidden_size)
        self.teacher = teacher

    def forward(self, input_ids, requires_backward=False):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        input_ids = input_ids[:, :sep_positions.max()+1]
        outputs = self.base_model.forward(mode='forward_emulator', \
                input_ids=input_ids, \
                positions_to_take=sep_positions, \
                softmax_temperature=self.config.softmax_temperature, \
                requires_backward=requires_backward, \
                rnn=self.rnn, \
                mlps=self.verticle_model, \
                mixture_components=self.mixture_components, \
                key_proj=self.key_proj, \
                query_proj=self.query_proj, \
                out_proj=self.out_proj)
        emulated_teacher_states = outputs.f_h_cs
        return emulated_teacher_states

    def computeLoss(self, input_ids, teacher_states):
        emulated_teacher_states = self.forward(input_ids=input_ids, requires_backward=True)
        batch_size = input_ids.shape[0]

        loss_function = nn.MSELoss(reduction='none')
        loss = 0
        correct = 0
        total = 0
        for teacher_state, emulated_teacher_state in zip(teacher_states, emulated_teacher_states):
            loss += loss_function(teacher_state, emulated_teacher_state).sum(-1) / 2
            correct += 1-(abs(teacher_state - emulated_teacher_state)).mean() #Check if reasonably close, used for quasi accuracy.
            total += 1
        loss = loss.mean()

        quasi_train_accuracy = correct / total #Keep track of closeness of emulated states to teacher states for a accuracy proxy.

        outputs = CausalLMOutputWithCrossAttentions(loss=loss)
        outputs.total_loss = loss * batch_size
        outputs.quasi_train_accuracy = quasi_train_accuracy.item()
        outputs.emulated_teacher_states = emulated_teacher_states
        return outputs

    def save_pretrained(self, save_directory) -> None:
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

    @torch.no_grad() #Freeze gradients.
    def evaluate(self, dataloader : DataLoader, ctx,):
        '''
        Calculates accuracy metrics on test data.
        '''
        self.eval() #Freeze loss function, gradients etc.

        total_instances = 0
        total_loss = 0
        sub_iteration = 0

        for batch in tqdm.tqdm(dataloader):
            #import pdb; pdb.set_trace()
            input_ids_cot = batch['input_ids_cot'].to(device)
            input_ids_only = batch['input_ids_only'].to(device)
            batch_size = input_ids_cot.shape[0]
            with ctx:
                teacher_states = self.teacher.extractStates(input_ids=input_ids_cot, delta=self.config.delta, subset=self.config.subset)
                outputs = self.computeLoss(input_ids=input_ids_cot, teacher_states=teacher_states)
                loss = outputs.loss
            total_loss += outputs.total_loss.item()
            total_instances += batch_size

            for i in range(len(input_ids_only)): #iterates through each individual batch
                sub_iteration += 1
                if sub_iteration >= len(dataloader.dataset)-2: # to limit spam of prediction examples.
                    #We print some of the states to compare.
                    print(f'Input: {self.tokenizer.decode(input_ids_only[i], skip_special_tokens=True)}')
                    print(f'Target H. Layer 1, V. Layer 1, first 9 states:')
                    print(np.round(teacher_states[0][0][:9].cpu().numpy(), decimals=4))
                    print (f'Predicted H. Layer 1, V. Layer 1, first 9 states: ')
                    print(np.round(outputs.emulated_teacher_states[0][0][:9].cpu().detach().numpy(), decimals=4))
                    print("")

        loss = total_loss / total_instances
        return outputs.quasi_train_accuracy, loss
    
    def predict(self, custom_data_handler : DatasetHandler) -> None:
        '''
        Used for custom test cases for fun. You can create custom test cases using the generateDataset using a DatasetHandler. Predicts 10 states.
        '''
        self.base_model.eval()

        # Load data
        tokenizer = self.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        custom_data_handler = DataLoader(custom_data_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

    
        for batch in tqdm.tqdm(custom_data_handler):
            input_ids_cot = batch['input_ids_cot'].to(device)
            input_ids_only = batch['input_ids_only'].to(device)

            teacher_states = self.teacher.extractStates(input_ids=input_ids_cot, delta=self.config.delta, subset=self.config.subset)
            emulated_teacher_states = self.forward(input_ids=input_ids_cot, requires_backward=True)

            #We print some of the states to compare.
            print (f'Input: {self.tokenizer.decode(input_ids_only[0], skip_special_tokens=True)}')
            print (f'Target Layer 1, Attention Head 1, first 9 states:')
            print(np.round(teacher_states[0][0][:9].cpu().numpy(), decimals=4))
            print (f'Predicted Layer 1, Attention Head 1, first 9 states:')
            print(np.round(emulated_teacher_states[0][0][:9].cpu().detach().numpy(), decimals=4))





    def trainModel(self, train_handler : DatasetHandler, test_handler : DatasetHandler, limit : float) -> None:
        '''
        Trains the model and automatically evaluates. 
        @limit hard caps the desired accuracy to stop training early if the threshold is met.
        '''
        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

        # Create Emulator 
        emulator  = self.to(device).to(ptdtype)

        # Load data
        tokenizer = self.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        train_dataloader = DataLoader(train_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=True)
        val_dataloader = DataLoader(test_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        # Create Optimizer
        trainable_params = list(emulator.parameters())
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(trainable_params, lr = self.config.eta, **extra_args)

        self.train() #Put model in training mode
        self.teacher.eval() #We want teacher to be fixed

        for p in self.teacher.parameters():
            p.requires_grad = False

        train_losses = []

        train_accs = []

        # Train
        iteration = 0
        for batch in tqdm.tqdm(train_dataloader):
            self.train()
        

            input_ids_cot_only  = batch['input_ids_cot'].to(device)
            input_w_nocot  = batch['input_ids_nocot'].to(device)
            with ctx:
                with torch.no_grad():
                    teacher_states = self.teacher.extractStates(input_ids=input_ids_cot_only, delta=self.config.delta, subset=self.config.subset)
                outputs = self.computeLoss(input_ids=input_w_nocot, teacher_states=teacher_states)
            loss = outputs.loss
            quasi_train_accuracy = outputs.quasi_train_accuracy

            #Stop training early to save resources.
            if quasi_train_accuracy > limit:
                print(f"Quasi accuracy limit reached, stopping training at quasi training accuracy: {quasi_train_accuracy:.6f}.")
                break

            loss.backward() #Calculates gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step() #Subtracts gradients.
            optimizer.zero_grad() #Set gradients to zero.

            #We want 10 updates on steps, accuracy and loss.
            if iteration % math.floor(len(train_dataloader)/10+1) == 0:
                print (f"Step: {iteration}. Loss: {loss:.6f}. Quasi Training Accuracy: {quasi_train_accuracy:.6f}.")
            iteration += 1

            train_losses.append(loss.item())
            train_accs.append(quasi_train_accuracy)

        print (f"\u2714 Evaluating test dataset now...")
        accuracy, loss = self.evaluate(val_dataloader, ctx)

        print (f'\u2192 Loss: {loss:.6f}; Quasi Test Accuracy: {accuracy:.6f}; Quasi Training Accuracy: {quasi_train_accuracy:.6f}.')
        emulator.save_pretrained(os.path.join(train_handler.path+r'\models\thought_emulator'))

        createLossPlot(train_losses) #Plots the loss and accuracy information over batches, so we can gage training performance.
        createAccuracyPlot(train_accs)