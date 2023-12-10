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

from source.configurations import MindReadingEmulatorConfig
from source.utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor, createAccuracyPlot, createLossPlot
from source.gpt2_implicit import GPT2LMHeadImplicitModel

from source.teacher import Teacher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

class MindReadingEmulator(nn.Module):
    def __init__(self, config : MindReadingEmulatorConfig, teacher : Teacher):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        #We turn each layer into a verticle set of layers.
        self.verticle_model = nn.ModuleList([nn.Sequential(
             nn.Linear(hidden_size, 4*hidden_size),
             nn.ReLU(),
             nn.Linear(4*hidden_size, hidden_size),
             ) for _ in range(num_layers)])

        self.teacher = teacher

    def forward(self, input_ids, positions_to_substitute, teacher_states, output_hidden_states=False):
        outputs = self.base_model.forward(mode='forward_student', \
                input_ids=input_ids, \
                positions_to_substitute=positions_to_substitute, \
                states_to_substitute=teacher_states, \
                output_hidden_states=output_hidden_states, \
                )
        return outputs

    def computeLoss(self, input_ids, labels, teacher_states):
        #import pdb; pdb.set_trace()
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        # First, project teacher states
        teacher_states = [self.verticle_model[l](teacher_states[l]) for l in range(len(teacher_states))]

        # Forward while substituting teacher states
        outputs = self.forward(input_ids, sep_positions, teacher_states)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs

    def save_pretrained(self, save_directory) -> None:
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

    def generate(self, input_ids, teacher_states, max_new_tokens=512, num_beams=1):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]
        beam_output = []
        # First, project teacher states
        teacher_states = [self.verticle_model[l](teacher_states[l]) for l in range(len(teacher_states))]
        for i in range(batch_size):
            input_ids_i = input_ids[i:i+1]
            sep_positions_i = sep_positions[i:i+1]
            input_ids_i = input_ids_i[:, :sep_positions_i+1]
            beam_output_i = self.base_model.generate(
                input_ids=input_ids_i,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                positions_to_substitute=sep_positions_i.repeat_interleave(num_beams, dim=0),
                states_to_substitute=[z[i:i+1].repeat_interleave(num_beams, dim=0) for z in teacher_states],
                mode='forward_student',
            )
            beam_output.append(beam_output_i)
        return beam_output
    
    @torch.no_grad() #Freeze gradients.
    def evaluate(self, dataloader : DataLoader, ctx,):
        '''
        Calculates accuracy metrics on test data.
        '''
        self.eval() #Freeze loss function, gradients etc.

        total_instances = 0
        total_tokens = 0
        total_correct = 0
        total_correct_tokens = 0
        total_loss = 0
        
        sub_iteration = 0
        for batch in tqdm.tqdm(dataloader):
            input_ids_all = batch['input_ids_all'].to(device)
            input_ids_nocot = batch['input_ids_nocot'].to(device)
            labels_nocot = batch['labels_nocot'].to(device)

            batch_size = input_ids_nocot.shape[0]
            with ctx:
                teacher_states = self.teacher.extractStates(input_ids=input_ids_all, delta=self.config.delta, subset=self.config.subset)
                outputs = self.computeLoss(input_ids=input_ids_nocot, labels=labels_nocot, teacher_states=teacher_states)
                loss = outputs.loss
                token_accuracy = outputs.token_accuracy.item()
            total_loss += outputs.total_loss.item()
            total_correct_tokens += outputs.total_correct.item()
            total_tokens += outputs.total_tokens
            total_instances += batch_size

            # Generate
            with ctx:
                beam_output = self.generate(
                    input_ids=input_ids_nocot,
                    teacher_states=teacher_states,
                    max_new_tokens=self.config.max_new_tokens,
                )

            # Evaluate
            sep_positions = get_sep_position(input_ids_all, self.tokenizer.eos_token_id)
            for i, (input_ids_all_i, beam_output_i) in enumerate(zip(labels_nocot, beam_output)):
                sub_iteration +=1
                sep_position = sep_positions[i].item()
                tgt = input_ids_all_i[sep_position+1:]
                tgt_text = self.tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extractAnswer(tgt_text)
                pred_text = self.tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
                pred_ans = extractAnswer(pred_text)
                if ans == pred_ans:
                    total_correct += 1
                if sub_iteration >= len(dataloader.dataset)-2: # to limit spam of prediction examples.
                    print (f'Input Layer 1, Attention Head 1, first 9 states:')
                    print(np.round(teacher_states[0][0][:9].cpu().numpy(), decimals=4))
                    print (f'Target: {tgt_text}')
                    print (f'Predicted: {pred_text}')
                    print ('')
        accuracy = total_correct / total_instances
        token_accuracy = total_correct_tokens / total_tokens
        loss = total_loss / total_tokens
        ppl = math.exp(loss)
        return accuracy, token_accuracy, ppl
    
    def predict(self, custom_data_handler : DatasetHandler) -> None:
        '''
        Used for custom test cases for fun. You can create custom test cases using the generateDataset using a DatasetHandler.
        '''

        dtype = 'float32'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

        # Load data
        tokenizer = self.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        custom_data_handler = DataLoader(custom_data_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        self.evaluate(custom_data_handler, ctx)

    def trainModel(self, train_handler : DatasetHandler, test_handler : DatasetHandler, limit : float) -> None:
        '''
        Trains the model and automatically evaluates. 
        @limit hard caps the desired accuracy to stop training early if the threshold is meet.
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


            input_ids_all = batch['input_ids_all'].to(device)
            input_w_nocot = batch['input_ids_nocot'].to(device)
            labels_w_nocot = batch['labels_nocot'].to(device)
            with ctx:
                with torch.no_grad():
                    teacher_states = self.teacher.extractStates(input_ids=input_ids_all, delta=self.config.delta, subset=self.config.subset)
                outputs = self.computeLoss(input_ids=input_w_nocot, labels=labels_w_nocot, teacher_states=teacher_states)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()

            #Stop training early to save resources.
            if token_accuracy > limit:
                print(f"Training accuracy limit reached, stopping training at training accuracy: {token_accuracy:.6f}.")
                break

            loss.backward() #Calculates gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step() #Subtracts gradients.
            optimizer.zero_grad() #Set gradients to zero.
            ppl = loss.exp().item()

            #We want 10 updates on steps, accuracy and loss.
            if iteration % math.floor(len(train_dataloader)/10+1) == 0:
                print (f"Step: {iteration}. CrossEntropyLoss: {loss:.6f}. Training Accuracy: {token_accuracy:.6f}.")
            iteration += 1

            train_losses.append(loss.item())
            train_accs.append(token_accuracy)

        print (f"\u2714 Evaluating test dataset now...")
        accuracy, token_accuracy, ppl = self.evaluate(val_dataloader, ctx)

        print (f'\u2192 Perplexitity: {ppl:.6f}; Test Accuracy: {accuracy:.6f}; Training Accuracy: {token_accuracy:.6f}.')
        emulator.save_pretrained(os.path.join(train_handler.path+r'\models\mindreading_emulator'))

        createLossPlot(train_losses) #Plots the loss and accuracy information over batches, so we can gage training performance.
        createAccuracyPlot(train_accs)