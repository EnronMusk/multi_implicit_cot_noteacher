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
    def __init__(self, config : ThoughtEmulatorConfig):
        super().__init__()
        self.config = config
        self.base_model = GPT2LMHeadImplicitModel.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        num_layers = len(self.base_model.transformer.h)
        hidden_size = self.base_model.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_ids):
        outputs = self.base_model.forward(input_ids=input_ids)
        return outputs

    def computeLoss(self, input_ids, labels):
        #import pdb; pdb.set_trace()
        outputs = self.forward(input_ids=input_ids)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
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

    def __generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True) -> list:
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]

        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)

        batch_size = input_ids.shape[0]
        beam_output = []

        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        generation_config.eos_token_id = -1
        logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
        stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])

        if sep_positions.eq(sep_positions[0]).all():
            input_ids = input_ids[:, :sep_positions[0]+1]
            beam_output = self.base_model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=1,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
            beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i:i+1]
                sep_positions_i = sep_positions[i:i+1]
                input_ids_i = input_ids_i[:, :sep_positions_i+1]
                beam_output_i = self.base_model.generate(
                    input_ids=input_ids_i,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
                beam_output.append(beam_output_i)

        return beam_output

    def save_pretrained(self, save_directory) -> None:
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))

    @torch.no_grad() #Freeze gradients.
    def evaluate(self, dataloader : DataLoader, ctx):
        '''
        Calculates accuracy metrics on test data and can also generate predictions.
        '''
        self.eval() #Freeze loss function, gradients etc.

        total_instances = 0
        total_tokens = 0
        total_correct_tokens = 0
        total_correct = 0
        total_loss = 0
        sub_iteration = 0

        for batch in tqdm.tqdm(dataloader):
            labels  = batch['labels_purecot_i'].to(device)
            input_ids_all  = batch['input_ids_purecot_i'].to(device)
            # Remove answer part
            sep_positions = get_sep_position(input_ids_all, self.tokenizer.eos_token_id)
            input_ids = input_ids_all[:, :sep_positions.max()+1]
            batch_size = input_ids.shape[0]
            with ctx:
                outputs = self.computeLoss(input_ids=input_ids_all, labels=labels)
            total_loss += outputs.total_loss.item()
            total_correct_tokens += outputs.total_correct.item()
            total_tokens += outputs.total_tokens
            total_instances += batch_size

            # Generate
            beam_output = self.__generate(
                input_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens,
            )
            for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
                sub_iteration += 1
                sep_position = sep_positions[i].item()
                tgt = input_ids_all_i[sep_position+1:]
                tgt_text = self.tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extractAnswer(tgt_text)
                pred_text = self.tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
                pred_ans = extractAnswer(pred_text)
                if ans == pred_ans:
                    total_correct += 1

                if sub_iteration >= len(dataloader.dataset)-2: # to limit spam of prediction examples.
                    print (f'Input: {self.tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
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

        teacher = self.to(device).to(ptdtype)
        # Load data
        tokenizer = teacher.tokenizer
        collate_fn = CoTDataCollator(tokenizer)
        custom_data_handler = DataLoader(custom_data_handler, batch_size = self.config.batch_size, collate_fn = collate_fn, shuffle=False)

        self.evaluate(custom_data_handler, ctx)





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

        emulator.train() #Put model in training mode

        train_losses = []

        train_accs = []

        # Train
        iteration = 0
        for batch in tqdm.tqdm(train_dataloader):
            emulator.train()
        

            pure_cot  = batch['labels_purecot_i'].to(device)
            input_w_nocot = batch['input_ids_purecot_i'].to(device)
            
            with ctx:
                outputs = self.computeLoss(input_ids=input_w_nocot, labels=pure_cot)
            loss = outputs.loss
            token_accuracy = outputs.token_accuracy.item()

            #Stop training early to save resources.
            if token_accuracy > limit:
                print(f"Accuracy limit reached, stopping training at training accuracy: {token_accuracy:.6f}.")
                break

            loss.backward() #Calculates graidents
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.max_grad_norm)
            optimizer.step() #Subtracts gradients.
            optimizer.zero_grad() #Set gradients to zero.

            ppl = loss.exp().item()

            #We want 10 updates on steps, accuracy and loss.
            if iteration % math.floor(len(train_dataloader)/10+1) == 0:
                print (f"Step: {iteration}. PPL: {ppl:.6f}. Training Accuracy: {token_accuracy:.6f}")
            iteration += 1

            train_losses.append(loss.item())
            train_accs.append(token_accuracy)

        print (f"\u2714 Evaluating test dataset now...")
        accuracy, token_accuracy, ppl = self.evaluate(val_dataloader, ctx)

        print (f'\u2192 PPL: {ppl:.6f}; Test Accuracy: {accuracy:.6f}; Training Accuracy: {token_accuracy:.6f}.')
        emulator.save_pretrained(os.path.join(train_handler.path+r'\models\thought_emulator'))

        createLossPlot(train_losses) #Plots the loss and accuracy information over batches, so we can gage training performance.
        createAccuracyPlot(train_accs)