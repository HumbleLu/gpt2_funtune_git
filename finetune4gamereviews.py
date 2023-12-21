import os, sys
import time
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

# Load the pre-trained DistilGPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('distilgpt2')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')

# set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load steam review dataset
df_reviews = pd.read_csv('dataset.csv')

# sample only 1000 reviews for training
train_data = df_reviews.sample(1000)
train_data.shape

# modify training text
train_txt_list = ['[GAME]' + str(train_data['app_name'][ind]) + 
                  '[SCORE]' + str(train_data['review_score'][ind]) + 
                  '[REVIEW]' + str(train_data['review_text'][ind]) 
                  for ind in train_data.index] 

# dataset loader for fine-tuning
class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="distilgpt2", max_length=1000):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx] 

train_dataset = GPT2Dataset(train_txt_list, tokenizer, max_length = 500)

# choose distilgpt2 as the tuned model
configuration = GPT2Config.from_pretrained('distilgpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("distilgpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))
batch_size = 20

# training set up
# some parameters I cooked up that work reasonably well
epochs = 5
learning_rate = 5e-4

train_dataloader = DataLoader(
    train_dataset,  # The training samples.
    sampler = RandomSampler(train_dataset), # Select batches randomly
    batch_size = batch_size # Trains with this batch size.
)

# this produces sample output every 100 steps
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate)

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

total_t0 = time.time()

training_stats = []

model = model.to(device)

for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    # ========================================

    print('Training...')

    t0 = time.time()
    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids,
                        labels=b_labels, 
                        attention_mask = b_masks,
                        token_type_ids=None)

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

            model.eval()

            sample_outputs = model.generate(
                                    do_sample=True,   
                                    top_k = 10, 
                                    max_length = 200,
                                    top_p = 0.90, 
                                    num_return_sequences=1
                                )
            for i, sample_output in enumerate(sample_outputs):
                  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            
            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("Average training loss: {0:.2f}".format(avg_train_loss))
    print("Training epoch took: {:}".format(training_time))

print("")
print("Training complete!")

# save fune-tuned model
output_dir = 'gamereveiw_distillgpt2'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)