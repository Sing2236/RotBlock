#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 09:22:09 2024

@author: mant
Using a tutorial by Chris McCormick
"""
#imports
import torch
import random
from random import randrange #used for testing
from torch import nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
import time, datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
import os

#first check if we have a gpu and use that, otherwise use the cpu
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using GPU: ', torch.cuda.get_device_name(0))
else:
    print('No GPU detected. Using CPU.')
    device = torch.device("cpu")


#add in data set as dataframe for title processing
#emojiframe to add emojis to tokenizer embedding
dataframe = pd.read_csv("./brainrot_data/brainrot_dataset.tsv", delimiter='\t', header=None, names=['title', 'score'])
emojiframe = pd.read_csv("./brainrot_data/emojis_for_tokenization_a.tsv", delimiter = '\t', header=None, names=['emoji'])

#getting lists of titles and their scores for training and emojis for tokenization
emojis = emojiframe.emoji.values
titles = dataframe.title.values
scores = dataframe.score.values

#load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#loading the model and the model weights for updating
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels = 2,
                                                      output_attentions = False,
                                                      output_hidden_states = False)
weights = model.bert.embeddings.word_embeddings.weight.data

'''
#test prints for info
print(weights.shape)
print(len(emojis))
print(len(titles))
print(len(scores))
'''
#updating weight dimensions aand embeddings in order to be big enough for the emojis
new_weights = torch.cat((weights, weights[101:3399]), 0)
new_emb = nn.Embedding.from_pretrained(new_weights, padding_idx=0, freeze=False)
model.bert.embeddings.word_embeddings = new_emb

'''
#test prints
print(model.embeddings)
print(type(emojis[0]))
'''
#adding the emojis to the tokenizer
for em in range(0, len(emojis)):
 tokenizer.add_tokens(emojis[em])

'''
#Test block for test tokenization
randnum = randrange(len(titles))
print('Original: ', titles[randnum])
print('Tokenized: ', tokenizer.tokenize(titles[randnum]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(titles[randnum])))
'''

'''
#Test block to try and get max sentence length bert will expect
max_len = 0
for ti in titles:
    input_ids = tokenizer.encode(ti, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
print('Max sentence length: ', max_len)
'''
#title tokenizing and mapping tokens to word ids
input_ids = []
attention_masks = []

'''
for every title we:
1. tokenize the sentence
2. add in our special characters
3. map tokens to their ids
4. pad/truncate using max length
5. create attention masks for the pad tokens
'''
for ti in titles:
    encoded_dict = tokenizer.encode_plus(
        ti, #title to encode
        add_special_tokens=True, #funny special tokens
        max_length=64,
        truncation=True,
        padding='max_length', #pad/truncate steps
        return_attention_mask=True, #construct attention masks
        return_tensors = 'pt') #get back pytorch tensors
    
    #adding encoded sentences/attention masks to their lists
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    
#convert those lists into tensors baybeee
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
scores = torch.tensor(scores)

'''
#Test block to verify tensor sizes match
print(input_ids.shape)
print(attention_masks.shape)
print(scores.shape)
'''
#combining the training inputs into a data set
dataset = TensorDataset(input_ids, attention_masks, scores)    
train_size = int(0.9 * len(dataset)) #90-10 train-val split calculations
val_size = len(dataset) - train_size

#dividing the dataset by randomly selecting samples
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

#init batch_size and epochs variables for training
batch_size = 32
epochs = 3

#creating the dataloader for our train/val sets. taking training samples in random order
train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset),
    batch_size = batch_size)

#doing the same as above but since validation order don't matter we just read them
validation_dataloader = DataLoader(
    val_dataset,
    sampler = SequentialSampler(val_dataset),
    batch_size = batch_size)

#use the listed device please and thank you~!
model.to(device)

#using AdamW which is a huggingface class as opposed to pytorch itself.
#i'm reading the W stands for Weight Decay fix but that could be wrong
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, #args.learning_rate - default is 5e-5
                  eps = 1e-8 #args.adam_epsilon - default is le-8
                  )
 
#total number of training steps is num of batches * num of epochs
total_steps = len(train_dataloader) * epochs
#creates the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps
                                            )

#function to calculate the accuracy of our predictions vs score
def flat_accuracy(preds, scores):
    pred_flat = np.argmax(preds, axis=1).flatten()
    scores_flat = scores.flatten()
    return np.sum(pred_flat == scores_flat) / len(scores_flat)

#helpful function to format elapsed times
#takes a time in seconds and returns string hh:mm:ss
def format_time(elapsed):
    #round to nearest second
    elapsed_rounded = int(round(elapsed))
    #format
    return str(datetime.timedelta(seconds=elapsed_rounded))

#This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

#setting seed val everywhere for reproducibility
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

#storing a quantity of stuff like training and validation loss, validation accuracy, timings, etc
training_stats = []
#training time for the whole run
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the current device using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: scores 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_scores = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_scores,
                             return_dict = False)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the current device using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: scores 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_scores = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_scores,
                                   return_dict = False)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        score_ids = b_scores.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, score_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

#Training Process Sumarry

pd.set_option('display.precision', 2)
df_stats = pd.DataFrame(data=training_stats)
df_stats = df_stats.set_index('epoch')
print(df_stats)

#Plotting validation loss for visual representations of accuracy

# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()

#Preparing test set
testframe = pd.read_csv("./brainrot_data/brainrot_testset.tsv", delimiter='\t', header=None, names=['title', 'score'])
t_titles = testframe.title.values
t_scores = testframe.score.values

t_input_ids = []
t_attention_masks = []

for ti in t_titles:
    t_encoded_dict=tokenizer.encode_plus(
        ti,
        add_special_tokens=True,
        max_length = 64,
        padding = 'max_length',
        return_attention_mask = True,
        return_tensors = 'pt')
    
    t_input_ids.append(t_encoded_dict['input_ids'])
    t_attention_masks.append(t_encoded_dict['attention_mask'])

t_input_ids = torch.cat(t_input_ids, dim=0)
t_attention_masks = torch.cat(t_attention_masks, dim=0)
t_scores = torch.tensor(t_scores)

t_batch_size = 32

prediction_data = TensorDataset(t_input_ids, t_attention_masks, t_scores)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set
print('Predicting labels for {:,} test sentences...'.format(len(t_input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_scores = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to current device
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_t_input_ids, b_t_input_mask, b_t_scores = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_t_input_ids, token_type_ids=None, 
                      attention_mask=b_t_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  score_ids = b_t_scores.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_scores.append(score_ids)

print('    DONE.')

print('Positive samples: %d of %d (%.2f%%)' % (testframe.score.sum(), len(testframe.score), (testframe.score.sum() / len(testframe.score) * 100.0)))

#accuracy evaluations
matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_scores)):
  
  # The predictions for this batch are a 2-column ndarray (one column for "0" 
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_scores_i = np.argmax(predictions[i], axis=1).flatten()
  
  # Calculate and store the coef for this batch.  
  matthews = matthews_corrcoef(true_scores[i], pred_scores_i)                
  matthews_set.append(matthews)
  
# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_scores = np.concatenate(true_scores, axis=0)

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_scores, flat_predictions)

print('Total MCC: %.3f' % mcc)

output_dir = './model_save/'
state_dict_output_dir = './model_save/state_dict.pt'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
#model.save_pretrained(output_dir, safe_serialization=False)
model.save_pretrained(output_dir)
torch.save(model.state_dict(), state_dict_output_dir)
tokenizer.save_pretrained(output_dir)


# Good practice: save your training arguments together with the trained model
#torch.save(args, os.path.join(output_dir, 'training_args.bin'))

