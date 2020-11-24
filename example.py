import datetime
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AdamW,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from the_wizard_express.datasets.dataset import QADataset

# Set the seed value all over the place to make this reproducible.
seed_val = 10
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

pretrainedModelName = "sshleifer/distilbart-xsum-6-6"

# Training and validation loss, validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Unpack the dataset - For simplified datasets, split the datasets manually
QADataset.write(
    os.path.realpath("../v1.0-simplified_simplified-nq-train.jsonl"),
    os.path.realpath("../training/"),
)

tokenizer = AutoTokenizer.from_pretrained(pretrainedModelName)
training_dataloader = QADataset(os.path.realpath("../training/"), tokenizer)
validation_dataloader = QADataset(os.path.realpath("../validation/"), tokenizer)


# training_dataloader = DataLoader(QADataset(initializeNQDataset(os.path.realpath("../training/xx-1-training.jsonl"), tokenizer)), num_workers=1, batch_size=1)
# validation_dataloader = DataLoader(QADataset(initializeNQDataset(os.path.realpath("../validation/xx-1-validation.jsonl"), tokenizer)), num_workers=1, batch_size=1)

readerModel = AutoModelForQuestionAnswering.from_pretrained(pretrainedModelName)

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# Adam with 'Weight Decay fix', see: https://arxiv.org/pdf/1711.05101.pdf
optimizer = AdamW(readerModel.parameters(), lr=2e-5, eps=1e-8)

# Number of training epochs. The BERT authors recommend between 2 and 4.
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=9891 * 4
)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("\n======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
    print("Training...")

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    readerModel.train()

    training_dataloader_len = len(training_dataloader)
    # For each batch of training data...
    for step, batch in enumerate(training_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

            # Report progress.
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, training_dataloader_len, elapsed
                )
            )

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: question and context inputs
        #   [1]: attention masks
        #   [2]: start position of answer in context
        #   [3]: end position of answer in context
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        start_positions = batch[2].to(device)
        end_positions = batch[3].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        readerModel.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        outputs = readerModel(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(readerModel.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / training_dataloader_len

    # Measure how long this epoch took.
    training_time = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

    print("\n  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("\nRunning Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    readerModel.eval()

    # Tracking variables
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: start position of answer in context
        #   [3]: end position of answer in context
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        start_positions = batch[2].to(device)
        end_positions = batch[3].to(device)

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
            outputs = readerModel(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

        loss = outputs[0]
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = start_positions.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        # TODO
    validation_dataloader_len = len(validation_dataloader)
    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / validation_dataloader_len
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / validation_dataloader_len

    # Measure how long the validation run took.
    validation_time = str(datetime.timedelta(seconds=int(round((time.time() - t0)))))

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

print("\nTraining complete!")

print(
    "Total training took {:} (h:mm:ss)".format(
        str(datetime.timedelta(seconds=int(round((time.time() - total_t0)))))
    )
)