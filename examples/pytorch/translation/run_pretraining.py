from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import torch

#train tokenizer
#---------------------------------------------------------------
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
# from tokenizers import SentencePieceBPETokenizer

# tokenizer = SentencePieceBPETokenizer()
# tokenizer.train_from_iterator(
#     text,
#     vocab_size=30_000,
#     min_frequency=5,
#     show_progress=True,
#     limit_alphabet=500,
# )

# Customize training
# t5_tokenizer = T5Tokenizer(tokenizer)

# Save files to disk
# t5_tokenizer.save_model(".", "t5_tokenizer")

tokenizer = T5Tokenizer.from_pretrained("t5-small")

from datasets import load_dataset

dataset = load_dataset("iwslt2017")

#create T5 config

conf = T5Config(num_layers=6, num_decoder_layers=6)

model = T5ForConditionalGeneration(conf)


#
# encoding = tokenizer(
#      [task_prefix + sequence for sequence in input_sequences],
#      padding="longest",
#      max_length=max_source_length,
#      truncation=True,
#      return_tensors="pt",
# )
#
# input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
#
# # encode the targets
# target_encoding = tokenizer(
#     [output_sequence_1, output_sequence_2],
#     padding="longest",
#     max_length=max_target_length,
#     truncation=True,
#     return_tensors="pt",
# )
# labels = target_encoding.input_ids
#
# # replace padding token id's of the labels by -100 so it's ignored by the loss
# labels[labels == tokenizer.pad_token_id] = -100
#
# # forward pass
# loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
# loss.item()
