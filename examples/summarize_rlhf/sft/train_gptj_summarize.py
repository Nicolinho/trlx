import random

import evaluate
import numpy as np
import torch
from summarize_dataset import TLDRDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from accelerate import Accelerator

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


if __name__ == "__main__":
    output_dir = "gptj-supervised-summarize-checkpoint"
    train_batch_size = 16
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    eval_batch_size = 1
    eval_steps = 500
    max_input_length = 550
    save_steps = 1000
    num_train_epochs = 5
    random.seed(42)

    device_map={'':torch.cuda.current_device()}
#    device_map={'':torch.cuda.current_device()} or device_map={'':torch.xpu.current_device()}
#    device_map="auto"



    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model_name = ['facebook/opt-350m', "EleutherAI/gpt-j-6B"][0]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False,
                                                 load_in_8bit=True, device_map=device_map, torch_dtype = torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # Set up the datasets
    data_path = "CarperAI/openai_summarize_tldr"
    train_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "train",
        max_length=max_input_length,
    )
#    train_dataset = train_dataset.map(num_proc=2)
    dev_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "valid",
        max_length=max_input_length,
    )

    # Set up the metric
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        half_precision_backend=True,
#        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
#        deepspeed="./ds_config_gptj.json",
    )

#    model = prepare_model_for_int8_training(model) # this gives an error in distributed setting

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    print(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
