import argparse
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

def main(args):
    # Load formatted data
    with open(args.data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # Extract all unique words from the dataset
    unique_words = set()
    word_splitter = re.compile(r'\w+|[^\w\s]+')

    for conv in conversations:
        input_text = conv['input']
        response_text = conv['response']
        input_words = word_splitter.findall(input_text)
        response_words = word_splitter.findall(response_text)
        unique_words.update(input_words)
        unique_words.update(response_words)

    # Create custom tokens
    custom_tokens = {word: word.lower() for word in unique_words}

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(list(custom_tokens.values()))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # Load the model and resize embeddings
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Prepare the dataset
    cleaned_data = []
    for conv in conversations:
        if conv["input"] and conv["response"]:
            cleaned_data.append({"input": conv["input"], "response": conv["response"]})

    data = {"text": []}
    for conv in cleaned_data:
        input_tokens = " ".join([custom_tokens.get(word, word) for word in word_splitter.findall(conv['input'])])
        response_tokens = " ".join([custom_tokens.get(word, word) for word in word_splitter.findall(conv['response'])])
        combined_text = f"<|startoftext|>{input_tokens} {response_tokens}<|endoftext|>"
        data["text"].append(combined_text)

    dataset = Dataset.from_dict(data)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(lambda examples: {"labels": examples["input_ids"]})

    # Split dataset
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_drop_last=True,
        fp16=True,
        gradient_accumulation_steps=8,
        ddp_find_unused_parameters=False,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train and save
    trainer.train()
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print(f"Model and tokenizer saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium", help="Pretrained model name.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory for training outputs.")
    parser.add_argument("--save_path", type=str, default="./trained_model", help="Path to save the trained model.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size.")
    parser.add_argument("--eval_steps", type=int, default=25, help="Evaluation steps.")
    parser.add_argument("--save_steps", type=int, default=25, help="Save steps.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    args = parser.parse_args()
    main(args)
