import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def main(args):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|user|>", "<|bot|>"]})
    model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    def chat_with_bot(input_text, model, tokenizer, max_length=128):
        formatted_input = f"<|user|> {input_text.strip()} <|bot|>"
        input_ids = tokenizer.encode(formatted_input, return_tensors="pt")
        response_ids = model.generate(
            input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )
        decoded_output = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return decoded_output.split("<|bot|>")[-1].strip()

    # Chat loop
    print("Chatbot is ready! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = chat_with_bot(user_input, model, tokenizer)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    args = parser.parse_args()
    main(args)
