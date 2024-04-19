def get_completion(query: str, model, tokenizer, system_prompt='', device='cuda:0', max_new_tokens=512) -> str:
    prompt = f'<bos><start_of_turn>user\n{system_prompt} \n\n{query} <end_of_turn>\n<start_of_turn>model'

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(device)

    generated_ids = model.generate(**inputs, max_length=500, num_return_sequences=1)

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded
