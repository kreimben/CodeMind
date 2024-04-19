def get_completion(query: str, pipeline, system_prompt='') -> str:
    messages = [
        {
            "role": "user",
            "content": f"{system_prompt}\n\n{query}",
        }
    ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True
    )[0]['generated_text']
    return outputs
