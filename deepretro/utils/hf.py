import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM



def build_chat_prompt(tokenizer, user_prompt, SYS_PROMPT, molecule_smiles=None):

    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt.replace('{target_smiles}', molecule_smiles),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model_name, smiles, user_prompt, top_p, max_tokens, SYS_PROMPT, temperature):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left side padding for generation https://discuss.huggingface.co/t/the-effect-of-padding-side/67188
    model.config.pad_token_id = tokenizer.pad_token_id

    prompt = build_chat_prompt(tokenizer, user_prompt, SYS_PROMPT, smiles)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=1,
            max_new_tokens=max_tokens,
            do_sample=False,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]

    out = outputs[:, prompt_len:]
    out = tokenizer.decode(out[0], skip_special_tokens=True)
    return out