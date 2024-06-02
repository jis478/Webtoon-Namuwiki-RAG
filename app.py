import os
from threading import Thread
import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from sentence_transformers import SentenceTransformer

from config import (
    server_port,
    model_id,
    token,
    filename,
    sim_score_threshold,
    k,
    SYS_PROMPT,
    max_new_tokens,
    top_p,
    temperature,
)
from utils import (
    search,
    format_prompt_namuwiki,
    prep_dataset_namuwiki,
)


# model loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=bnb_config,
    # token=token,
)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# dataset preparation
dataset = prep_dataset_namuwiki(filename, embedder)


# chat
def chat_namuwiki(query, history):
    scores, retrieved_documents = search(embedder, dataset, query, k)
    user_prompt = format_prompt_namuwiki(
        query, retrieved_documents, scores, sim_score_threshold
    )
    messages = [
        {
            "role": "system",
            "content": SYS_PROMPT,
        },
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(  # streaming enabled
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        eos_token_id=terminators,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        if ("답변" or "##") in text:
            continue
        outputs.append(text)
        yield "".join(outputs)


# gradio
demo = gr.ChatInterface(
    fn=chat_namuwiki,
    chatbot=gr.Chatbot(
        show_label=True,
        show_share_button=True,
        show_copy_button=True,
        likeable=True,
        layout="bubble",
        bubble_full_width=False,
    ),
    theme="Soft",
    examples=[["DPU가 뭐야?"], ["범죄도시4 마동석은 누구야?"]],
    title="웹툰 Namuwiki RAG",
)
demo.launch(debug=True, server_port=server_port)
