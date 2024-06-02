from datasets import Dataset


def encode_hf(row, embedder):
    """a function that embeds terms in the dataset"""
    row["embeddings"] = embedder.encode(row["terms"])
    return row


def search(embedder, dataset, query: str, k: int = 3):
    """a function that embeds a new query and returns the most probable results"""
    embedded_query = embedder.encode(query)
    scores, retrieved_examples = dataset.get_nearest_examples(
        "embeddings",
        embedded_query,
        k=k,
    )
    return scores, retrieved_examples


def format_prompt_namuwiki(
    prompt: str, retrieved_documents: dict, scores: list, score_threshold: int
):
    """using the retrieved documents we will prompt the model to generate our responses"""
    PROMPT = "## Context\n"
    retrieval_flag = False
    for idx in range(len(scores)):
        if scores[idx] <= score_threshold:  # only include relevant chunks
            text = retrieved_documents["terms"][idx][:1000]
            PROMPT += f"{text}\n"
            retrieval_flag = True
        else:
            break
    if not retrieval_flag:  # no relevant chunks found
        PROMPT += "None"

    PROMPT += f"##Question\n 위에 내용을 바탕으로 다음 질문에 아주 친절하게 답해주세요. 질문: {prompt}"
    return PROMPT


def prep_dataset_namuwiki(filename: str, embedder):
    """cleaning the given text file and generating a huggingface dataset with embeddings"""
    with open(filename, "r") as file:
        corpus = file.read()
    lines = corpus.split("\n")
    result = {}
    for line in lines:
        key = line.split(":")[0]
        val = " ".join(line.split(":")[1:])
        result[key] = val
    data_dict = {
        "terms": lines,
        "descriptions": lines,
    }
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(encode_hf, batched=True, fn_kwargs={"embedder": embedder})
    dataset = dataset.add_faiss_index("embeddings")
    return dataset


def prep_dataset_namuwiki_v2(filename: str, embedder):
    """cleaning the given text file and generating a huggingface dataset with embeddings"""
    with open(filename, "r") as file:
        corpus = file.read()
    lines = corpus.split("\n")
    result = {}
    for line in lines:
        key = line.split(":")[0]
        val = " ".join(line.split(":")[1:])
        result[key] = val
    # data_dict = {"terms": list(result.keys()), "descriptions": list(result.values())}
    data_dict = {
        "terms": list(result.keys()),
        "descriptions": list(result.values()),
    }  # 전체 line 을 검색 대상으로 삼는다.
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(encode_hf, batched=True, fn_kwargs={"embedder": embedder})
    dataset = dataset.add_faiss_index("embeddings")
    return dataset


def prep_dataset(filename: str, embedder):
    """cleaning the given text file and generating a huggingface dataset with embeddings"""
    with open(filename, "r") as file:
        corpus = file.read()
    corpus = corpus.replace("▶링크 바로가기", "").strip()
    corpus = corpus.replace("▶영상 바로가기", "").strip()
    lines = corpus.split("\n")
    result = {}
    key = None
    value = []
    for line in lines:
        if line.startswith("* "):
            if key:
                result[key] = "\n".join(value).strip()
            if ":" in line:
                key, val = line[2:].split(":", 1)
            elif "=" in line:
                key, val = line[2:].split("=", 1)
            value = [val.strip()]
        else:
            value.append(line.strip())
    if key:
        result[key] = "\n".join(value).strip()
    data_dict = {"terms": list(result.keys()), "descriptions": list(result.values())}
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(encode_hf, batched=True, fn_kwargs={"embedder": embedder})
    dataset = dataset.add_faiss_index("embeddings")
    return dataset


def parse_response(result):
    """[Only for batch decoding] extracting the response part of the generated outputs"""
    return (
        result[0].split("assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "")
    )
