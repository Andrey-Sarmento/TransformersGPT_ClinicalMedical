# Package
import torch
import random

# Vocabulary, Encoder and Decoder
with open("vocabulary.txt", encoding="utf-8") as f:
    vocabulary = [x.replace(r"\n", "\n") for x in f.read().splitlines()]
    voc_index = {char: idx for idx, char in enumerate(vocabulary)}


def Encoder(text, vocabulary_idxs = voc_index):
    """Converte texto em números, com base no vocabulário."""
    tokens = list(text)
    idxs = [vocabulary_idxs.get(k, vocabulary_idxs["<unk>"]) for k in tokens]
    return idxs


def Decoder(idxs, vocabulary_idxs = voc_index):
    """Converte uma lista de índices em texto, com base no vocabulário."""
    idxs = [i if 0 <= i < len(vocabulary) else vocabulary.index("<unk>") for i in idxs]
    return ''.join([list(vocabulary_idxs.keys())[i] for i in idxs])


# Get corpus chunks
def get_chunks(corpus, chunk_size=128):
    start = random.randint(0, len(corpus) - chunk_size)
    chunk = corpus[start:start + chunk_size]
    idxs = Encoder(chunk)
    return {"chunk": chunk, "idxs": idxs}


# Batch
def Batch(corpus, batch_size, block_size, device="cpu", train=True):
    corpus = corpus[:int(len(corpus)*0.9)] if train else corpus[int(len(corpus)*0.9):]
    x = torch.empty((batch_size, block_size), dtype=torch.long)
    y = torch.empty((batch_size, block_size), dtype=torch.long)
    
    for b in range(batch_size):
        chunk = get_chunks(corpus, chunk_size=block_size+1)
        x[b] = torch.tensor(chunk["idxs"][:-1])
        y[b] = torch.tensor(chunk["idxs"][1:])

    return {"X": x.to(device), "Y": y.to(device)}


# Generate text
def generate_text(
        model,
        prompt,
        max_new_tokens=200,
        block_size=128,
        device="cpu",
        top_k=10,
    ):

    print(prompt, end="", flush=True)
    context_ids = Encoder(prompt)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Recorte do contexto e PAD à direita
            ctx = context_ids[-block_size:] if len(context_ids) > block_size else context_ids
            x = torch.full((1, block_size), 0, dtype=torch.long, device=device)
            x[0, :len(ctx)] = torch.tensor(ctx, dtype=torch.long, device=device)

            # Forward
            logits = model(x)
            logits_step = logits[:, len(ctx)-1, :]

            # Top-k
            values, indices = torch.topk(logits_step, k=top_k, dim=-1)
            logits_filtered = torch.full_like(logits_step, float('-inf'))
            logits_filtered.scatter_(1, indices, values)
            probs = torch.softmax(logits_filtered, dim=-1)

            # Amostragem
            next_id = int(torch.multinomial(probs, num_samples=1))
            context_ids.append(next_id)
            print(Decoder([next_id]), end="", flush=True)

    return Decoder(context_ids)
