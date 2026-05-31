import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True   # ✅ ensures input is [batch, seq, embed]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        # embeddings + positional encoding
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # transformer encoder
        x = self.transformer_encoder(x)
        
        # use first token ([CLS]-like) representation
        x = x[:, 0, :]
        
        # classification logits
        return self.fc(x)


model = TransformerClassifier(
    vocab_size=75306,
    embed_dim=128,
    num_heads=4,
    hidden_dim=256,
    num_layers=2,
    num_classes=6,
    max_len=64
)
class Tokenizer:
    def __init__(self, vocab, max_len=64):
        self.vocab = vocab
        self.max_len = max_len
        # cache special tokens for speed
        self.pad = vocab["<PAD>"]
        self.unk = vocab["<UNK>"]
        self.cls = vocab["<CLS>"]
        self.sep = vocab["<SEP>"]

    def __call__(self, text):
        tokens = text.split()
        ids = [self.vocab.get(tok, self.unk) for tok in tokens]
        ids = [self.cls] + ids[:self.max_len-2] + [self.sep]
        # pad
        if len(ids) < self.max_len:
            ids += [self.pad] * (self.max_len - len(ids))
        return ids

state_dict = torch.load("best_scratch_transformer.pt", map_location="cpu")

model.load_state_dict(state_dict)

vocab = torch.load ("vocab.dic")

tokenizer = Tokenizer (vocab) 
def prediction (text ) : 

    transform = {
    0 :  "sadness" , 
    1 : "joy" , 
    2 : " love" , 
    3 : "anger" , 
    4 : "fear" , 
    5 : "surprise"

}

    ids = tokenizer(text)  # using your Tokenizer class
    input_ids = torch.tensor([ids]).to("cpu")  # shape: [batch_size, seq_len]
    outputs = model(input_ids)  # forward pass
    pred = torch.argmax(outputs, dim=1)   # get predicted class index

    return transform[pred.item()]

print (prediction ("im very happy" ))