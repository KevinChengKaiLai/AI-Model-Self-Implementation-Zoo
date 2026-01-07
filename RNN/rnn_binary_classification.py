# rnn_binary_classification.py
"""This is an example for dealing with imbalance data"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import wandb
from LSTM import LSTM_classification
from argparse import ArgumentParser

DATAPATH = 'dataset/emails.csv'

def parse_args():
    parser = ArgumentParser(description="Train spam classifier with RNN or LSTM")
    
    # Model architecture
    parser.add_argument("--architecture", type=str, default="RNN", 
                       choices=["RNN", "LSTM"],
                       help="Model architecture to use")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                       help="Learning rate for optimizer")
    
    # Model hyperparameters
    parser.add_argument("--hidden_size", type=int, default=256,
                       help="Hidden size of RNN/LSTM")
    parser.add_argument("--embedding_dim", type=int, default=64,
                       help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.2,
                       help="Dropout probability")
    parser.add_argument("--max_length", type=int, default=300,
                       help="Maximum sequence length")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable wandb logging")
    
    return parser.parse_args()

class CharTokenizer:
    def __init__(self):
        
        self.special_tokens = ['<PAD>', '<UNK>']
        import string
        chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + ' '
        self.vocab = self.special_tokens + list(chars)

        self.char2idx = {char:idx for idx,char in enumerate(self.vocab)}
        self.idx2char = {idx:char for idx,char in enumerate(self.vocab)}

        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.char2idx[self.pad_token]
        self.unk_idx = self.char2idx[self.unk_token]

    def encode(self, text):
        return [self.char2idx.get(char,self.unk_idx) for char in text]

    def decode(self, indices):
        return "".join([self.idx2char[idx] for idx in indices])
    
    def __len__(self):
        return len(self.vocab)



class SpamDataset(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer: CharTokenizer, max_length):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        # How many samples?
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. Get email text and label
        text = self.df.iloc[idx]['text']
        label = self.df.iloc[idx]['spam']

        # 2. Tokenize text
        text_token =  self.tokenizer.encode(text)

        n = len(text_token)
        # 3. Handle padding/truncation
        if n > self.max_length:
            # truncation
            text_token = text_token[:self.max_length]
        if n < self.max_length:
            # pad
            num_to_pad = self.max_length - n
            text_token += [self.tokenizer.pad_idx] * num_to_pad

        # 4. Convert to tensors
        input_tensor = torch.tensor(text_token)
        label_tensor = torch.tensor(label
                                    )
        # 5. Return (input_tensor, label_tensor)
        return (input_tensor, label_tensor)

class RNN_classification(nn.Module):
    # h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
    # x_t: input at timestep t, shape (batch, input_size)
    # h_{t-1}: previous hidden state, shape (batch, hidden_size)
    # W_ih: input-to-hidden weights, shape (input_size, hidden_size)
    # W_hh: hidden-to-hidden weights, shape (hidden_size, hidden_size)

    def __init__(self, in_dim, hidden_size):
        super(RNN_classification, self).__init__()

        self.in_dim = in_dim
        self.hidden_size = hidden_size

        self.W_ih = nn.Linear(in_dim,hidden_size)
        self.W_hh = nn.Linear(hidden_size,hidden_size)
        self.tanh = nn.Tanh()
    
    def forward(self,x:torch.Tensor):
        # x (batch_size, seq_len, in_dim)
        batch_size, seq_len, _ = x.shape

        # init h0
        h_t_minus1 = torch.zeros((batch_size,self.hidden_size))

        for t in range(seq_len):
            x_t = x[:,t,:]
            h_t = self.tanh(self.W_ih(x_t) + self.W_hh(h_t_minus1))
            h_t_minus1 = h_t

        return h_t_minus1
    

class SpamClassifier(nn.Module):
    ''' rnn classifier'''
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(SpamClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = RNN_classification(in_dim=embedding_dim, hidden_size=hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    
    def forward(self,x):

        # x (batch_size, seq_len) - token indices
        out = self.embedding(x)
        out = self.rnn(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out
    

class SpamClassifierLSTM(nn.Module):
    ''' LSTM classifier'''
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(SpamClassifierLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = LSTM_classification(in_dim=embedding_dim, hidden_size=hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    
    def forward(self,x):

        # x (batch_size, seq_len) - token indices
        out = self.embedding(x)
        out = self.rnn(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out


if __name__ == '__main__':

    args = parse_args()
    
    # Auto-generate run name from args
    run_name = f"{args.architecture}-e{args.epochs}-lr{args.learning_rate}-h{args.hidden_size}-len{args.max_length}"
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="spam-classification-rnn",
            name=run_name,
            config=vars(args)  # Convert args to dict
        )
        config = wandb.config
    else:
        config = args
    
    # Now use config.architecture, config.learning_rate, etc.
    print(f"Training with config:")
    print(f"  Architecture: {config.architecture}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Hidden Size: {config.hidden_size}")


    # Load data
    num_epochs = config.epochs
    df = pd.read_csv(DATAPATH)
    train_df, test_df = train_test_split(df,test_size=0.2,random_state=config.seed)
        
    tokenizer = CharTokenizer()
    train_dataset = SpamDataset(train_df, tokenizer=tokenizer, max_length=config.max_length)
    test_dataset = SpamDataset(test_df, tokenizer=tokenizer, max_length=config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    if config.architecture == 'RNN':
        print(f'model selected:  {config.architecture}')
        model = SpamClassifier(
            vocab_size=len(tokenizer.vocab),
            embedding_dim=config.embedding_dim, 
            hidden_size=config.hidden_size,
            num_classes=2 
        )
    elif config.architecture == 'LSTM':
        print(f'model selected:  {config.architecture}')
        model = SpamClassifierLSTM(
        vocab_size=len(tokenizer),
        embedding_dim= config.embedding_dim,
        hidden_size=config.hidden_size,
        num_classes= 2
    )
    else:
        raise ValueError(f'unknown model selected:  {config.architecture}')


    # Watch the model -- logs gradients and topology
    # 1. Tracking Gradients
    # Every time loss.backward() is called, PyTorch calculates gradients. 
    # wandb.watch hooks into the model to capture these values.
    # 2. Visualizing Topology
    # W&B will automatically generate a graph/diagram of your model's architecture.
    # log_freq=100 tells W&B to only sample and log the internal state every 100 batches.
    wandb.watch(model, log_freq=100)

    class_counts = train_df['spam'].value_counts().sort_index()
    total = len(train_df)
    weights = torch.tensor([total/class_counts[0], total/class_counts[1]], dtype=torch.float32)

    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)

    print("Start Training ...... ")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            pred = model(batch_x)
            loss = loss_fn(pred,batch_y)
            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()

        # evaluation
        model.eval()
        total_test_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                pred = model(batch_x)
                loss = loss_fn(pred,batch_y)
                y_pred = torch.argmax(pred,dim=1)

                all_predictions.extend(y_pred.tolist())
                all_labels.extend(batch_y.tolist())
                total_test_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        
        report = classification_report(all_labels, all_predictions, target_names=['Not Spam', 'Spam'], output_dict=True)
        accuracy = report['accuracy']
        spam_f1 = report['Spam']['f1-score']
        print(f'Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Test Acc: {accuracy:.4f}')

        # 4. Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_loss": avg_test_loss,
            "test_accuracy": accuracy,
            "spam_f1_score": spam_f1,
            # Log confusion matrix for imbalanced data insight
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_predictions,
                class_names=['Not Spam', 'Spam']
            )
        })
    
    if not args.no_wandb:
        wandb.finish()

            
