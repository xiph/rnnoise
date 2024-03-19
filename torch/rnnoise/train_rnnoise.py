import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import os
import rnnoise

class RNNoiseDataset(torch.utils.data.Dataset):
    def __init__(self,
                features_file,
                sequence_length=2000):

        self.sequence_length = sequence_length

        self.data = np.memmap(features_file, dtype='float32', mode='r')
        dim = 87

        self.nb_sequences = self.data.shape[0]//self.sequence_length//dim
        self.data = self.data[:self.nb_sequences*self.sequence_length*dim]

        self.data = np.reshape(self.data, (self.nb_sequences, self.sequence_length, dim))

    def __len__(self):
        return self.nb_sequences

    def __getitem__(self, index):
        return self.data[index, :, :42].copy(), self.data[index, :, 42:64].copy()

def mask(g):
    return torch.clamp(g+1, max=1)

adam_betas = [0.8, 0.98]
adam_eps = 1e-8
batch_size=192
lr_decay = 0.001
lr = 0.003
epsilon = 1e-5
epochs = 1000
checkpoint_dir='checkpoint'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint = dict()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

checkpoint['model_args']    = ()
checkpoint['model_kwargs']  = {'gru_size': 256}
model = rnnoise.RNNoise(*checkpoint['model_args'], **checkpoint['model_kwargs'])
dataset = RNNoiseDataset('rnn_feat1.f32')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps)


# learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x : 1 / (1 + lr_decay * x))


if __name__ == '__main__':
    model.to(device)
    states = None
    for epoch in range(1, epochs + 1):

        running_loss = 0

        print(f"training epoch {epoch}...")
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features, gain) in enumerate(tepoch):
                optimizer.zero_grad()
                features = features.to(device)
                gain = gain.to(device)

                pred_gain, states = model(features, states=states)
                states = [state.detach() for state in states]
                gain = gain[:,4:,:]

                loss = torch.mean(mask(gain)*(torch.sqrt(pred_gain)-torch.sqrt(torch.clamp(gain, min=0)))**2)

                loss.backward()
                optimizer.step()

                scheduler.step()

                running_loss += loss.detach().cpu().item()
                tepoch.set_postfix(loss=f"{running_loss/(i+1):8.5f}",
                                   )

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'lossgen_{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = running_loss / len(dataloader)
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)
