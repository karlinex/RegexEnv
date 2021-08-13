from torch.nn.utils.rnn import PackedSequence

from RegexEnv import RegexEnv
import torch
import torch.nn as nn


class Model(nn.Module):
    # state_size - sequence of letters
    # action_size - Q values for every next letter
    def __init__(self, state_size, action_size, hidden_size):
        super(Model, self).__init__()

        # (B, Seq, 1) => (B, Seq, hidden_size)
        # [[100, 200, 201]] => [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        self.embeddings = torch.nn.Embedding(
            num_embeddings=action_size,
            embedding_dim=hidden_size
        )

        self.rnn = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )

        self.fc = torch.nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(in_features=state_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_size),
            nn.ReLU()
        )

        size_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'model size: {size_params}')

    def forward(self, s_t0):

        lengths = 0 # TODO noteikt, kur katram s_t0 elementam sakas -1
        # ja B = 3 tad piem lengths = [1, 4, 10]

        s_t0 = self.embeddings.forward(s_t0)

        seq_s_t0: PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(
            s_t0,
            torch.LongTensor(lengths),
            batch_first=True,
            enforce_sorted=False)

        seq_rrn_out, _ = self.rnn.forward(seq_s_t0)

        rrn_out = torch.nn.utils.rnn.pad_packed_sequence(
            seq_rrn_out,
            batch_first=True
        )

        # rrn_out => (B, max_len, hidden)

        stacked_last = []
        for idx_sample_in_batch in range(rrn_out.size(0)):
            length_sample = lengths[idx_sample_in_batch]
            rrn_out_sample = rrn_out[idx_sample_in_batch]
            last_hidden = rrn_out_sample[length_sample - 1]
            stacked_last.append(last_hidden)

        stacked_last = torch.stack(stacked_last) # (B, hidden)

        q_values = self.fc.forward(stacked_last)

        return q_values


text_length = 30
env = RegexEnv(text_length)
state_shape = env.observation_space.shape
for t in range(text_length):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()