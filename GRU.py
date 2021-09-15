import csv
from datetime import datetime

from torch.nn.utils.rnn import PackedSequence

from RegexEnv import RegexEnv
import copy

import argparse
import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-device', default='cuda', type=str)

parser.add_argument('-learning_rate', default=1e-3, type=float)
parser.add_argument('-batch_size', default=64, type=int)
parser.add_argument('-episodes', default=1000, type=int)

parser.add_argument('-replay_buffer_size', default=2000, type=int)
parser.add_argument('-target_alpha', default=0.5, type=float)
parser.add_argument('-target_update', default=5000, type=int)

parser.add_argument('-replay_times', default=1, type=int)

parser.add_argument('-hidden_size', default=512, type=int)

parser.add_argument('-gamma', default=0.6, type=float)
parser.add_argument('-epsilon', default=0.9, type=float)
parser.add_argument('-epsilon_min', default=0.1, type=float)
parser.add_argument('-epsilon_decay', default=0.999, type=float)

parser.add_argument('-max_steps', default=38, type=int)

args, other_args = parser.parse_known_args()

if not torch.cuda.is_available():
    args.device = 'cpu'

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

        self.rnn = torch.nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )

        self.fc = torch.nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.BatchNorm1d(num_features=hidden_size),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=action_size),
            nn.ReLU()
        )

        size_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'model size: {size_params}')

    def forward(self, s_t0):
        lengths_s = []
        for idx in range(len(s_t0)):
            each = s_t0[idx]
            length_s = len(each[each >= 0])
            lengths_s.append(torch.LongTensor([length_s]).squeeze())
        lengths = torch.stack(lengths_s)

        # lengths = torch.where(torch.squeeze(s_t0, 1)!=-1)
        # for loop, squeeze nedrikst veikt
        # ja B = 3 tad piem lengths = [1, 4, 10]

        seq_s_t0: PackedSequence = torch.nn.utils.rnn.pack_padded_sequence(
            s_t0,
            torch.LongTensor(lengths),
            batch_first=True,
            enforce_sorted=False)

        s_t0_data = self.embeddings.forward(seq_s_t0.data)
        seq_s_emb_t0 = PackedSequence(
            s_t0_data, seq_s_t0.batch_sizes, seq_s_t0.sorted_indices, seq_s_t0.unsorted_indices
        )

        seq_rrn_out, _ = self.rnn.forward(seq_s_emb_t0)

        rrn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
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

class ReplayPriorityMemory:
    def __init__(self, size, batch_size, prob_alpha=1):
        self.size = size
        self.batch_size = batch_size
        self.prob_alpha = prob_alpha
        self.memory = []
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        new_priority = np.max(self.priorities) * 0.8 if self.memory else 1.0

        self.memory.append(transition)
        if len(self.memory) > self.size:
            del self.memory[0]
        pos = len(self.memory) - 1
        self.priorities[pos] = new_priority

    def sample(self):
        probs = np.array(self.priorities)
        if len(self.memory) < len(probs):
            probs = probs[:len(self.memory)]

        probs += 1e-8
        probs = probs ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        return samples, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority.item()

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.is_double = True

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = args.gamma  # discount rate
        self.epsilon = args.epsilon  # exploration rate
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
        self.learning_rate = args.learning_rate
        self.device = args.device
        self.q_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.q_t_model = Model(self.state_size, self.action_size, args.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.q_model.parameters(),
            lr=self.learning_rate,
        )

        self.replay_memory = ReplayPriorityMemory(args.replay_buffer_size, args.batch_size)

    def update_q_t_model(self):
        print('update_q_t_model')
        state_current = self.q_t_model.state_dict()
        state_new = copy.copy(self.q_t_model.state_dict())
        alpha = args.target_alpha
        for key, state_new_each in state_new.items():
            state_new[key] = alpha * state_new_each + (1 - alpha) * state_current[key]
        self.q_t_model.load_state_dict(state_new)

    def act(self, s_t0):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                s_t0 = torch.LongTensor(s_t0).to(args.device)
                s_t0 = s_t0.unsqueeze(dim=0)
                if s_t0[0, 0].item() == -1:
                    s_t0[0, 0] = 0

                self.q_model = self.q_model.eval()
                q_all = self.q_model.forward(s_t0)
                self.q_model = self.q_model.train()
                a_t0 = q_all.squeeze().argmax().cpu().item()
                return a_t0

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        self.optimizer.zero_grad()
        batch, replay_idxes = self.replay_memory.sample()
        s_t0, a_t0, r_t1, s_t1, is_end = zip(*batch)

        s_t0 = torch.LongTensor(s_t0).to(args.device)
        a_t0 = torch.LongTensor(a_t0).to(args.device)
        r_t1 = torch.FloatTensor(r_t1).to(args.device)
        s_t1 = torch.LongTensor(s_t1).to(args.device)
        is_not_end = torch.FloatTensor((np.array(is_end) == False) * 1.0).to(args.device)

        idxes = torch.arange(args.batch_size).to(args.device)

        q_t0_all = self.q_model.forward(s_t0)
        q_t0 = q_t0_all[idxes, a_t0]

        q_t1_all = self.q_t_model.forward(s_t1).detach()
        a_t1 = q_t1_all.argmax(dim=1)

        q_t1 = q_t1_all[idxes, a_t1]

        q_t1_final = r_t1 + is_not_end * args.gamma * q_t1

        td_error = torch.abs(q_t0 - q_t1_final)
        self.replay_memory.update_priorities(replay_idxes, td_error)

        loss = torch.mean(td_error)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().item()


# environment name
env = RegexEnv(args.max_steps)

all_scores = []
all_losses = []
all_t = []

agent = DQNAgent(
    env.observation_space.shape[0],
    env.action_space.n
)
plt.figure()

is_end = False
t_total = 0

score_plt = plt.subplot(3, 1, 1)
plt.ylabel('Punkti')
loss_plt = plt.subplot(3, 1, 2)
plt.ylabel('Kļūda')
steps_plt = plt.subplot(3, 1, 3)
plt.ylabel('Soļi')
plt.xlabel('Epizodes')
plt.ion()
plt.show()

filename = f"results/{ datetime.now().strftime('%Y_%m_%d-%H_%M_%S') }.csv"
result_file = open(filename, 'w', newline = '', encoding='utf-8-sig')
csv_writer = csv.writer(result_file, delimiter=';')

for e in range(args.episodes):
    s_t0 = env.reset()
    reward_total = 0
    episode_loss = []
    for t in range(args.max_steps):
        t_total += 1

        a_t0 = agent.act(s_t0)
        s_t1, r_t1, is_end, _ = env.step(a_t0)

        reward_total += r_t1

        if t == args.max_steps - 1:
            r_t1 = -100
            is_end = True

        agent.replay_memory.push(
            (s_t0, a_t0, r_t1, s_t1, is_end)
        )
        s_t0 = s_t1

        if len(agent.replay_memory) > args.batch_size:
            loss_agg = []
            agent.update_epsilon()
            for _ in range(args.replay_times):
                loss_agg.append(agent.replay())
            if t_total > args.target_update or len(all_losses) == 0:
                agent.update_q_t_model()
                t_total = 0
            loss = np.mean(loss_agg)
            episode_loss.append(loss)

        if is_end:
            all_scores.append(reward_total)
            all_losses.append(np.mean(episode_loss))
            break

    all_t.append(t)
    print(
        f'episode: {e}/{args.episodes} '
        f'generated text: {env.address_text} '
        f'loss: {all_losses[-1]} '
        f'score: {reward_total} '
        f't: {t} '
        f'e: {agent.epsilon} '
        f'mem: {len(agent.replay_memory)} ')

    csv_writer.writerow([e, env.address_text, all_losses[-1], reward_total, t, agent.epsilon, len(agent.replay_memory)])
    result_file.flush()
    score_plt.plot(all_scores)
    loss_plt.plot(all_losses)
    steps_plt.plot(all_t)

    plt.draw()
    plt.pause(0.001)

result_file.close()
env.close()