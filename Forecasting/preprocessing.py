import torch
import torch.utils.data as data

class RollingWinDataset(data.Dataset):

    def __init__(self, sequence, inp_seq, out_seq, stride = 0, step = 1):

        super().__init__()

        self.sequence   = sequence
        self.inp_seq    = inp_seq
        self.out_seq    = out_seq
        self.stride     = stride
        self.step       = step

        self.rolling_window()

        return

    def rolling_window(self):

        windows = self.sequence.unfold(
            dimension = 1, 
            size = self.inp_seq + self.stride + self.out_seq, 
            step = self.step)

        inp_win    = torch.zeros([*windows.shape[:2], self.inp_seq])
        target_win = torch.zeros([*windows.shape[:2], self.out_seq])

        for i1 in range(windows.shape[1]):

            inp_win[:, i1, :]       = windows[:, i1, :self.inp_seq]
            target_win[:, i1, :]    = windows[:, i1, -self.out_seq:]

        self.inp_win    = inp_win
        self.target_win = target_win
        self.length     = windows.shape[1]
        
        return

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        inp_win     = self.inp_win[:,idx,:].T
        target_win  = self.target_win[:,idx,:].T

        return inp_win, target_win