




import torch
import numpy as np

def sep_x_y(tensor_to_separate):
    x = []
    y = []
    for i in range(len(tensor_to_separate)):
        current_seq = tensor_to_separate[i, :, :, :, :]
        for j in range(current_seq.shape[0]):
            if j >= 10:
                current_x = current_seq[(j-10):j].numpy()
                x.append(current_x)
                current_y = current_seq[j].unsqueeze(dim = 0).numpy()
                y.append(current_y)
    x = np.asarray(x)
    x = torch.from_numpy(x).type(torch.FloatTensor)
    y = np.asarray(y)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    return(x, y)
