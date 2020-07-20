




from torch.utils import data

class train_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x, y, data_indices):
        'Initialization'
        self.x = x
        self.y = y
        self.data_indices = data_indices
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_indices)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        IDs = self.data_indices[index]

        # Load data and get label
        curr_x = self.x[IDs, :, :, :, :]
        curr_y = self.y[IDs, :, :, :, :]

        #return X, y
        return(curr_x, curr_y)
    
class validation_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, x_validation, y_validation, data_indices):
        'Initialization'
        self.x_validation = x_validation
        self.y_validation = y_validation
        self.data_indices = data_indices
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_indices)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        IDs = self.data_indices[index]

        # Load data and get label
        curr_x = self.x_validation[IDs, :, :, :, :]
        curr_y = self.y_validation[IDs, :, :, :, :]

        #return X, y
        return(curr_x, curr_y)
