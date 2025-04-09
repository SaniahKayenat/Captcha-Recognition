import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear_1 = nn.Linear(4800, 64)
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        # print(images.size())
        # images = images / 255.0 
        x = self.conv_1(images)
        print(x)
        # print(x)
        x = F.relu(x)
        # print(x)
        
        x = self.pool_1(x)
        x = F.relu(self.conv_2(x))
        x = self.pool_2(x)
        # print(x.size())
        x = x.permute(0, 3, 1, 2)
        # print(x.size())
        x = x.view(bs, x.size(1), -1)
        # print(x.size())
        x = F.relu(self.linear_1(x))
        # print(x.size())
        x = self.drop_1(x)
        x, _ = self.lstm(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)
        if targets is not None:
            # print('i am here')
            log_probs = F.log_softmax(x, 2)
            # print(log_probs)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            # print(input_lengths,target_lengths)
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            
            return x, loss

        return x, None


if __name__ == "__main__":
    cm = CaptchaModel(19)
    img = torch.rand((1, 3, 50, 200))
    x, _ = cm(img, torch.rand((1, 5)))