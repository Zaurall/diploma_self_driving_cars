import torch
import torch.nn as nn

# TODO загрузка весов


def lowpass_filter(data, cutoff=2.0, fs=20.0, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# class NanoFFModel(pl.LightningModule):
#     def __init__(
#             self,
#             # hidden_dims: tuple[int, int, int] = (16, 8, 4),
#             # from_weights: bool = False,
#             # trial: optuna.Trial | None = None,
#             # platform: str | None = None,
#     ):
#         self.model = nn.Sequential(
#             nn.Linear(4, hidden_dims[0]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[1], hidden_dims[2]),
#             nn.ReLU(),
#             nn.Linear(hidden_dims[2], 2),
#         )
#         if from_weights:
#             with open("/Users/eric/PycharmProjects/openpilot/selfdrive/car/torque_data/neural_ff_weights.json",
#                       "r") as f:
#                 bolt_weights = json.load(f)["CHEVROLET_BOLT_EUV"]
#             self.model[0].weight.data = torch.tensor(bolt_weights["w_1"]).T
#             self.model[0].bias.data = torch.tensor(bolt_weights["b_1"])
#             self.model[2].weight.data = torch.tensor(bolt_weights["w_2"]).T
#             self.model[2].bias.data = torch.tensor(bolt_weights["b_2"])
#             self.model[4].weight.data = torch.tensor(bolt_weights["w_3"]).T
#             self.model[4].bias.data = torch.tensor(bolt_weights["b_3"])
#             self.model[6].weight.data = torch.tensor(bolt_weights["w_4"]).T
#             self.model[6].bias.data = torch.tensor(bolt_weights["b_4"])

#         # define constant parameters
#         self.input_norm_mat = nn.Parameter(torch.tensor([[-3.0, 3.0], [-3.0, 3.0], [0.0, 40.0], [-3.0, 3.0]]), requires_grad=False)
#         self.output_norm_mat = nn.Parameter(torch.tensor([-1.0, 1.0]), requires_grad=False)
#         self.temperature = 100.0


def build_model(cfg):
    mcfg = cfg["model"][cfg["meta"]["model_type"]]
    if mcfg["type"] == "LSTM":
        return LSTMModel(
            input_dim=( len(cfg["data"]["features"]) + cfg["data"].get("future_k", 0) ),
            hidden_size=mcfg["hidden_size"],
            num_layers=mcfg["num_layers"],
            output_dim=len(cfg["data"]["target_column"]),
            dropout=mcfg.get("dropout", 0.0)
        )
    elif mcfg["type"] == "GRU":
        return GRUModel(
            input_dim=( len(cfg["data"]["features"]) + cfg["data"].get("future_k", 0) ),
            hidden_size=mcfg["hidden_size"],
            num_layers=mcfg["num_layers"],
            output_dim=len(cfg["data"]["target_column"]),
            dropout=mcfg.get("dropout", 0.0)
        )
    elif mcfg["type"] == "MLP":
        return MLPModel(
            input_dim=( len(cfg["data"]["features"]) + cfg["data"].get("future_k", 0) ),
            hidden_size=mcfg["hidden_size"],
            output_dim=len(cfg["data"]["target_column"]),
            dropout=mcfg.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {mcfg['type']}")