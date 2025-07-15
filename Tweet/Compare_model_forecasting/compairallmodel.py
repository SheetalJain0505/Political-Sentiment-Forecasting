

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math
from typing import List, Tuple, Dict, Any
import os
import json
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")

"""# --------------------- COMMON COMPONENTS ---------------------"""

class TimeSeriesDataset(Dataset):
    """Custom Dataset for time-series data with multiple features."""
    def __init__(self, data: np.ndarray, seq_len: int, label_len: int, pred_len: int, augment: bool = False, noise_std: float = 0.01):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.augment = augment
        self.noise_std = noise_std
        self.X_enc, self.X_dec, self.y = self._create_sequences()

    def _data_augmentation(self, data: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.noise_std, data.shape)
        return data + noise

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_enc, X_dec, y = [], [], []
        for i in range(len(self.data) - self.seq_len - self.pred_len + 1):
            enc_end = i + self.seq_len
            dec_end = enc_end + self.label_len + self.pred_len
            x_enc = self.data[i:enc_end]
            x_dec = np.zeros((self.label_len + self.pred_len, self.data.shape[1]))
            x_dec[:self.label_len] = self.data[enc_end-self.label_len:enc_end]
            target = self.data[enc_end][0]  # Assuming target is the first column
            if self.augment:
                x_enc = self._data_augmentation(x_enc)
                x_dec[:self.label_len] = self._data_augmentation(x_dec[:self.label_len])
            X_enc.append(x_enc)
            X_dec.append(x_dec)
            y.append(target)
        return np.array(X_enc), np.array(X_dec), np.array(y)

    def __len__(self) -> int:
        return len(self.X_enc)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_enc = torch.tensor(self.X_enc[idx], dtype=torch.float32)
        x_dec = torch.tensor(self.X_dec[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        x_mark_enc = torch.zeros_like(x_enc)
        x_mark_dec = torch.zeros_like(x_dec)
        return x_enc, x_mark_enc, x_dec, x_mark_dec, y

def get_dataloader(data: np.ndarray, seq_len: int, label_len: int, pred_len: int, batch_size: int,
                  split_ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2), augment: bool = True,
                  noise_std: float = 0.01, shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = TimeSeriesDataset(data, seq_len, label_len, pred_len, augment, noise_std)
    total_size = len(dataset)
    train_size = int(total_size * split_ratios[0])
    val_size = int(total_size * split_ratios[1])
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, val_loader, test_loader

def plot_results(train_loss_history: Dict[str, List[float]], val_loss_history: Dict[str, List[float]],
                 predictions: List[float], actuals: List[float], model_name: str, output_dir: str,
                 target_column: str = 'sentiment') -> None:
    def smooth_curve(data: List[float], window_size: int = 5) -> np.ndarray:
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

    def exponential_moving_average(data: List[float], alpha: float = 0.3) -> np.ndarray:
        return pd.Series(data).ewm(alpha=alpha, adjust=False).mean().values

    # Plot SMAPE loss
    val_loss_smoothed = smooth_curve(val_loss_history['smape'], window_size=5)
    val_loss_ema = exponential_moving_average(val_loss_history['smape'], alpha=0.3)

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history['smape'], label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_loss_history['smape'], label='Val Loss (Raw)', color='orange', alpha=0.3)
    plt.plot(val_loss_smoothed, label='Val Loss (SMA)', color='orange', linewidth=2)
    plt.plot(val_loss_ema, label='Val Loss (EMA)', color='green', linewidth=2, linestyle='--')
    plt.title(f"Training vs Validation Loss (Smoothed) - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("SMAPE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{model_name}_loss_smape.png")
    plt.close()

    # Plot MSE loss
    val_loss_smoothed_mse = smooth_curve(val_loss_history['mse'], window_size=5)
    val_loss_ema_mse = exponential_moving_average(val_loss_history['mse'], alpha=0.3)

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history['mse'], label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_loss_history['mse'], label='Val Loss (Raw)', color='orange', alpha=0.3)
    plt.plot(val_loss_smoothed_mse, label='Val Loss (SMA)', color='orange', linewidth=2)
    plt.plot(val_loss_ema_mse, label='Val Loss (EMA)', color='green', linewidth=2, linestyle='--')
    plt.title(f"Training vs Validation Loss (MSE) - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{model_name}_loss_mse.png")
    plt.close()

    # Plot MAE loss
    val_loss_smoothed_mae = smooth_curve(val_loss_history['mae'], window_size=5)
    val_loss_ema_mae = exponential_moving_average(val_loss_history['mae'], alpha=0.3)

    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history['mae'], label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_loss_history['mae'], label='Val Loss (Raw)', color='orange', alpha=0.3)
    plt.plot(val_loss_smoothed_mae, label='Val Loss (SMA)', color='orange', linewidth=2)
    plt.plot(val_loss_ema_mae, label='Val Loss (EMA)', color='green', linewidth=2, linestyle='--')
    plt.title(f"Training vs Validation Loss (MAE) - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{model_name}_loss_mae.png")
    plt.close()

    # Plot Forecasted vs Actual
    plt.figure(figsize=(12, 5))
    plt.plot(predictions, label=f'Forecasted with {model_name}', color='blue')
    plt.plot(actuals, label='Actual', color='orange')
    plt.title(f'Forecasted vs Actual {target_column.capitalize()} - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/{model_name}_forecast.png")
    plt.close()


class ProbAttention(nn.Module):
    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: float = None, attention_dropout: float = 0.2, output_attention: bool = False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in: torch.Tensor, V: torch.Tensor, scores: torch.Tensor, index: torch.Tensor, L_Q: int, attn_mask: Any) -> Tuple[torch.Tensor, Any]:
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: Any) -> Tuple[torch.Tensor, Any]:
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        if self.mask_flag:
            u = L_Q
        else:
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.transpose(2,1).contiguous(), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: float = None, attention_dropout: float = 0.2, output_attention: bool = False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: Any) -> Tuple[torch.Tensor, Any]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class MultiHeadAttention(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys: int = None, d_values: int = None, mix: bool = False):
        super(MultiHeadAttention, self).__init__()
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: Any) -> Tuple[torch.Tensor, Any]:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None, dropout: float = 0.2, activation: str = "relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Any = None) -> Tuple[torch.Tensor, Any]:
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm2(x+y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers: nn.ModuleList, lstm: nn.Module = None, temporal_conv: nn.Module = None, norm_layer: nn.Module = None):
        super(Encoder, self).__init__()
        self.lstm = lstm
        self.temporal_conv = temporal_conv
        self.attn_layers = attn_layers
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Any = None) -> Tuple[torch.Tensor, List[Any]]:
        attns = []
        if self.lstm is not None:
            x, _ = self.lstm(x)
        if self.temporal_conv is not None:
            x = self.temporal_conv(x)
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, d_ff: int = None, dropout: float = 0.2, activation: str = "relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Any = None, cross_mask: Any = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, norm_layer: nn.Module = None):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Any = None, cross_mask: Any = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pos_embedding[:, :seq_len, :]

class FeatureEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super(FeatureEmbedding, self).__init__()
        self.feature_projection = nn.Linear(c_in, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_projection(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.2):
        super(DataEmbedding, self).__init__()
        self.feature_embeddings = nn.ModuleList([FeatureEmbedding(1, d_model) for _ in range(c_in)])
        self.position_embedding = LearnablePositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        embedded_features = []
        for i in range(C):
            feature = x[:, :, i:i+1]
            embedded = self.feature_embeddings[i](feature)
            embedded_features.append(embedded)
        x = sum(embedded_features)
        x = x + self.position_embedding(x)
        return self.dropout(x)

# --------------------- LOSS FUNCTIONS ---------------------
def smape_loss_with_label_smoothing(y_pred: torch.Tensor, y_true: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    epsilon = 1e-8
    denominator = (torch.abs(y_true) + torch.abs(y_pred)).clamp(min=epsilon)
    y_true_smoothed = (1 - smoothing) * y_true + smoothing * 0.5
    return torch.mean(200 * torch.abs(y_pred - y_true_smoothed) / denominator)

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)

def mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))



def preprocess_dataframe(df: pd.DataFrame, target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Tuple[np.ndarray, MinMaxScaler]:
    df = df.copy()

    # If 'Time' column doesn't exist, create it based on the index
    if 'Time' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            # If the index is a DatetimeIndex, calculate days from the earliest date
            df['Time'] = (df.index - df.index.min()).days
        else:
            # If the index is not a DatetimeIndex, use a numerical sequence as time
            df['Time'] = np.arange(len(df))

    required_columns = [target_column] + additional_columns
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[required_columns])
    return scaled, scaler

"""# --------------------- MODEL IMPLEMENTATIONS ---------------------"""

class PRTPlus(nn.Module):
    def __init__(self, enc_in: int = 4, dec_in: int = 4, c_out: int = 1, seq_len: int = 30, label_len: int = 15, out_len: int = 1,
                 factor: int = 5, d_model: int = 32, n_heads: int = 4, e_layers: int = 1, d_layers: int = 1, d_ff: int = 128,
                 dropout: float = 0.2, attn: str = 'prob', activation: str = 'gelu', output_attention: bool = False):
        super(PRTPlus, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        Attn = ProbAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadAttention(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiHeadAttention(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, n_heads, mix=True),
                    MultiHeadAttention(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: Any = None, dec_self_mask: Any = None, dec_enc_mask: Any = None) -> torch.Tensor:
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

class Informer(nn.Module):
    def __init__(self, enc_in: int = 4, dec_in: int = 4, c_out: int = 1, seq_len: int = 30, label_len: int = 15, out_len: int = 1,
                 factor: int = 5, d_model: int = 96, n_heads: int = 8, e_layers: int = 2, d_layers: int = 1, d_ff: int = 384,
                 dropout: float = 0.1, attn: str = 'prob', activation: str = 'gelu', output_attention: bool = False):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.attn = attn
        self.output_attention = output_attention

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        Attn = ProbAttention if attn == 'prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MultiHeadAttention(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                       d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiHeadAttention(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, n_heads, mix=True),
                    MultiHeadAttention(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                       d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: Any = None, dec_self_mask: Any = None, dec_enc_mask: Any = None) -> torch.Tensor:
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

class RNNModel(nn.Module):
    def __init__(self, enc_in: int = 4, dec_in: int = 4, c_out: int = 1, seq_len: int = 30, label_len: int = 15, out_len: int = 1,
                 d_model: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(RNNModel, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.rnn = nn.RNN(input_size=enc_in, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder_rnn = nn.RNN(input_size=dec_in, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: Any = None, dec_self_mask: Any = None, dec_enc_mask: Any = None) -> torch.Tensor:
        _, hidden = self.rnn(x_enc)
        dec_out, _ = self.decoder_rnn(x_dec, hidden)
        out = self.fc(dec_out)
        return out[:, -self.pred_len:, :]

class LSTMModel(nn.Module):
    def __init__(self, enc_in: int = 4, dec_in: int = 4, c_out: int = 1, seq_len: int = 30, label_len: int = 15, out_len: int = 1,
                 d_model: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.lstm = nn.LSTM(input_size=enc_in, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder_lstm = nn.LSTM(input_size=dec_in, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: Any = None, dec_self_mask: Any = None, dec_enc_mask: Any = None) -> torch.Tensor:
        _, (hidden, cell) = self.lstm(x_enc)
        dec_out, _ = self.decoder_lstm(x_dec, (hidden, cell))
        out = self.fc(dec_out)
        return out[:, -self.pred_len:, :]

class GRUModel(nn.Module):
    def __init__(self, enc_in: int = 4, dec_in: int = 4, c_out: int = 1, seq_len: int = 30, label_len: int = 15, out_len: int = 1,
                 d_model: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.pred_len = out_len
        self.label_len = label_len
        self.gru = nn.GRU(input_size=enc_in, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.decoder_gru = nn.GRU(input_size=dec_in, hidden_size=d_model, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(d_model, c_out)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                enc_self_mask: Any = None, dec_self_mask: Any = None, dec_enc_mask: Any = None) -> torch.Tensor:
        _, hidden = self.gru(x_enc)
        dec_out, _ = self.decoder_gru(x_dec, hidden)
        out = self.fc(dec_out)
        return out[:, -self.pred_len:, :]

class ARIMAModel:
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None

    def fit(self, data: np.ndarray) -> None:
        self.model = ARIMA(data[:, 0], order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, data: np.ndarray, seq_len: int, label_len: int, pred_len: int) -> Tuple[List[float], List[float]]:
        predictions, actuals = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            enc_end = i + seq_len
            history = data[i:enc_end, 0]
            self.model = ARIMA(history, order=self.order)
            self.model_fit = self.model.fit()
            forecast = self.model_fit.forecast(steps=pred_len)
            predictions.append(forecast[-1])
            actuals.append(data[enc_end, 0])
        return predictions, actuals

"""# --------------------- TRAINING AND EVALUATION ---------------------"""

def train_neural_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                       epochs: int = 50, clip_value: float = 1.0, patience: int = 5,
                       device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Initialize dictionaries to store all loss histories
    train_loss_history = {'smape': [], 'mse': [], 'mae': []}
    val_loss_history = {'smape': [], 'mse': [], 'mae': []}
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = {'smape': [], 'mse': [], 'mae': []}
        for xb_enc, xb_mark_enc, xb_dec, xb_mark_dec, yb in train_loader:
            xb_enc, xb_mark_enc, xb_dec, xb_mark_dec, yb = (
                xb_enc.to(device), xb_mark_enc.to(device), xb_dec.to(device),
                xb_mark_dec.to(device), yb.unsqueeze(1).to(device)
            )
            pred = model(xb_enc, xb_mark_enc, xb_dec, xb_mark_dec)

            # Compute all losses
            smape = smape_loss_with_label_smoothing(pred.squeeze(-1), yb)
            mse = mse_loss(pred.squeeze(-1), yb)
            mae = mae_loss(pred.squeeze(-1), yb)

            # Use SMAPE for optimization
            optimizer.zero_grad()
            smape.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            # Store losses
            train_losses['smape'].append(smape.item())
            train_losses['mse'].append(mse.item())
            train_losses['mae'].append(mae.item())

        model.eval()
        val_losses = {'smape': [], 'mse': [], 'mae': []}
        with torch.no_grad():
            for xb_enc, xb_mark_enc, xb_dec, xb_mark_dec, yb in val_loader:
                xb_enc, xb_mark_enc, xb_dec, xb_mark_dec, yb = (
                    xb_enc.to(device), xb_mark_enc.to(device), xb_dec.to(device),
                    xb_mark_dec.to(device), yb.unsqueeze(1).to(device)
                )
                pred = model(xb_enc, xb_mark_enc, xb_dec, xb_mark_dec)

                # Compute all losses
                smape = smape_loss_with_label_smoothing(pred.squeeze(-1), yb)
                mse = mse_loss(pred.squeeze(-1), yb)
                mae = mae_loss(pred.squeeze(-1), yb)

                val_losses['smape'].append(smape.item())
                val_losses['mse'].append(mse.item())
                val_losses['mae'].append(mae.item())

        # Compute mean losses for the epoch
        mean_train_losses = {key: np.mean(val) for key, val in train_losses.items()}
        mean_val_losses = {key: np.mean(val) for key, val in val_losses.items()}

        # Append to histories
        for key in train_loss_history:
            train_loss_history[key].append(mean_train_losses[key])
            val_loss_history[key].append(mean_val_losses[key])

        # Early stopping based on SMAPE
        val_loss_smoothed = pd.Series(val_loss_history['smape']).rolling(window=5, min_periods=1).mean().values
        if len(val_loss_smoothed) > 0:
            current_val_loss = val_loss_smoothed[-1]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                counter = 0
                best_model_state = model.state_dict()
            else:
                counter += 1

        scheduler.step(mean_val_losses['smape'])
        print(f"Epoch {epoch+1}/{epochs} | Train SMAPE: {mean_train_losses['smape']:.4f} | Val SMAPE: {mean_val_losses['smape']:.4f} | "
              f"Train MSE: {mean_train_losses['mse']:.4f} | Val MSE: {mean_val_losses['mse']:.4f} | "
              f"Train MAE: {mean_train_losses['mae']:.4f} | Val MAE: {mean_val_losses['mae']:.4f}")

        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    model.eval()
    predictions, actuals = [], []
    final_losses = {'smape': 0.0, 'mse': 0.0, 'mae': 0.0}
    with torch.no_grad():
        for xb_enc, xb_mark_enc, xb_dec, xb_mark_dec, yb in test_loader:
            xb_enc, xb_mark_enc, xb_dec, xb_mark_dec = (
                xb_enc.to(device), xb_mark_enc.to(device), xb_dec.to(device),
                xb_mark_dec.to(device)
            )
            pred = model(xb_enc, xb_mark_enc, xb_dec, xb_mark_dec)
            predictions.append(pred.cpu().item())
            actuals.append(yb.item())

    # Compute final losses on test set
    predictions_tensor = torch.tensor(predictions)
    actuals_tensor = torch.tensor(actuals)
    final_losses['smape'] = smape_loss_with_label_smoothing(predictions_tensor, actuals_tensor).item()
    final_losses['mse'] = mse_loss(predictions_tensor, actuals_tensor).item()
    final_losses['mae'] = mae_loss(predictions_tensor, actuals_tensor).item()

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "predictions": predictions,
        "actuals": actuals,
        "final_losses": final_losses
    }

def train_non_neural_model(model: Any, data: np.ndarray, df: pd.DataFrame, seq_len: int, label_len: int, pred_len: int) -> Dict[str, Any]:
    if isinstance(model, ARIMAModel):
        predictions, actuals = model.predict(data, seq_len, label_len, pred_len)
    else:  # Prophet
        predictions, actuals = model.predict(df, seq_len, label_len, pred_len)

    # Compute all losses
    predictions_tensor = torch.tensor(predictions)
    actuals_tensor = torch.tensor(actuals)
    final_losses = {
        'smape': smape_loss_with_label_smoothing(predictions_tensor, actuals_tensor).item(),
        'mse': mse_loss(predictions_tensor, actuals_tensor).item(),
        'mae': mae_loss(predictions_tensor, actuals_tensor).item()
    }

    return {
        "train_loss_history": {'smape': [], 'mse': [], 'mae': []},
        "val_loss_history": {'smape': [], 'mse': [], 'mae': []},
        "predictions": predictions,
        "actuals": actuals,
        "final_losses": final_losses
    }

def save_results(result: Dict[str, Any], model_name: str, output_dir: str = "results", target_column: str = 'sentiment') -> None:
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save metrics as JSON
    metrics = {
        "train_loss_history": {
            "smape": [float(x) for x in result["train_loss_history"]["smape"]],
            "mse": [float(x) for x in result["train_loss_history"]["mse"]],
            "mae": [float(x) for x in result["train_loss_history"]["mae"]]
        },
        "val_loss_history": {
            "smape": [float(x) for x in result["val_loss_history"]["smape"]],
            "mse": [float(x) for x in result["val_loss_history"]["mse"]],
            "mae": [float(x) for x in result["val_loss_history"]["mae"]]
        },
        "predictions": [float(x) for x in result["predictions"]],
        "actuals": [float(x) for x in result["actuals"]],
        "final_losses": {
            "smape": float(result["final_losses"]["smape"]),
            "mse": float(result["final_losses"]["mse"]),
            "mae": float(result["final_losses"]["mae"])
        }
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate and save plots
    plot_results(
        train_loss_history=result["train_loss_history"],
        val_loss_history=result["val_loss_history"],
        predictions=result["predictions"],
        actuals=result["actuals"],
        model_name=model_name,
        output_dir=model_dir,
        target_column=target_column
    )

"""# --------------------- INDIVIDUAL MODEL RUN FUNCTIONS ---------------------"""

def run_prt_plus(df: pd.DataFrame, seq_len: int = 30, label_len: int = 15, pred_len: int = 1,
                 batch_size: int = 16, epochs: int = 50, patience: int = 5,
                 target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Dict[str, Any]:
    scaled_data, _ = preprocess_dataframe(df, target_column, additional_columns)
    train_loader, val_loader, test_loader = get_dataloader(
        data=scaled_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        split_ratios=(0.6, 0.2, 0.2),
        augment=True,
        noise_std=0.01
    )
    model = PRTPlus(
        enc_in=len([target_column] + additional_columns),
        dec_in=len([target_column] + additional_columns),
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len
    )
    print("Training PRTPlus...")
    result = train_neural_model(model, train_loader, val_loader, test_loader, epochs=epochs, patience=patience)
    print(f"PRTPlus - Final Test SMAPE Loss: {result['final_losses']['smape']:.4f}")
    print(f"PRTPlus - Final Test MSE Loss: {result['final_losses']['mse']:.4f}")
    print(f"PRTPlus - Final Test MAE Loss: {result['final_losses']['mae']:.4f}")
    save_results(result, "PRTPlus", target_column=target_column)
    return result

def run_informer(df: pd.DataFrame, seq_len: int = 30, label_len: int = 15, pred_len: int = 1,
                 batch_size: int = 16, epochs: int = 50, patience: int = 5,
                 target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Dict[str, Any]:
    scaled_data, _ = preprocess_dataframe(df, target_column, additional_columns)
    train_loader, val_loader, test_loader = get_dataloader(
        data=scaled_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        split_ratios=(0.6, 0.2, 0.2),
        augment=True,
        noise_std=0.01
    )
    model = Informer(
        enc_in=len([target_column] + additional_columns),
        dec_in=len([target_column] + additional_columns),
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len
    )
    print("Training Informer...")
    result = train_neural_model(model, train_loader, val_loader, test_loader, epochs=epochs, patience=patience)
    print(f"Informer - Final Test SMAPE Loss: {result['final_losses']['smape']:.4f}")
    print(f"Informer - Final Test MSE Loss: {result['final_losses']['mse']:.4f}")
    print(f"Informer - Final Test MAE Loss: {result['final_losses']['mae']:.4f}")
    save_results(result, "Informer", target_column=target_column)
    return result

def run_rnn(df: pd.DataFrame, seq_len: int = 30, label_len: int = 15, pred_len: int = 1,
            batch_size: int = 16, epochs: int = 50, patience: int = 5,
            target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Dict[str, Any]:
    scaled_data, _ = preprocess_dataframe(df, target_column, additional_columns)
    train_loader, val_loader, test_loader = get_dataloader(
        data=scaled_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        split_ratios=(0.6, 0.2, 0.2),
        augment=True,
        noise_std=0.01
    )
    model = RNNModel(
        enc_in=len([target_column] + additional_columns),
        dec_in=len([target_column] + additional_columns),
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len
    )
    print("Training RNN...")
    result = train_neural_model(model, train_loader, val_loader, test_loader, epochs=epochs, patience=patience)
    print(f"RNN - Final Test SMAPE Loss: {result['final_losses']['smape']:.4f}")
    print(f"RNN - Final Test MSE Loss: {result['final_losses']['mse']:.4f}")
    print(f"RNN - Final Test MAE Loss: {result['final_losses']['mae']:.4f}")
    save_results(result, "RNN", target_column=target_column)
    return result

def run_lstm(df: pd.DataFrame, seq_len: int = 30, label_len: int = 15, pred_len: int = 1,
             batch_size: int = 16, epochs: int = 50, patience: int = 5,
             target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Dict[str, Any]:
    scaled_data, _ = preprocess_dataframe(df, target_column, additional_columns)
    train_loader, val_loader, test_loader = get_dataloader(
        data=scaled_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        split_ratios=(0.6, 0.2, 0.2),
        augment=True,
        noise_std=0.01
    )
    model = LSTMModel(
        enc_in=len([target_column] + additional_columns),
        dec_in=len([target_column] + additional_columns),
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len
    )
    print("Training LSTM...")
    result = train_neural_model(model, train_loader, val_loader, test_loader, epochs=epochs, patience=patience)
    print(f"LSTM - Final Test SMAPE Loss: {result['final_losses']['smape']:.4f}")
    print(f"LSTM - Final Test MSE Loss: {result['final_losses']['mse']:.4f}")
    print(f"LSTM - Final Test MAE Loss: {result['final_losses']['mae']:.4f}")
    save_results(result, "LSTM", target_column=target_column)
    return result

def run_gru(df: pd.DataFrame, seq_len: int = 30, label_len: int = 15, pred_len: int = 1,
            batch_size: int = 16, epochs: int = 50, patience: int = 5,
            target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Dict[str, Any]:
    scaled_data, _ = preprocess_dataframe(df, target_column, additional_columns)
    train_loader, val_loader, test_loader = get_dataloader(
        data=scaled_data,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        batch_size=batch_size,
        split_ratios=(0.6, 0.2, 0.2),
        augment=True,
        noise_std=0.01
    )
    model = GRUModel(
        enc_in=len([target_column] + additional_columns),
        dec_in=len([target_column] + additional_columns),
        c_out=1,
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len
    )
    print("Training GRU...")
    result = train_neural_model(model, train_loader, val_loader, test_loader, epochs=epochs, patience=patience)
    print(f"GRU - Final Test SMAPE Loss: {result['final_losses']['smape']:.4f}")
    print(f"GRU - Final Test MSE Loss: {result['final_losses']['mse']:.4f}")
    print(f"GRU - Final Test MAE Loss: {result['final_losses']['mae']:.4f}")
    save_results(result, "GRU", target_column=target_column)
    return result

def run_arima(df: pd.DataFrame, seq_len: int = 30, label_len: int = 15, pred_len: int = 1,
              target_column: str = 'sentiment', additional_columns: List[str] = ['Retweets', 'Likes', 'Time']) -> Dict[str, Any]:
    scaled_data, _ = preprocess_dataframe(df, target_column, additional_columns)
    model = ARIMAModel()
    print("Training ARIMA...")
    result = train_non_neural_model(model, scaled_data, df, seq_len, label_len, pred_len)
    print(f"ARIMA - Final Test SMAPE Loss: {result['final_losses']['smape']:.4f}")
    print(f"ARIMA - Final Test MSE Loss: {result['final_losses']['mse']:.4f}")
    print(f"ARIMA - Final Test MAE Loss: {result['final_losses']['mae']:.4f}")
    save_results(result, "ARIMA", target_column=target_column)
    return result

alltweet=pd.read_csv(r'C:\Documents\DAIICT\SEM 2\Deep learning lab\project\compairision_forcasting\model comparision\best_model_for_today.csv')


run_prt_plus(alltweet, target_column='cardiff_score')

run_rnn(alltweet, target_column='cardiff_score')

run_informer(alltweet, target_column='cardiff_score')

run_lstm(alltweet, target_column='cardiff_score')

run_gru(alltweet, target_column='cardiff_score')

run_arima(alltweet, target_column='cardiff_score')
