import torch
from torch import nn
import torch.nn.functional as F

from functools import partial
import math


class SDEPModel(nn.Module):
    """
    Self-distillation network with ensemble prototypes.
    """
    def __init__(self, n_networks=5, n_prototypes=10, prototype_dim=512,
                 temp_student=1., temp_teacher=1.):
        super(SDEPModel, self).__init__()
        self.teacher = SDEPModule(n_networks)
        self.student = SDEPModule(n_networks)
        # self.prototypes_ensemble = nn.ParameterList([
        #     nn.Parameter(
        #         torch.randn(config["n_prototypes"], config["prototype_dim"])
        #     ) for _ in range(config["n_networks"])]
        # )
        self.prototypes_ensemble = nn.Parameter(
            torch.randn(n_networks, n_prototypes, prototype_dim)
        )
        self.temperature_student = temp_student
        self.temperature_teacher = temp_teacher
        self.softmax = nn.Softmax(dim=-1)

    def student_forward(self, x):
        return self.student(x)

    def teacher_forward(self, x):
        return self.teacher(x)

    def get_embedding(self, x):
        return self.teacher.encoder(x)

    def forward(self, x_global, x_local):
        z_global = torch.stack(self.teacher_forward(x_global)) # n_networks x batch_size x n_prototypes
        z_local = torch.stack(self.student_forward(x_local)) # n_networks x batch_size x n_prototypes
        proto_t = self.prototypes_ensemble.transpose(1, 2)
        distances_teacher = z_global @ proto_t # n_networks x batch_size x n_prototypes
        distances_student = z_local @ proto_t # n_networks x batch_size x n_prototypes
        probs_teacher = self.softmax(distances_teacher / self.temperature_teacher) # n_networks x batch_size x n_prototypes
        probs_student = self.softmax(distances_student / self.temperature_student) # n_networks x batch_size x n_prototypes
        return probs_teacher, probs_student


class SDEPModule(nn.Module):
    def __init__(self, n_networks):
        super(SDEPModule, self).__init__()
        self.encoder = ECAPA_TDNN()
        self.projector_ensemble = nn.ModuleList([MLPProjector() for _ in range(n_networks)])

    def forward(self, x):
        x, _ = self.encoder(x)
        return [projector(x) for projector in self.projector_ensemble]


class MLPProjector(nn.Module):
    def __init__(self):
        super(MLPProjector, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.GELU(),
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class VADModel(nn.Module):
    def __init__(self, threshold=0.003):
        super(VADModel, self).__init__()
        self.backbone = ECAPA_TDNN()
        self.classification_head = lambda x: (x > threshold).float()

    def forward(self, x):
        _, attention_scores = self.backbone(x, return_attention=True)
        attention_scores = attention_scores.mean(dim=1)
        out = self.classification_head(attention_scores)
        return out


class ECAPA_TDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    Adapted from https://github.com/alibaba-damo-academy/3D-Speaker/blob/main/speakerlab/models/rdino/ECAPA_TDNN.py.
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=512,
        activation=torch.nn.ReLU,
        channels=list(x*2 for x in [512, 512, 512, 512, 1536]),
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
        log_input=True,
        n_mels=80,
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.instancenorm   = nn.InstanceNorm1d(n_mels)
        self.n_mels     = n_mels
        self.log_input  = log_input
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None, return_attention=False):
        """Returns the embedding vector.
        """
        x = x + 1e-6
        if self.log_input:
            x = x.log()
        x = self.instancenorm(x).detach()

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x, attention_scores = self.asp(x, lengths=lengths, return_attention=return_attention)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        x = x.squeeze(1)

        return x, attention_scores


class Conv1d(nn.Module):

    def __init__(
        self,
        out_channels,
        kernel_size,
        in_channels,
        stride=1,
        dilation=1,
        padding='same',
        groups=1,
        bias=True,
        padding_mode='reflect',
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.padding_mode = padding_mode

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        if self.padding == 'same':
            x = self._manage_padding(x, self.kernel_size, self.dilation,
                                     self.stride)

        elif self.padding == 'causal':
            num_pad = (self.kernel_size - 1) * self.dilation
            x = F.pad(x, (num_pad, 0))

        elif self.padding == 'valid':
            pass

        else:
            raise ValueError(
                "Padding must be 'same', 'valid' or 'causal'. Got "
                + self.padding)

        wx = self.conv(x)

        return wx

    def _manage_padding(
        self,
        x,
        kernel_size: int,
        dilation: int,
        stride: int,
    ):
        L_in = x.shape[-1]
        padding = get_padding_elem(L_in, stride, kernel_size, dilation)
        x = F.pad(x, padding, mode=self.padding_mode)

        return x


class BatchNorm1d(nn.Module):
    def __init__(
        self,
        input_size,
        eps=1e-05,
        momentum=0.1,
    ):
        super().__init__()
        self.norm = nn.BatchNorm1d(
            input_size,
            eps=eps,
            momentum=momentum,
        )

    def forward(self, x):
        return self.norm(x)


class TDNNBlock(nn.Module):
    """An implementation of TDNN.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation=nn.ReLU,
        groups=1,
    ):
        super(TDNNBlock, self).__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )
        self.activation = activation()
        self.norm = BatchNorm1d(input_size=out_channels)

    def forward(self, x):
        return self.norm(self.activation(self.conv(x)))


class Res2NetBlock(torch.nn.Module):
    """An implementation of Res2NetBlock w/ dilation.
    """
    def __init__(
        self, in_channels, out_channels, scale=8, kernel_size=3, dilation=1
    ):
        super(Res2NetBlock, self).__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        self.blocks = nn.ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x):
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y


class SEBlock(nn.Module):
    """An implementation of squeeze-and-excitation block.
    """
    def __init__(self, in_channels, se_channels, out_channels):
        super(SEBlock, self).__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, lengths=None):
        L = x.shape[-1]
        if lengths is not None:
            mask = length_to_mask(lengths * L, max_len=L, device=x.device)
            mask = mask.unsqueeze(1)
            total = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / total
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    """
    def __init__(self, channels, attention_channels=128, global_context=True):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x, lengths=None, return_attention=False):
        """Calculates mean and std for a batch (input tensor).
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=2) # I will use it as a VAD score
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        if return_attention:
            return pooled_stats, attn
        else:
            return pooled_stats, None


class SERes2NetBlock(nn.Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        res2net_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        activation=torch.nn.ReLU,
        groups=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            activation=activation,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


def length_to_mask(length, max_len=None, dtype=None, device=None):
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype).expand(
            len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def get_padding_elem(L_in: int, stride: int, kernel_size: int, dilation: int):
    if stride > 1:
        n_steps = math.ceil(((L_in - kernel_size * dilation) / stride) + 1)
        L_out = stride * (n_steps - 1) + kernel_size * dilation
        padding = [kernel_size // 2, kernel_size // 2]

    else:
        L_out = (L_in - dilation * (kernel_size - 1) - 1) // stride + 1

        padding = [(L_in - L_out) // 2, (L_in - L_out) // 2]
    return padding