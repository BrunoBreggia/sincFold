from torch import nn
from torch.nn.functional import cross_entropy, scaled_dot_product_attention, interpolate, pad
import torch as tr
from tqdm import tqdm
import pandas as pd
import math

from sincfold.metrics import contact_f1
from sincfold.utils import mat2bp, postprocessing
from sincfold._version import __version__

SINCFOLD_WEIGHTS = f'https://github.com/sinc-lab/sincFold/raw/main/weights/sincFold_weights_{__version__}.pmt'

def sincfold(pretrained=False, weights=None, **kwargs):
    """ 
    SincFold: a deep learning-based model for RNA secondary structure prediction
    pretrained (bool): Use pretrained weights
    **kwargs: Model hyperparameters
    """
    model = SincFold(**kwargs)
    if pretrained:
        print("Load pretrained weights...")
        model.load_state_dict(tr.hub.load_state_dict_from_url(SINCFOLD_WEIGHTS, map_location=tr.device(model.device)))
    else:
        if weights is not None:
            print(f"Load weights from {weights}")
            model.load_state_dict(tr.load(weights, map_location=tr.device(model.device)))
        else:
            print("No weights provided, using random initialization")
        
    return model

class SincFold(nn.Module):
    def __init__(
        self,
        train_len=0,
        embedding_dim=1,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        loss_l1=0,
        loss_beta=0,
        scheduler="none",
        verbose=True,
        interaction_prior=False,
        output_th=0.5,
        **kwargs
    ):
        """Base classifier model from embedding sequence-
        negative_weigth: not_conected/conected proportion in error weights."""
        super().__init__()

        self.device = device
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.loss_l1 = loss_l1
        self.loss_beta = loss_beta
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th

        mid_ch = 1
        self.interaction_prior = interaction_prior
        if interaction_prior != "none":
            mid_ch = 2

        # Define architecture
        self.build_graph(embedding_dim, mid_ch=mid_ch, **kwargs) 
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        # lr scheduler
        self.scheduler_name = scheduler
        if scheduler == "plateau":
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, verbose=True
            )
        elif scheduler == "cycle":
            self.scheduler = tr.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=lr, steps_per_epoch=train_len, epochs=self.config["max_epochs"]
            )
        else:
            self.scheduler = None

        self.to(device)
  
    def build_graph(
        self,
        embedding_dim,
        kernel=3,
        filters=32,
        num_layers=2,
        dilation_resnet1d=3,
        resnet_bottleneck_factor=0.5,
        mid_ch=1,
        kernel_resnet2d=5,
        bottleneck1_resnet2d=256,
        bottleneck2_resnet2d=128,
        filters_resnet2d=256,
        rank=64,
        dilation_resnet2d=3,
        **kwargs
    ):
        pad = (kernel - 1) // 2

        self.use_restrictions = mid_ch != 1

        # Reemplazo la convolucion inicial 1D por una capa de Embedding
        self.embedding = nn.Embedding(85, filters) # vocab_size=85 para 3-mers
        
        self.resnet1d = [
            #nn.Conv1d(embedding_dim, filters, kernel, padding="same")
            ]

        for k in range(num_layers):
            self.resnet1d.append(
                ResidualLayer1D(
                    dilation_resnet1d,
                    resnet_bottleneck_factor,
                    filters,
                    kernel,
                )
            )

        self.resnet1d = nn.Sequential(*self.resnet1d)

        # Calculo de Query y Key para matriz de atencion (self-attention w/1 head)
        self.WQ = nn.Linear(filters, filters) # heads=1
        self.WK = nn.Linear(filters, filters) # heads=1
        self.WV = nn.Linear(filters, filters) # heads=1
        self.multiHead = nn.MultiheadAttention(filters, num_heads=1, batch_first=True)

        # Capas para procesamiento 2D comprimido
        self.resnet2d = [nn.Conv2d(
            in_channels=1, # depends on amount of heads
            out_channels=filters_resnet2d, 
            kernel_size=7, 
            padding="same"
        )]
        self.resnet2d += [
            ResidualBlock2D(
                filters_resnet2d,
                bottleneck1_resnet2d,
                kernel_resnet2d,
                dilation_resnet2d,
            ), ResidualBlock2D(
                filters_resnet2d,
                bottleneck2_resnet2d,
                kernel_resnet2d,
                dilation_resnet2d,
            ), 
        ]
        
        self.resnet2d = nn.Sequential(*self.resnet2d)

        self.conv2Dout = nn.Conv2d(
            in_channels=filters_resnet2d,
            out_channels=1,
            kernel_size=kernel_resnet2d,
            padding="same",
        )

        # Capas para procesamiento 2D expandido
        self.resnet2d_exp = [nn.Conv2d(
            in_channels=1, # depends on amount of heads
            out_channels=filters_resnet2d, 
            kernel_size=7, 
            padding="same"
        )]
        self.resnet2d_exp += [
            ResidualBlock2D(
                filters_resnet2d,
                bottleneck1_resnet2d,
                kernel_resnet2d,
                dilation_resnet2d,
            ), ResidualBlock2D(
                filters_resnet2d,
                bottleneck2_resnet2d,
                kernel_resnet2d,
                dilation_resnet2d,
            ), 
        ]
        
        self.resnet2d_exp = nn.Sequential(*self.resnet2d_exp)

        self.conv2Dout_exp = nn.Conv2d(
            in_channels=filters_resnet2d,
            out_channels=1,
            kernel_size=kernel_resnet2d,
            padding="same",
        )


    def forward(self, batch):
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]
        Lk = x.shape[1]
        L = max(batch["length"])
        print("Original length:", L)
        
        print("Tensor shape:", x.shape)
        y = self.embedding(x.int())
        #y = y.transpose(1, 2)
        print("Tensor shape:", y.shape)

        # Self-attention
        q = self.WQ(y)
        k = self.WK(y)
        v = self.WV(y)
        _, attn_matrix = self.multiHead(q, k, v)
        # _, attn_matrix = scaled_dot_product_attention(q, k, v, need_weights=True)
        print("Attention matrix", attn_matrix.shape)

        y0 = tr.unsqueeze(attn_matrix, dim=1)
        print(y0.shape)

        # expanded = unpool_kmer_matrix(sym, L)
        k = 3
        target_size = y0.shape[-1]*k
        # new_shape = list(y0.shape)
        # new_shape[2:] = target_size, target_size
        expanded = interpolate(y0, size=(target_size,)*2, mode='nearest')
        print("Interpolated shape:", expanded.shape)
        if target_size < L:
            padding = L-target_size
            expanded = pad(expanded, (0 ,padding, 0, padding), mode="constant", value=0)  # padding -> (left, right, top, bottom)
        print("Padded shape:", expanded.shape)

        y = self.resnet2d_exp(expanded)
        y = self.conv2Dout(tr.relu(y)).squeeze(1)
        print("Convolved shape:", y.shape)

        # if batch["canonical_mask"] is not None:
        #     y = y.multiply(batch["canonical_mask"].to(self.device))

        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2

        return y, y0

    def loss_func(self, yhat, y):
        """yhat and y are [N, M, M]"""
        y = y.view(y.shape[0], -1)
        yhat, y0 = yhat  # yhat is the final ouput and y0 is the cnn output

        yhat = yhat.view(yhat.shape[0], -1)
        y0 = y0.view(y0.shape[0], -1)

        # Add l1 loss, ignoring the padding
        l1_loss = tr.mean(tr.relu(yhat[y != -1]))

        # yhat has to be shape [N, 2, L].
        yhat = yhat.unsqueeze(1)
        # yhat will have high positive values for base paired and high negative values for unpaired
        yhat = tr.cat((-yhat, yhat), dim=1)
        
        y0 = y0.unsqueeze(1)
        y0 = tr.cat((-y0, y0), dim=1)

        print("---------------------------------")
        print(y0.shape)
        print(y.shape)

        #error_loss1 = cross_entropy(y0, y, ignore_index=-1, weight=self.class_weight)
        
        error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)
    

        loss = (
            error_loss
            # + self.loss_beta * error_loss1
            + self.loss_l1 * l1_loss
        )
        return loss

    def fit(self, loader):
        self.train()
        metrics = {"loss": 0, "f1": 0}

        if self.verbose:
            loader = tqdm(loader)

        for batch in loader: 
            
            y = batch["contact"].to(self.device)
            batch.pop("contact")
            self.optimizer.zero_grad()  # Cleaning cache optimizer
            y_pred = self(batch)
            
            loss = self.loss_func(y_pred, y)
            # y_pred is a composed tensor, we need to get the final pred
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            f1 = contact_f1(
                y.cpu(), y_pred.detach().cpu(), batch["length"], method="triangular"
            )

            metrics["loss"] += loss.item()
            metrics["f1"] += f1

            loss.backward()
            self.optimizer.step()

            if self.scheduler_name == "cycle":
                    self.scheduler.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        self.eval()
        metrics = {"loss": 0, "f1": 0, "f1_post": 0}

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:  
                y = batch["contact"].to(self.device)
                batch.pop("contact")
                lengths = batch["length"]
                

                y_pred = self(batch)
                loss = self.loss_func(y_pred, y)
                metrics["loss"] += loss.item()

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = postprocessing(y_pred.cpu(), batch["canonical_mask"])

                f1 = contact_f1(y.cpu(), y_pred.cpu(), lengths, th=self.output_th, reduce=True, method="triangular")
                f1_post = contact_f1(
                    y.cpu(), y_pred_post.cpu(), lengths, th=self.output_th, reduce=True, method="triangular")

                metrics["f1"] += f1
                metrics["f1_post"] += f1_post

        for k in metrics:
            metrics[k] /= len(loader)

        if self.scheduler_name == "plateau":
            self.scheduler.step(metrics["f1_post"])

        return metrics

    def pred(self, loader, logits=False):
        self.eval()

        if self.verbose:
            loader = tqdm(loader)

        predictions, logits_list = [], [] 
        with tr.no_grad():
            for batch in loader: 
                
                lengths = batch["length"]
                seqid = batch["id"]
                sequences = batch["sequence"]


                y_pred = self(batch)
                
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = postprocessing(y_pred.cpu(), batch["canonical_mask"])

                for k in range(y_pred_post.shape[0]):
                    if logits:
                        logits_list.append(
                            (seqid[k],
                             y_pred[k, : lengths[k], : lengths[k]].squeeze().cpu(),
                             y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            ))
                    predictions.append(
                        (seqid[k],
                        sequences[k],
                            mat2bp(
                                y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            )                         
                        )
                    )
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])

        return predictions, logits_list

class ResidualLayer1D(nn.Module):
    def __init__(
        self,
        dilation,
        resnet_bottleneck_factor,
        filters,
        kernel_size,
    ):
        super().__init__()

        num_bottleneck_units = math.floor(resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(
                filters,
                num_bottleneck_units,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(num_bottleneck_units),
            nn.ReLU(),
            nn.Conv1d(
                num_bottleneck_units, 
                filters, 
                kernel_size=1, 
                padding="same"
            ),
        )

    def forward(self, x):
        return x + self.layer(x)


class ResidualBlock2D(nn.Module):
    def __init__(self, filters, filters1, kernel_size, dilation):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters1, kernel_size, padding="same"),
            nn.BatchNorm2d(filters1),
            nn.ReLU(),
            nn.Conv2d(
                filters1, filters, kernel_size, dilation=dilation, padding="same"
            ),
        )

    def forward(self, x):
        return self.layer(x) + x