from torch import nn
from torch.nn.functional import cross_entropy
import torch as tr
from tqdm import tqdm
import pandas as pd
import math

from sincfold.metrics import contact_f1
from sincfold.utils import mat2bp, postprocessing, unpool_kmer_matrix
from sincfold._version import __version__

SINCFOLD_WEIGHTS = f'https://github.com/sinc-lab/sincFold/raw/main/weights/sincFold_weights_{__version__}.pmt'


def sincfold(pretrained=False, weights=None, **kwargs):
    """ 
    SincFold: a deep learning-based model for RNA secondary structure prediction
    
    Args:
        pretrained (bool): Use pretrained weights
        weights (str): Path to custom weights file
        **kwargs: Model hyperparameters including:
            - kmer_embedding (bool): Use k-mer tokenization (default False)
            - kmer_size (int): Size of k-mer (default 3)
            - transformer_layers (int): Number of transformer encoder layers
            - transformer_heads (int): Number of attention heads
            - transformer_dim (int): Transformer hidden dimension
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
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        loss_l1=0,
        loss_beta=0,
        scheduler="none",
        verbose=True,
        interaction_prior=False,
        output_th=0.5,
        # K-mer embedding parameters
        kmer_embedding=False,
        kmer_size=3,
        # Transformer parameters
        transformer_layers=2,
        transformer_heads=4,
        transformer_dim=128,
        **kwargs
    ):
        """SincFold model with optional k-mer tokenization and transformer encoder.
        
        Architecture options:
        - Original mode (kmer_embedding=False): 
            Nucleotide-level 1D conv → pairwise matrix → 2D conv → output
        - K-mer mode (kmer_embedding=True):
            K-mer embedding → 1D conv → Transformer encoder → attention matrix 
            → unpooling to LxL → 2D conv → output
            
        Args:
            train_len: Number of training samples (for scheduler)
            embedding_dim: Embedding dimension (4 for nucleotides, 64+ for k-mers)
            device: Device to run on
            negative_weight: Weight for negative class in loss
            lr: Learning rate
            loss_l1: L1 regularization weight
            loss_beta: Beta loss weight
            scheduler: Learning rate scheduler type
            verbose: Print progress
            interaction_prior: Use interaction prior
            output_th: Output threshold
            kmer_embedding: Use k-mer tokenization instead of nucleotide-level
            kmer_size: Size of k-mer (default 3)
            transformer_layers: Number of transformer encoder layers
            transformer_heads: Number of attention heads in transformer
            transformer_dim: Hidden dimension of transformer
        """
        super().__init__()

        self.device = device
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.loss_l1 = loss_l1
        self.loss_beta = loss_beta
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th
        
        # K-mer specific parameters
        self.kmer_embedding = kmer_embedding
        self.kmer_size = kmer_size
        
        mid_ch = 1
        self.interaction_prior = interaction_prior
        if interaction_prior != "none":
            mid_ch = 2

        # Build the model graph
        self.build_graph(embedding_dim, mid_ch=mid_ch, 
                        transformer_layers=transformer_layers,
                        transformer_heads=transformer_heads,
                        transformer_dim=transformer_dim,
                        **kwargs)
        self.optimizer = tr.optim.Adam(self.parameters(), lr=lr)

        # Learning rate scheduler
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
        # Transformer parameters
        transformer_layers=2,
        transformer_heads=4,
        transformer_dim=128,
        **kwargs
    ):
        """Build the model architecture.
        
        For k-mer embedding mode:
            1. 1D ResNet: processes k-mer embeddings [B, vocab_size, L_k] -> [B, filters, L_k]
            2. Transformer Encoder: produces attention matrix [B, L_k, L_k]
            3. Unpooling: expands to [B, L, L]
            4. 2D ResNet: refines contact matrix -> output [B, L, L]
        
        For original (nucleotide) mode:
            1. 1D ResNet: [B, 4, L] -> [B, filters, L]
            2. Rank projection: produces pairwise matrix [B, L, L]
            3. 2D ResNet: refines contact matrix -> output [B, L, L]
        """
        pad = (kernel - 1) // 2

        self.use_restrictions = mid_ch != 1
        
        # Store transformer config
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dim = transformer_dim

        # ===================================================================
        # 1D ResNet: Process embeddings (works for both k-mer and nucleotide)
        # ===================================================================
        self.resnet1d = [nn.Conv1d(embedding_dim, filters, kernel, padding="same")]

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

        # ===================================================================
        # Path 1: Original rank projection (for nucleotide-level or as fallback)
        # ===================================================================
        self.convrank1 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )
        self.convrank2 = nn.Conv1d(
            in_channels=filters,
            out_channels=rank,
            kernel_size=kernel,
            padding=pad,
            stride=1,
        )

        # ===================================================================
        # Path 2: Transformer Encoder (for k-mer embedding mode)
        # Produces attention matrix at k-mer resolution
        # ===================================================================
        if self.kmer_embedding:
            # Project 1D ResNet output to transformer dimension
            self.transformer_proj = nn.Linear(filters, transformer_dim)
            
            # Transformer encoder layer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=transformer_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers
            )
            
            # Project transformer output back to rank dimension for pairwise attention
            # This gives us the "attention matrix" at k-mer resolution
            self.transformer_output_proj = nn.Linear(transformer_dim, rank)

        # ===================================================================
        # 2D ResNet: Process contact matrix at nucleotide resolution
        # ===================================================================
        self.resnet2d = [nn.Conv2d(
            in_channels=mid_ch, out_channels=filters_resnet2d, kernel_size=7, padding="same"
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
            )
        ]
        
        self.resnet2d = nn.Sequential(*self.resnet2d)

        self.conv2Dout = nn.Conv2d(
            in_channels=filters_resnet2d,
            out_channels=1,
            kernel_size=kernel_resnet2d,
            padding="same",
        )

    def forward(self, batch):
        """Forward pass of the model.
        
        For k-mer embedding mode:
            1. 1D ResNet on k-mer embeddings [B, vocab_size, L_k]
            2. Transformer encoder produces attention at k-mer resolution
            3. Unpool to nucleotide resolution [B, L, L]
            4. 2D ResNet for refinement
            
        For original (nucleotide) mode:
            1. 1D ResNet on nucleotide embeddings [B, 4, L]
            2. Rank projection produces pairwise matrix
            3. 2D ResNet for refinement
        """
        x = batch["embedding"].to(self.device)  # [B, vocab_size, L_k] or [B, 4, L]
        batch_size = x.shape[0]
        
        # Get lengths from batch
        L = batch["length"][0] if isinstance(batch["length"], list) else batch["length"][0]
        if isinstance(L, tr.Tensor):
            L = L.item()
        
        L_k = x.shape[2]  # Contracted length (L_k = L - k + 1 for k-mer, or L for nucleotide)
        
        # ===================================================================
        # Step 1: 1D ResNet processing
        # ===================================================================
        y = self.resnet1d(x)  # [B, filters, L_k]
        
        if self.kmer_embedding:
            # ===============================================================
            # K-MER PATH: Transformer + Unpooling
            # ===============================================================
            
            # Project to transformer dimension
            y_trans = tr.transpose(y, -2, -1)  # [B, L_k, filters]
            y_trans = self.transformer_proj(y_trans)  # [B, L_k, transformer_dim]
            
            # Transformer encoder - produces contextualized k-mer representations
            # The self-attention in transformer naturally produces pairwise attention
            y_trans = self.transformer_encoder(y_trans)  # [B, L_k, transformer_dim]
            
            # Create pairwise attention matrix from transformer output
            # Method: use dot product of transformer outputs to create attention matrix
            y_trans_proj = self.transformer_output_proj(y_trans)  # [B, L_k, rank]
            
            # Symmetrize the attention matrix
            ya = y_trans_proj  # [B, L_k, rank]
            yb = tr.transpose(ya, -2, -1)  # [B, rank, L_k]
            y_attn = ya @ yb / math.sqrt(self.transformer_dim)  # [B, L_k, L_k] - scaled dot product
            
            # Symmetrize
            yt = tr.transpose(y_attn, -1, -2)
            y_attn = (y_attn + yt) / 2
            
            # Store intermediate output at k-mer resolution for loss computation
            y0_kmer = y_attn.view(-1, L_k, L_k)
            
            # ===============================================================
            # UNPOOLING: Expand from k-mer resolution to nucleotide resolution
            # ===============================================================
            # Get actual sequence lengths for each sample in batch
            lengths = batch["length"]
            if isinstance(lengths, list):
                max_L = max(lengths)
            else:
                max_L = lengths.max().item()
            
            # Unpool the k-mer attention matrix to nucleotide resolution
            # This is the key step: (L-k+1) x (L-k+1) -> L x L
            y0 = unpool_kmer_matrix(y_attn, max_L, k=self.kmer_size)
            # y0 shape: [B, L, L]
            
            L = max_L  # Update L to full sequence length
            
        else:
            # ===============================================================
            # ORIGINAL PATH: Rank projection (no transformer)
            # ===============================================================
            ya = self.convrank1(y)
            ya = tr.transpose(ya, -1, -2)

            yb = self.convrank2(y)

            y = ya @ yb
            yt = tr.transpose(y, -1, -2)
            y = (y + yt) / 2

            y0 = y.view(-1, L_k, L_k)
            y0_kmer = y0  # Same as y0 for non-kmer mode

        # ===================================================================
        # Step 2: Interaction prior (optional)
        # ===================================================================
        if self.interaction_prior != "none":
            prob_mat = batch["interaction_prior"].to(self.device)
            x1 = tr.zeros([batch_size, 2, L, L]).to(self.device)
            x1[:, 0, :, :] = y0
            x1[:, 1, :, :] = prob_mat
        else:
            x1 = y0.unsqueeze(1)

        # ===================================================================
        # Step 3: 2D ResNet processing for refinement
        # ===================================================================
        y = self.resnet2d(x1)
        
        # Output projection
        y = self.conv2Dout(tr.relu(y)).squeeze(1)
        
        # Apply canonical mask if provided
        if batch["canonical_mask"] is not None:
            y = y.multiply(batch["canonical_mask"].to(self.device))
        
        # Symmetrize output
        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2

        return y, y0_kmer

    def loss_func(self, yhat, y):
        """Compute loss.
        
        Args:
            yhat: Tuple of (final_output, intermediate_output)
                  - final_output: [N, L, L] - final prediction at nucleotide resolution
                  - intermediate_output: [N, L_k, L_k] - 1D path output (k-mer or rank resolution)
            y: Ground truth contact matrix [N, L, L] at nucleotide resolution
        """
        y = y.view(y.shape[0], -1)
        yhat, y0 = yhat  # yhat is the final output and y0 is the 1D path output

        yhat = yhat.view(yhat.shape[0], -1)
        
        # Add l1 loss, ignoring the padding
        l1_loss = tr.mean(tr.relu(yhat[y != -1]))

        # yhat has to be shape [N, 2, L].
        yhat = yhat.unsqueeze(1)
        # yhat will have high positive values for base paired and high negative values for unpaired
        yhat = tr.cat((-yhat, yhat), dim=1)
        
        # Compute loss on final output
        error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)
        
        # Compute loss on intermediate 1D output
        # In k-mer mode, y0 is at k-mer resolution while y is at nucleotide resolution
        # We need to either skip this loss or pool y to k-mer resolution
        if self.kmer_embedding:
            # For k-mer mode: skip the intermediate loss since resolutions don't match
            # Or could implement pooling here if desired
            error_loss1 = tr.tensor(0.0, device=yhat.device)
        else:
            # Original mode: both at same resolution
            y0 = y0.view(y0.shape[0], -1)
            y0 = y0.unsqueeze(1)
            y0 = tr.cat((-y0, y0), dim=1)
            error_loss1 = cross_entropy(y0, y, ignore_index=-1, weight=self.class_weight)

        loss = (
            error_loss
            + self.loss_beta * error_loss1
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
            nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"),
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