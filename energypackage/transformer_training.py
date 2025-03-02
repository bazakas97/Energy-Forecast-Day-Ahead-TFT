import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################################
# 1) Gated Residual Network (GRN)
##############################################################################
class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network as described in the TFT paper:
    - Two feed-forward layers with optional context
    - Gating mechanism
    - Residual skip-connection
    - Layer normalization
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = None,
        dropout: float = 0.1,
        context_size: int = None,
        activation: nn.Module = nn.ELU()
    ):
        super().__init__()
        if output_size is None:
            output_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context_fc = nn.Linear(context_size, hidden_size, bias=False) if context_size else None

        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Gating + residual
        self.gate = nn.Linear(input_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x, context=None):
        """
        x: shape (..., input_size)
        context: shape (..., context_size) or None
        """
        # 1) First transformation
        x_out = self.fc1(x)
        if (self.context_fc is not None) and (context is not None):
            x_out = x_out + self.context_fc(context)
        x_out = self.activation(x_out)
        x_out = self.dropout(x_out)

        # 2) Second transformation
        x_out = self.fc2(x_out)

        # 3) Gating
        gating = torch.sigmoid(self.gate(x))

        # 4) Residual skip
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        # 5) Combine + layer norm
        x_out = gating * x_out + (1 - gating) * skip
        x_out = self.layer_norm(x_out)
        return x_out


##############################################################################
# 2) Variable Selection Network (VSN)
##############################################################################
class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network:
    - For each time step (or static input), transforms each variable with a GRN
    - Computes selection weights via another GRN on the concatenated inputs
    - Weighted sum across variables
    """
    def __init__(
        self,
        input_size: int,     # dimension of each variable (e.g. 1 if scalar)
        num_inputs: int,     # number of variables
        hidden_size: int,
        dropout: float = 0.1,
        context_size: int = None
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size

        # GRN for computing selection weights from concatenated inputs
        self.flattened_grn = GatedResidualNetwork(
            input_size=input_size * num_inputs,
            hidden_size=hidden_size,
            output_size=num_inputs,
            dropout=dropout,
            context_size=context_size
        )

        # GRN for each individual variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=None
            ) for _ in range(num_inputs)
        ])

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        """
        x: shape (batch, time, num_inputs, input_size) OR (batch, num_inputs, input_size) for static
        context: optional (batch, time, context_size) or (batch, context_size)

        Returns:
          combined: shape (batch, time, hidden_size) or (batch, hidden_size)
          weights:  shape (batch, time, num_inputs) or (batch, num_inputs)
        """
        is_static = (x.dim() == 3)  # (batch, num_inputs, input_size)
        if is_static:
            # Insert a time dimension
            x = x.unsqueeze(1)  # => (batch, 1, num_inputs, input_size)

        batch_size, time_steps, num_vars, in_size = x.size()

        # Flatten across variables
        x_flat = x.view(batch_size * time_steps, num_vars * in_size)

        # Reshape context
        if context is not None:
            if context.dim() == 2:
                # If shape (batch, context_size), expand over time
                context = context.unsqueeze(1).expand(-1, time_steps, -1)
            context = context.view(batch_size * time_steps, -1)

        # 1) Compute selection weights
        sparse_weights = self.flattened_grn(x_flat, context=context)   # => (batch*time, num_vars)
        sparse_weights = self.softmax(sparse_weights)
        sparse_weights = sparse_weights.view(batch_size, time_steps, num_vars)

        # 2) Transform each variable with its own GRN
        var_outputs = []
        for i in range(num_vars):
            xi = x[:, :, i, :]   # (batch, time, in_size)
            xi = xi.view(batch_size * time_steps, in_size)
            vi = self.variable_grns[i](xi)  # => (batch*time, hidden_size)
            vi = vi.view(batch_size, time_steps, self.hidden_size)
            var_outputs.append(vi)

        var_outputs = torch.stack(var_outputs, dim=2)  # (batch, time, num_vars, hidden_size)

        # 3) Weighted sum
        weights_expanded = sparse_weights.unsqueeze(-1)  # (batch, time, num_vars, 1)
        combined = (var_outputs * weights_expanded).sum(dim=2)  # (batch, time, hidden_size)

        if is_static:
            # Remove time dimension of size 1
            combined = combined.squeeze(1)       # (batch, hidden_size)
            sparse_weights = sparse_weights.squeeze(1)  # (batch, num_vars)

        return combined, sparse_weights


##############################################################################
# 3) LSTM with Gating (static enrichment)
##############################################################################
class LSTMwithGating(nn.Module):
    """
    LSTM block that can incorporate static enrichment into its hidden & cell states.
    The static context can be used to gate the final hidden/cell states.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        static_context_size: int = None
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # If we have static context, we can gate the final states
        if static_context_size is not None:
            self.hidden_gating = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=static_context_size
            )
            self.cell_gating = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                context_size=static_context_size
            )
        else:
            self.hidden_gating = None
            self.cell_gating = None

    def forward(self, x, h_c=None, static_context=None):
        """
        x: (batch, time, input_size)
        h_c: tuple of (h, c) each shape (num_layers, batch, hidden_size)
        static_context: (batch, hidden_size) if provided
        """
        output, (h, c) = self.lstm(x, h_c)

        if static_context is not None and self.hidden_gating is not None:
            # Gate only the final layer's hidden/cell for simplicity
            h_last = h[-1]  # (batch, hidden_size)
            c_last = c[-1]

            # Gate with static context
            h_gated = self.hidden_gating(h_last, context=static_context)
            c_gated = self.cell_gating(c_last, context=static_context)

            # Replace final layer states
            h[-1] = h_gated
            c[-1] = c_gated

        return output, (h, c)


##############################################################################
# 4) Multi-Head Attention (Interpretable) + gating
##############################################################################
class InterpretableMultiHeadAttention(nn.Module):
    """
    Standard multi-head attention, plus gating with a GRN.
    Used for the "Masked Interpretable Multi-Head Attention" in the TFT paper.
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Gated residual on the attention output
        self.gate = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size  # pass the original 'query' as context
        )

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: (batch, time, hidden_size)
        mask: optional attention mask
        """
        attn_out, attn_weights = self.mha(query=query, key=key, value=value, attn_mask=mask)
        # Gated residual, using query as 'context'
        out = self.gate(attn_out, context=query)
        return out, attn_weights


##############################################################################
# 5) Full TFT Model with single forward(x) interface
##############################################################################
class TFT(nn.Module):
    """
    "Full" TFT architecture but with a single forward(x, static_inputs=None) signature
    so it matches your main.py usage:
      - x: (batch, forecast_length, input_dim)
      - If static_inputs are provided, shape (batch, static_input_dim)
    Returns:
      - predictions: (batch, forecast_length, num_quantiles)
    """

    def __init__(
        self,
        input_dim: int,          # total dynamic features = # future covariates + 1 past target
        hidden_dim: int = 256,
        forecast_length: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        quantiles: list = [0.1, 0.5, 0.9],
        static_input_dim: int = 0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_length = forecast_length
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.static_input_dim = static_input_dim

        # --- (A) Static covariate encoder (if we have static features) ---
        if static_input_dim > 0:
            self.static_vsn = VariableSelectionNetwork(
                input_size=1,  # each static feature is scalar
                num_inputs=static_input_dim,
                hidden_size=hidden_dim,
                dropout=dropout,
                context_size=None
            )
        else:
            self.static_vsn = None

        # We treat the last dynamic feature as the "past observed target"
        # and the other (input_dim - 1) as known future covariates.
        self.num_past = 1
        self.num_future = input_dim - 1

        # --- (B) Variable Selection for past & future ---
        self.encoder_vsn = VariableSelectionNetwork(
            input_size=1,      # each feature is scalar
            num_inputs=self.num_past,
            hidden_size=hidden_dim,
            dropout=dropout,
            context_size=(hidden_dim if static_input_dim > 0 else None)
        )
        self.decoder_vsn = VariableSelectionNetwork(
            input_size=1,
            num_inputs=self.num_future,
            hidden_size=hidden_dim,
            dropout=dropout,
            context_size=(hidden_dim if static_input_dim > 0 else None)
        )

        # --- (C) LSTM encoder & decoder (with optional static gating) ---
        self.encoder_lstm = LSTMwithGating(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            static_context_size=(hidden_dim if static_input_dim > 0 else None)
        )
        self.decoder_lstm = LSTMwithGating(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            static_context_size=(hidden_dim if static_input_dim > 0 else None)
        )

        # --- (D) Multi-head attention (decoder -> encoder) ---
        self.attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # --- (E) Position-wise feed-forward after attention (GRN) ---
        self.post_attn_gating = GatedResidualNetwork(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            output_size=hidden_dim,
            dropout=dropout,
            context_size=hidden_dim
        )

        # --- (F) Final projection to quantiles ---
        self.output_layer = nn.Linear(hidden_dim, self.num_quantiles)

    def forward(self, x, static_inputs=None):
        """
        x: (batch, forecast_length, input_dim)
           - Last feature is the observed target
           - The first (input_dim - 1) features are known future covariates
        static_inputs: (batch, static_input_dim), optional

        Returns:
            quantile_preds: (batch, forecast_length, num_quantiles)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)  # should be forecast_length

        # --- 1) Encode static features (if any) ---
        if self.static_vsn is not None and static_inputs is not None:
            # (batch, static_input_dim) -> (batch, static_input_dim, 1)
            static_expanded = static_inputs.unsqueeze(-1)
            static_context, _ = self.static_vsn(static_expanded)  # => (batch, hidden_dim)
        else:
            static_context = None

        # --- 2) Separate the target vs. covariates ---
        # Past (target) => shape (batch, seq_len, 1)
        # Future (covariates) => shape (batch, seq_len, input_dim-1)
        x_hist = x[:, :, -1:].clone()      # (batch, seq_len, 1)
        x_future = x[:, :, : (self.input_dim - 1)].clone()  # (batch, seq_len, num_future)

        # For the VSN, we need shape (batch, seq_len, num_vars, input_size=1)
        x_hist_vsn = x_hist.unsqueeze(-1)      # => (batch, seq_len, 1, 1)
        x_future_vsn = x_future.unsqueeze(-1)  # => (batch, seq_len, (input_dim-1), 1)

        # --- 3) Variable selection on past & future ---
        enc_features, _ = self.encoder_vsn(x_hist_vsn, context=static_context)   # => (batch, seq_len, hidden_dim)
        dec_features, _ = self.decoder_vsn(x_future_vsn, context=static_context) # => (batch, seq_len, hidden_dim)

        # --- 4) LSTM encoder ---
        enc_output, (enc_h, enc_c) = self.encoder_lstm(enc_features, h_c=None, static_context=static_context)
        # enc_output: (batch, seq_len, hidden_dim)

        # --- 5) LSTM decoder ---
        dec_output, (dec_h, dec_c) = self.decoder_lstm(dec_features, h_c=(enc_h, enc_c), static_context=static_context)
        # dec_output: (batch, seq_len, hidden_dim)

        # --- 6) Multi-head attention: dec_output attends to enc_output
        attn_out, attn_weights = self.attention(dec_output, enc_output, enc_output)
        # => (batch, seq_len, hidden_dim), (batch, seq_len, seq_len)

        # --- 7) Position-wise feed-forward (GRN) to fuse attention & decoder
        fused_dec_out = self.post_attn_gating(attn_out, context=dec_output)
        # => (batch, seq_len, hidden_dim)

        # --- 8) Final projection to quantiles
        quantile_preds = self.output_layer(fused_dec_out)  # (batch, seq_len, num_quantiles)
        return quantile_preds

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    @classmethod
    def load(cls, filepath, **kwargs):
        model = cls(**kwargs)
        model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        model.eval()
        return model
