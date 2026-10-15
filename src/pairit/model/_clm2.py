import importlib.resources
import random
from collections.abc import Callable

import lightning.pytorch as pl
import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaConfig, LlamaForCausalLM, get_scheduler, pipeline

from pairit.constants import SchedulerType
from pairit.tokenization import PmlmTokenizer, PmlmTokenizerTransform, SmilesTokenizerFast
from pairit.transforms import Transform

from ._clm_configuration import PCLM_CONFIG_ARGS

# Import PEFT for LoRA support
from peft import LoraConfig, get_peft_model

# Import RDKit and atomic_datasets for validity checking
try:
    from rdkit import Chem
    from atomic_datasets.utils.rdkit import is_molecule_sane
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Validity checking will be disabled.")


class PairITCLM2(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "CLM_mini",
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.98,
        eps: float = 1e-12,
        num_training_steps: int = 10_000,
        num_warmup_steps: int = 1000,
        transform_fn: Callable | Transform | None = None,
        tokenizer_dir: str | None = "pmlm_tokenizer",
        ckpt_path: str = None,
        max_length: int = 512,
        num_key_value_heads: int = None,
        attention_bias: bool = False,
        scheduler: SchedulerType = "constant_with_warmup",
        model_kwargs: dict = None,
        scheduler_kwargs: dict = None,
        num_validation_generations: int = 100,  # Changed from validation_fraction
        # LoRA parameters
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: list = None,
        lora_bias: str = "none",
        lora_task_type: str = "CAUSAL_LM",
        # Conditional loss parameters
        conditional_loss_only: bool = False,
        # Optional: compute/log split losses for (before SEP) vs (after SEP)
        compute_split_loss: bool = False,
        # Input token masking for regularization
        mask_rate: float = 0.0,
        weight_decay: float = 0.01,
    ):
        """
        Causal language model over SMILES strings.

        Parameters
        ----------
        model_name: one of PCLM_CONFIG_ARGS.keys()
        num_key_value_heads: This is the number of key_value heads that should be used to implement
            Grouped Query Attention. If`num_key_value_heads=num_attention_heads`, the model will
            use Multi Head Attention (MHA), if `num_key_value_heads=1 the model will use
            Multi Query Attention (MQA) otherwise GQA is used.
        scheduler: str, optional
            The type of learning rate scheduler.
        model_kwargs: dict, optional
            Additional keyword arguments to pass to the model.
        scheduler_kwargs: dict, optional
            Additional keyword arguments to pass to the scheduler.
        num_validation_generations: int, optional
            Number of molecules to generate for validity checking (default: 100).
        use_lora: bool, optional
            Whether to use LoRA fine-tuning (default: False).
        lora_rank: int, optional
            LoRA rank parameter (default: 16).
        lora_alpha: int, optional
            LoRA alpha parameter (default: 32).
        lora_dropout: float, optional
            LoRA dropout rate (default: 0.1).
        target_modules: list, optional
            List of module names to apply LoRA to (default: None).
        lora_bias: str, optional
            LoRA bias setting (default: "none").
        lora_task_type: str, optional
            LoRA task type (default: "CAUSAL_LM").
        conditional_loss_only: bool, optional
            If True, compute loss only on the second molecule (after SEP token).
            If False, compute loss on the entire sequence (default: False).
        mask_rate: float, optional
            Fraction of non-special input tokens to replace with <mask> during
            training as regularization. 0.0 disables masking (default).
            Only applied during training, not validation.

        """
        super().__init__()
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps

        self.model_name = model_name
        self._num_training_steps = num_training_steps
        self._num_warmup_steps = num_warmup_steps
        # Be robust to Hydra/bash quoting like '\"/abs/path.ckpt\"'
        if isinstance(ckpt_path, str):
            s = ckpt_path.strip()
            if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
                s = s[1:-1].strip()
            ckpt_path = s
        self._ckpt_path = ckpt_path
        self._tokenizer_dir = tokenizer_dir
        self._max_length = max_length
        self._attention_bias = attention_bias
        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self._num_validity_checks = num_validation_generations  # Store number of validity checks
        model_kwargs = model_kwargs or {}
        
        # LoRA parameters
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.lora_bias = lora_bias
        self.lora_task_type = lora_task_type
        
        # Conditional loss parameters
        self.conditional_loss_only = conditional_loss_only
        self.compute_split_loss = bool(compute_split_loss)
        self._mask_rate = mask_rate
        self._weight_decay = weight_decay

        if self._tokenizer_dir is not None:
            path = importlib.resources.files("pairit") / "assets" / self._tokenizer_dir
            #self.tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False) #SmilesTokenizerFast(padding="max_length", truncation=True, max_length=self._max_length) #
            self._transform_fn = transform_fn #or PmlmTokenizerTransform(
            #     path,
            #     padding="max_length",
            #     truncation=True,
            #     max_length=self._max_length,
            #     mlm=False,
            # )
            self.tokenizer = self._transform_fn.tokenizer # what did from_pretrained do? 

        config_args = PCLM_CONFIG_ARGS[model_name]
        if num_key_value_heads is None:
            num_key_value_heads = config_args["num_attention_heads"]
        self._num_key_value_heads = num_key_value_heads

        config = LlamaConfig(
            **config_args,
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            vocab_size=len(self.tokenizer.get_vocab()),
            max_position_embeddings=self._max_length,
            num_key_value_heads=self._num_key_value_heads,
            attention_bias=self._attention_bias,
            **model_kwargs,
        )
        self.model = LlamaForCausalLM(config)
        self.config = self.model.config

        # Apply LoRA if enabled (before loading checkpoint)
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type=self.lora_task_type,
            )

            self.model.use_cache = False
            self.model = get_peft_model(self.model, lora_config)
            print(f"LoRA enabled: {self.model.num_parameters():,} total parameters")
            print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
            # For LoRA models, load checkpoint manually to avoid parameter mismatch
            if self._ckpt_path is not None:
                print(f"Loading checkpoint for LoRA model from: {self._ckpt_path}")
                checkpoint = torch.load(self._ckpt_path, map_location='cpu', weights_only=False)
                # Load only the base model parameters, skip LoRA parameters
                if 'state_dict' in checkpoint:
                    model_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
                else:
                    model_state_dict = checkpoint
                
                # Filter out LoRA parameters and map keys for LoRA-wrapped model
                base_model_state_dict = {}
                for k, v in model_state_dict.items():
                    if not any(lora_key in k for lora_key in ['lora_A', 'lora_B', 'lora_embedding']):
                        # Map checkpoint keys to LoRA-wrapped model keys
                        # checkpoint: "model.embed_tokens.weight" -> LoRA model: "base_model.model.embed_tokens.weight"
                        if k.startswith('model.'):
                            mapped_key = 'base_model.' + k
                        else:
                            mapped_key = k
                        base_model_state_dict[mapped_key] = v
                
                # Load base model weights
                self.model.load_state_dict(base_model_state_dict, strict=False)
                print("Base model checkpoint loaded successfully for LoRA")
        
        self.save_hyperparameters(logger=False)

    def on_load_checkpoint(self, checkpoint):
        """
        Override checkpoint loading to prevent parameter mismatch for LoRA models.
        """
        if self.use_lora and self._ckpt_path is not None:
            print("Skipping Lightning checkpoint loading for LoRA model (already loaded manually)")
            return
        # For non-LoRA models, use default behavior
        super().on_load_checkpoint(checkpoint)

    def load_state_dict(self, state_dict: dict, strict: bool = True):  # type: ignore[override]
        """
        Checkpoint compatibility shim for tokenizer-vocab changes.

        When we append new special tokens to the tokenizer, the model is initialized with a
        larger `vocab_size` than older checkpoints. Lightning will then try to load
        `embed_tokens.weight` / `lm_head.weight` with a mismatched first dimension.

        We pad/truncate these matrices so old checkpoints can be loaded into newer models
        (and vice versa), preserving the shared prefix of token IDs.
        """
        try:
            current = super().state_dict()
            updated = dict(state_dict)

            def _resize_matrix(key: str):
                if key not in updated or key not in current:
                    return
                w_ckpt = updated[key]
                w_cur = current[key]
                if getattr(w_ckpt, "shape", None) is None or getattr(w_cur, "shape", None) is None:
                    return
                if w_ckpt.shape == w_cur.shape:
                    return
                # Expect [vocab, dim]
                if len(w_ckpt.shape) != 2 or len(w_cur.shape) != 2:
                    return
                new_w = w_cur.clone()
                n = min(int(w_ckpt.shape[0]), int(w_cur.shape[0]))
                d = min(int(w_ckpt.shape[1]), int(w_cur.shape[1]))
                new_w[:n, :d] = w_ckpt[:n, :d]
                updated[key] = new_w

            # Common HF LlamaForCausalLM keys under Lightning prefix `model.`
            _resize_matrix("model.model.embed_tokens.weight")
            _resize_matrix("model.lm_head.weight")

            # Also handle non-nested variants just in case (older checkpoints / different wrappers)
            _resize_matrix("model.embed_tokens.weight")
            _resize_matrix("lm_head.weight")

            return super().load_state_dict(updated, strict=False)
        except Exception:
            # Fall back to default behavior if anything unexpected happens.
            return super().load_state_dict(state_dict, strict=strict)

    def training_step(self, batch, batch_idx):
        loss, _logits, extra = self._compute_loss(batch, compute_split_loss=self.compute_split_loss)
        ppl = torch.exp(loss)

        self.log("train_loss", loss, sync_dist=True)
        self.log("train_perplexity", ppl, sync_dist=True)
        if self._mask_rate > 0:
            self.log("train_mask_rate", self._mask_rate, sync_dist=True)
        if isinstance(extra, dict):
            if "loss_first" in extra and extra["loss_first"] is not None:
                self.log("train_loss_first", extra["loss_first"], sync_dist=True)
            if "loss_second" in extra and extra["loss_second"] is not None:
                self.log("train_loss_second", extra["loss_second"], sync_dist=True)

        return {"loss": loss}

    def _has_trainer(self) -> bool:
        """Check if the model is attached to a trainer."""
        return hasattr(self, '_trainer') and self._trainer is not None

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        loss, _logits, extra = self._compute_loss(batch, compute_split_loss=self.compute_split_loss)
        ppl = torch.exp(loss)

        # Skip logging if no trainer is attached (e.g., during standalone evaluation)
        if not self._has_trainer():
            return {"loss": loss}

        # Keep checkpoint monitoring stable across 1 vs N val dataloaders.
        # When multiple val dataloaders are present, log the *global* checkpoint metric
        # only for the first loader to avoid duplicate key collisions.
        if int(dataloader_idx) == 0:
            # Without add_dataloader_idx=False, Lightning may suffix metrics with `/dataloader_idx_X`.
            # on_step=True logs per batch, on_epoch=True logs aggregated value at epoch end
            self.log("val_loss", loss, sync_dist=True, add_dataloader_idx=False, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val_perplexity", ppl, sync_dist=True, add_dataloader_idx=False, on_step=True, on_epoch=True, prog_bar=True)

        # Optional: per-dataset metrics when available (MixedMoleculeImprovementDataModule sets this).
        ds_name = None
        if isinstance(batch, dict):
            ds_name = batch.get("dataset_name", None)
        if ds_name:
            self.log(f"val/{ds_name}/loss", loss, sync_dist=True, add_dataloader_idx=False)
            self.log(f"val/{ds_name}/perplexity", ppl, sync_dist=True, add_dataloader_idx=False)
        if isinstance(extra, dict):
            if "loss_first" in extra and extra["loss_first"] is not None:
                self.log("val_loss_first", extra["loss_first"], sync_dist=True)
            if "loss_second" in extra and extra["loss_second"] is not None:
                self.log("val_loss_second", extra["loss_second"], sync_dist=True)

        # Avoid returning a key that collides with a logged metric name; Lightning may treat
        # returned step outputs as metrics and raise if the same key was logged with
        # different options.
        return {"loss": loss}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._lr,
            betas=(self._beta1, self._beta2),
            eps=self._eps,
            weight_decay=self._weight_decay,
        )

        scheduler = get_scheduler(
            self.scheduler,
            optimizer,
            num_training_steps=self._num_training_steps,
            num_warmup_steps=self._num_warmup_steps,
            scheduler_specific_kwargs=self.scheduler_kwargs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _compute_loss(self, batch, compute_split_loss: bool = False, **_ignored_kwargs):
        """
        this is how loss is computed internally i think
        https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/gpt2/modeling_gpt2.py#L1043
        logits = output["logits"]
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        PmlmTokenizerTransform prepares the labels, loss computation happens in model.forward()
        """

        if "labels" in batch.keys():
            labels = batch["labels"].squeeze(1)
        else:   
            # bit inconsistent to do this here vs in the datamodule
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100 # new based on docs https://huggingface.co/docs/transformers/en/model_doc/llama#transformers.LlamaForCausalLM
            labels = labels.squeeze(1)

        # Keep an unmasked copy for split-loss computation (before applying conditional_loss_only)
        labels_raw = labels

        # If conditional_loss_only is True, mask out the first molecule tokens
        if self.conditional_loss_only:
            labels = self._mask_first_molecule_tokens(labels, batch["attention_mask"].squeeze(1))

        # why are we squeezing?
        # something goes wrong when you squeeze unconditionally (without specifying the dimension) -- it's a bug when the batch size is 1
        input_ids = batch["input_ids"].squeeze(1)

        # Apply input token masking during training only (regularization)
        if self.training and self._mask_rate > 0:
            input_ids = self._mask_input_tokens(input_ids)

        output = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=batch["attention_mask"].squeeze(1),
        )
        loss = output["loss"]
        logits = output["logits"]

        extra = None
        if compute_split_loss:
            # Compute split losses (first vs second molecule) from the same logits.
            # We define:
            # - first: tokens strictly before the first <sep>
            # - second: tokens strictly after the first <sep> (i.e., same mask as conditional_loss_only)
            sep_id = int(self.tokenizer.sep_token_id)
            input_ids = batch["input_ids"].squeeze(1)
            labels_first = labels_raw.clone()
            labels_second = labels_raw.clone()

            # Mask based on SEP position in the *labels/input* stream
            for i in range(input_ids.shape[0]):
                sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
                if len(sep_positions) == 0:
                    # No SEP => treat entire sequence as "first"
                    labels_second[i, :] = -100
                    continue
                sep_pos = int(sep_positions[0].item())
                # first: mask sep and everything after
                labels_first[i, sep_pos:] = -100
                # second: mask everything up to and including sep
                labels_second[i, : sep_pos + 1] = -100

            # Shift to align next-token prediction (same as HF causal LM loss)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_first = labels_first[..., 1:].contiguous()
            shift_second = labels_second[..., 1:].contiguous()

            import torch.nn.functional as F

            def _masked_mean_ce(sl, lab):
                # lab: [B, L], sl: [B, L, V]
                valid = (lab != -100)
                n = int(valid.sum().item())
                if n == 0:
                    return None, 0
                loss_sum = F.cross_entropy(
                    sl.view(-1, sl.size(-1)),
                    lab.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                return (loss_sum / float(n)), n

            loss_first, n_first = _masked_mean_ce(shift_logits, shift_first)
            loss_second, n_second = _masked_mean_ce(shift_logits, shift_second)
            extra = {
                "loss_first": loss_first,
                "loss_second": loss_second,
                "n_tokens_first": n_first,
                "n_tokens_second": n_second,
            }

        return loss, logits, extra

    def _mask_input_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Randomly replace a fraction of non-special input tokens with <mask>.

        Special tokens (CLS, SEP, EOS, PAD) are never masked.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs of shape [batch_size, seq_len].

        Returns
        -------
        torch.Tensor
            Input IDs with random tokens replaced by mask_token_id.
        """
        masked = input_ids.clone()
        rand = torch.rand(masked.shape, device=masked.device)

        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        }
        is_special = torch.zeros_like(masked, dtype=torch.bool)
        for sid in special_ids:
            if sid is not None:
                is_special |= (masked == sid)

        mask_positions = (rand < self._mask_rate) & ~is_special
        masked[mask_positions] = self.tokenizer.mask_token_id
        return masked

    def _mask_first_molecule_tokens(self, labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mask out tokens from the first molecule (before SEP token) in the labels.
        
        Parameters
        ----------
        labels : torch.Tensor
            Labels tensor of shape [batch_size, seq_len]
        attention_mask : torch.Tensor
            Attention mask tensor of shape [batch_size, seq_len]
            
        Returns
        -------
        torch.Tensor
            Labels tensor with first molecule tokens masked out (set to -100)
        """
        sep_token_id = self.tokenizer.sep_token_id
        masked_labels = labels.clone()
        
        for i in range(labels.shape[0]):
            # Find SEP token position in this sequence
            sep_positions = (labels[i] == sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_positions) > 0:
                # Mask everything up to and including the SEP token
                sep_pos = sep_positions[0]
                masked_labels[i, :sep_pos + 1] = -100
        
        return masked_labels

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model for ONNX compiling.

        Parameters
        ----------
        input_ids: torch.Tensor
            The input tensor.
        attention_mask: torch.Tensor
            The attention mask tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.

        """
        preds = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = preds["hidden_states"]  # hidden reps (B, L, H)

        hidden_states = torch.stack(hidden_states, dim=1)  # (B, num layers, L, H)

        return hidden_states

    def sample(
        self,
        seed_seq: str = "M",
        max_length: int = 100,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 950,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
    ):
        # if wanted to avoid adding start and end tokens: self.tokenizer._tokenizer.post_processor = None 
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        
        outseqs = generator(
            seed_seq,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # "this is an awkward expression to step into" -sam
        outseqs = [samp["generated_text"].replace("\n", "") for samp in outseqs]

        return outseqs

    def sequences_to_log_likelihoods(self, sequences: list[str]) -> torch.Tensor:
        outputs = [self.get_nll_and_logits(s) for s in sequences]
        nlls = torch.stack([o[0] for o in outputs])
        return -nlls

    def batch_to_log_likelihoods(self, batch) -> torch.Tensor:
        """
        Adapted from https://github.com/huggingface/transformers/blob/v4.35.0/src/transformers/models/gpt2/modeling_gpt2.py#L1043
        tok_seqs shape: (B, L) of tokenized sequences
        """
        tok_seqs = batch["input_ids"].squeeze(1).to(self.device)
        BZ, _ = tok_seqs.shape
        with torch.no_grad():
            outputs = self.model(input_ids=tok_seqs, labels=tok_seqs)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = tok_seqs[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            loss_per_tok = loss.view(BZ, -1)

            # Compute loss over non-padding tokens but includes CLS and EOS
            mask = shift_labels != self.tokenizer.pad_token_id

        return -1 * torch.where(mask, loss_per_tok, torch.zeros_like(loss_per_tok)).sum(axis=1) / mask.sum(axis=1)

    def get_nll_and_logits(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor(self.tokenizer(sequence)["input_ids"], device=self.device).unsqueeze(0)
        with torch.inference_mode():
            outputs = self.model(input_ids, labels=input_ids)
        nll, logits = outputs.loss, outputs.logits  # (B, L, V), includes CLS and EOS

        return nll, logits

    def sequences_to_latents(self, sequences: list[str]) -> torch.Tensor:
        input_ids = torch.concat([toks["input_ids"].to(self.device) for toks in self._transform_fn(sequences)])
        with torch.inference_mode():
            hidden_states = self.model(input_ids=input_ids, output_hidden_states=True)["hidden_states"]  # [-1]

        return hidden_states

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
