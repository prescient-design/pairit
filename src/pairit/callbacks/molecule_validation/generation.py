"""
Molecule Generation
===================

Handles conditional and unconditional molecule generation.
"""

from typing import List, Tuple, Optional
import torch
import time


class MoleculeGenerator:
    """
    Handles molecule generation for validation.
    
    Provides methods for both conditional (given input) and unconditional
    (from scratch) molecule generation.
    """
    
    @staticmethod
    def generate_conditional(pl_module, unmasked_ids: torch.Tensor, temperature: float = 0.95) -> Tuple[torch.Tensor, float]:
        """
        Generate a molecule conditionally from a conditioning sequence.
        
        Matches the original implementation in _molecule_validation_callback.py line 783-800.
        
        Args:
            pl_module: PyTorch Lightning module with model.generate() method
            unmasked_ids: Conditioning token IDs, shape [seq_len]
            temperature: Sampling temperature (default: 0.95)
            
        Returns:
            Tuple of (generated_tokens, generation_time_seconds)
        """
        start_time = time.time()
        
        # Generate using pl_module.model.generate (NOT pl_module.generate)
        generated = pl_module.model.generate(
            input_ids=unmasked_ids.unsqueeze(0).to(pl_module.model.device),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=pl_module._max_length, 
            pad_token_id=pl_module.tokenizer.pad_token_id,
            eos_token_id=pl_module.tokenizer.eos_token_id,
        )
        
        generation_time = time.time() - start_time
        
        return generated, generation_time
    
    @staticmethod
    def batch_generate_conditional(pl_module, input_ids: torch.Tensor, 
                                   attention_mask: torch.Tensor, temperature: float = 0.95,
                                   num_generations_per_seed: int = 1,
                                   pair_seeds: Optional[List[int]] = None) -> List[torch.Tensor]:
        """
        Batch generate molecules using conditional generation.
        
        Supports generating multiple outputs per conditioning molecule by repeating
        each conditioning sequence num_generations_per_seed times.
        
        Args:
            pl_module: PyTorch Lightning module
            input_ids: Batch of input token IDs, shape [batch_size, seq_len]
            attention_mask: Batch of attention masks, shape [batch_size, seq_len]
            temperature: Sampling temperature (default: 0.95)
            num_generations_per_seed: Number of generations per conditioning molecule (default: 1)
            
        Returns:
            List of generated token tensors. Length = batch_size * num_generations_per_seed.
            Results are ordered as: [seed0_gen0, seed0_gen1, ..., seed1_gen0, seed1_gen1, ...]
        """
        from .smiles_utils import SmilesUtils

        if pair_seeds is not None:
            if len(pair_seeds) != len(input_ids):
                raise ValueError("One generation seed is required per input pair")
            device = torch.device(pl_module.model.device)
            device_index = device.index if device.index is not None else 0
            devices = [device_index] if device.type == "cuda" else []
            result = []
            for i, seed in enumerate(pair_seeds):
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(seed)
                    if devices:
                        torch.cuda.default_generators[device_index].manual_seed(seed)
                    result.extend(MoleculeGenerator.batch_generate_conditional(
                        pl_module, input_ids[i:i + 1], attention_mask[i:i + 1],
                        temperature=temperature, num_generations_per_seed=num_generations_per_seed,
                    ))
            return result
        
        batch_size = input_ids.shape[0]
        
        # Extract conditioning sequences for all samples in the batch
        conditioning_sequences = []
        valid_indices = []
        
        for i in range(batch_size):
            cond_ids, cond_mask = SmilesUtils.extract_conditioning_sequence(
                input_ids[i], attention_mask[i], 
                pl_module.tokenizer.sep_token_id, 
                pl_module.tokenizer.eos_token_id,
                include_separator=True  # Include SEP token for conditional generation
            )
            
            # Filter out masked tokens (e.g., padding)
            unmasked_ids = cond_ids[cond_mask.bool()]

            if len(unmasked_ids) > 0:  # Only add non-empty sequences
                conditioning_sequences.append(unmasked_ids)
                valid_indices.append(i)
        
        if not conditioning_sequences:
            return [torch.tensor([]) for _ in range(batch_size * num_generations_per_seed)]
        
        # Repeat each conditioning sequence num_generations_per_seed times
        # This allows us to generate multiple outputs per seed in one batch
        repeated_conditioning_sequences = []
        repeated_valid_indices = []
        for seq, valid_idx in zip(conditioning_sequences, valid_indices):
            for _ in range(num_generations_per_seed):
                repeated_conditioning_sequences.append(seq)
                repeated_valid_indices.append(valid_idx)
        
        # Pad conditioning sequences to the same length
        max_cond_len = max(len(seq) for seq in repeated_conditioning_sequences)
        target_device = pl_module.model.device
        
        # Use pad_token_id for LEFT padding (required for generation)
        pad_token_id = pl_module.tokenizer.pad_token_id
        padded_conditioning = torch.full(
            (len(repeated_conditioning_sequences), max_cond_len),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=target_device
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask_padded = torch.zeros(
            len(repeated_conditioning_sequences), max_cond_len, dtype=torch.long, device=target_device
        )

        # LEFT PADDING: place sequences at the END of the padded tensor
        for i, seq in enumerate(repeated_conditioning_sequences):
            seq_len = len(seq)
            start_pos = max_cond_len - seq_len  # Calculate left-padding offset
            padded_conditioning[i, start_pos:] = seq.to(target_device)
            attention_mask_padded[i, start_pos:] = 1  # Mark real tokens
        
        # Generate all molecules in a single batch using pl_module.model.generate
        generated_sequences = pl_module.model.generate(
            input_ids=padded_conditioning,
            attention_mask=attention_mask_padded,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=pl_module._max_length, 
            pad_token_id=pl_module.tokenizer.pad_token_id,
            eos_token_id=pl_module.tokenizer.eos_token_id,
        )
        if len(generated_sequences) != len(repeated_valid_indices):
            raise ValueError("Generator returned fewer or more attempts than requested")
        
        # Create result list with empty tensors for invalid samples
        # Result has batch_size * num_generations_per_seed elements
        result = [torch.tensor([]) for _ in range(batch_size * num_generations_per_seed)]
        
        # Fill in the valid generated sequences
        # Map from repeated index to output index: seed_idx * num_generations_per_seed + gen_idx
        for i, valid_idx in enumerate(repeated_valid_indices):
            gen_idx_within_seed = i % num_generations_per_seed
            output_idx = valid_idx * num_generations_per_seed + gen_idx_within_seed
            result[output_idx] = generated_sequences[i]
        
        return result
    
    @staticmethod
    def generate_unconditional_batch(
        pl_module,
        cls_token_id: int,
        num_samples: int,
        temperature: float = 0.95,
        prompt_token_ids: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """
        Batch generate molecules from CLS token (unconditional generation).
        
        Matches the original implementation in _molecule_validation_callback.py line 1968-2020.
        
        Args:
            pl_module: PyTorch Lightning module
            cls_token_id: CLS token ID to start generation
            num_samples: Number of molecules to generate
            temperature: Sampling temperature (default: 0.95)
            
        Returns:
            List of token ID lists (one per generated molecule)
        """
        # Build prompt tensor: <cls> [prompt...]
        prompt = [int(cls_token_id)]
        if prompt_token_ids:
            prompt.extend([int(x) for x in prompt_token_ids])

        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=pl_module.model.device).unsqueeze(0)
        # Shape: [num_samples, prompt_len]
        prompt_batch = prompt_tensor.repeat(int(num_samples), 1)

        # Generate all molecules in a single batch using pl_module.model.generate
        generated_sequences = pl_module.model.generate(
            input_ids=prompt_batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=400,  # Hard-coded in original (line 2010: max_new_tokens=400)
            pad_token_id=pl_module.tokenizer.pad_token_id,
            eos_token_id=pl_module.tokenizer.eos_token_id,
        )

        # Convert to list of lists - each inner list is one generated sequence
        if generated_sequences is None:
            return [[] for _ in range(num_samples)]
        
        # Shape: [num_samples, seq_len] -> List[List[int]]
        return [seq.tolist() for seq in generated_sequences]
    
    @staticmethod
    def extract_conditioning_from_batch(input_ids: torch.Tensor, 
                                       attention_mask: torch.Tensor,
                                       tokenizer) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Extract conditioning sequences from a batch.
        
        Extracts sequences up to the first SEP or EOS token for each batch item.
        
        Args:
            input_ids: Batch of input token IDs, shape [batch_size, seq_len]
            attention_mask: Batch of attention masks, shape [batch_size, seq_len]
            tokenizer: Tokenizer with special token IDs
            
        Returns:
            Tuple of (list of conditioning_ids, list of conditioning_masks)
        """
        from .smiles_utils import SmilesUtils
        
        sep_token_id = tokenizer.sep_token_id
        eos_token_id = tokenizer.eos_token_id
        
        batch_size = input_ids.shape[0]
        conditioning_ids_list = []
        conditioning_masks_list = []
        
        for i in range(batch_size):
            ids = input_ids[i]
            mask = attention_mask[i]
            
            # Extract conditioning sequence
            cond_ids, cond_mask = SmilesUtils.extract_conditioning_sequence(
                ids, mask, sep_token_id, eos_token_id
            )
            
            conditioning_ids_list.append(cond_ids)
            conditioning_masks_list.append(cond_mask)
        
        return conditioning_ids_list, conditioning_masks_list

