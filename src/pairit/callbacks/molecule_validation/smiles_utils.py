"""
SMILES Utilities
================

Utilities for SMILES validation, canonicalization, and token conversion.
"""

from typing import Optional, Tuple, Union, List

DATASET_START_TOKEN = "<ds>"
import torch
from rdkit import Chem
from atomic_datasets.utils.rdkit import is_molecule_sane

# Import molecule_metrics functions
from pairit.callbacks.molecule_metrics import canonicalize_smiles
from pairit.eval import is_valid_smiles


def shorten_str(s: str, n: int = 300) -> str:
    """
    Shorten a string for display.
    
    Args:
        s: String to shorten
        n: Maximum length
        
    Returns:
        Shortened string with '...' if truncated
    """
    if not isinstance(s, str):
        return s
    return ''.join(s.split(' '))[:n] + '...' if len(s) > n else s


class SmilesUtils:
    """Utility class for SMILES validation and conversion."""

    @staticmethod
    def _strip_dataset_identity_tokens(tokenizer, decoded: str) -> str:
        """
        Remove dataset-identity prompting tokens from a decoded string.

        We prefer using the tokenizer's registered additional_special_tokens so we
        don't need to know max_dataset_id here.
        """
        # Remove the fixed dataset-start token if present
        if DATASET_START_TOKEN in decoded:
            decoded = decoded.replace(DATASET_START_TOKEN, "")

        # Remove any dataset-id tokens that were registered as special tokens
        extra = getattr(tokenizer, "additional_special_tokens", None) or []
        for tok in extra:
            if tok.startswith("<ds_") and tok.endswith(">"):
                if tok in decoded:
                    decoded = decoded.replace(tok, "")

        return decoded
    
    @staticmethod
    def _strip_prompt_special_tokens(tokenizer, decoded: str) -> str:
        """
        Remove any registered prompting special tokens from a decoded string.

        We remove all additional_special_tokens except structural tokens (e.g. <sep>)
        which are used to split paired SMILES.
        """
        extra = getattr(tokenizer, "additional_special_tokens", None) or []
        sep_tok = getattr(tokenizer, "sep_token", "<sep>")
        for tok in extra:
            if tok == sep_tok:
                continue
            if tok and tok in decoded:
                decoded = decoded.replace(tok, "")
        return decoded
    
    @staticmethod
    def is_valid(smiles: str, use_sanity_check: bool = True) -> bool:
        """
        Check if a SMILES string is valid.
        
        Args:
            smiles: SMILES string to validate
            use_sanity_check: Whether to use atomic_datasets sanity check
            
        Returns:
            True if valid, False otherwise
        """
        if smiles is None:
            return False
        
        # Use molecule_metrics
        try:
            sanity_fn = is_molecule_sane if use_sanity_check else None
            return is_valid_smiles(smiles, sanity_fn=sanity_fn)
        except Exception:
            return False
        
        # Fallback to RDKit validation
        return SmilesUtils.is_valid_rdkit(smiles, use_sanity_check=use_sanity_check)
    
    @staticmethod
    def is_valid_rdkit(smiles: str, use_sanity_check: bool = True, 
                      verbose: bool = False) -> bool:
        """
        Check if a SMILES string is valid using RDKit directly.
        
        Args:
            smiles: SMILES string to validate
            use_sanity_check: Whether to use atomic_datasets sanity check
            verbose: Whether to print detailed error messages
            
        Returns:
            True if valid, False otherwise
        """
        if smiles is None:
            return False
        
        # Suppress RDKit output
        import os, sys
        stderr = sys.stderr
        devnull = open(os.devnull, 'w')
        sys.stderr = devnull
        
        try:
            # Parse SMILES with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                if verbose:
                    print(f"Failed to parse SMILES: {shorten_str(smiles)}")
                return False
            
            # Check sanity if requested
            if use_sanity_check:
                is_sane = is_molecule_sane(mol)
                if not is_sane and verbose:
                    print(f"Molecule failed sanity check: {shorten_str(smiles)}")
                return is_sane
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"Exception processing SMILES: {shorten_str(smiles, n=500)}")
                print(f"Error: {str(e)}")
            return False
        finally:
            # Restore stderr
            sys.stderr = stderr
            devnull.close()
    
    @staticmethod
    def canonicalize(smiles: str) -> Optional[str]:
        """
        Canonicalize a SMILES string.
        
        Args:
            smiles: SMILES string to canonicalize
            
        Returns:
            Canonical SMILES string, or None if invalid
        """
        if not SmilesUtils.is_valid(smiles):
            return None
        
        # Use molecule_metrics
        try:
            return canonicalize_smiles(smiles)
        except Exception:
            pass
        
        # Fallback to RDKit
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None
    
    @staticmethod
    def tokens_to_smiles(tokenizer, tokens: Union[List[int], torch.Tensor]) -> Tuple[str, Optional[str]]:
        """
        Convert token IDs to SMILES string(s).
        
        Handles special tokens (CLS, EOS, PAD, SEP) and splits on SEP token
        to extract multiple SMILES if present.
        
        Args:
            tokenizer: Tokenizer with decode method and special token attributes
            tokens: Token IDs (list or tensor)
            
        Returns:
            Tuple of (smiles1, smiles2) where smiles2 is None if no SEP token
        """
        # Convert tensor to list
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.reshape(-1).tolist()
        
        # Decode tokens
        decoded = tokenizer.decode(tokens)
        
        # Remove special tokens
        for token in [tokenizer.cls_token, tokenizer.eos_token, tokenizer.pad_token]:
            if token in decoded:
                decoded = decoded.replace(token, "")

        # Remove dataset identity prompt tokens (if present)
        decoded = SmilesUtils._strip_dataset_identity_tokens(tokenizer, decoded)
        # Remove any other prompt-related special tokens (e.g. property prompting)
        decoded = SmilesUtils._strip_prompt_special_tokens(tokenizer, decoded)
        
        # Split on SEP token if present
        if tokenizer.sep_token in decoded:
            parts = decoded.split(tokenizer.sep_token)
            smiles1 = ''.join(parts[0].split(' ')).strip()
            smiles2 = ''.join(parts[1].split(' ')).strip()
            return (smiles1, smiles2)
        
        # No SEP token, return single SMILES
        return (''.join(decoded.split(' ')).strip(), None)
    
    @staticmethod
    def extract_conditioning_sequence(input_ids: torch.Tensor, 
                                     attention_mask: torch.Tensor,
                                     sep_token_id: int, 
                                     eos_token_id: int,
                                     include_separator: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract conditioning sequence up to separator or end token.
        
        Args:
            input_ids: Input token IDs, shape [seq_len] or [batch_size, seq_len]
            attention_mask: Attention mask, same shape as input_ids
            sep_token_id: Separator token ID
            eos_token_id: End-of-sequence token ID
            include_separator: If True, include the SEP/EOS token in the output.
                             If False, exclude it (default for backward compatibility).
            
        Returns:
            Tuple of (conditioning_ids, conditioning_mask)
        """
        # Ensure 1D tensors
        if input_ids.dim() > 1:
            input_ids = input_ids.reshape(-1)
        if attention_mask.dim() > 1:
            attention_mask = attention_mask.reshape(-1)
        
        # Find position of SEP or EOS token
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
        
        # Determine cutoff position
        cutoff = len(input_ids)
        if len(sep_positions) > 0:
            cutoff = min(cutoff, sep_positions[0].item())
        if len(eos_positions) > 0:
            cutoff = min(cutoff, eos_positions[0].item())
        
        # Include separator if requested
        if include_separator and cutoff < len(input_ids):
            cutoff += 1
        
        # Extract conditioning sequence
        conditioning_ids = input_ids[:cutoff]
        conditioning_mask = attention_mask[:cutoff]
        
        return conditioning_ids, conditioning_mask
    
    @staticmethod
    def extract_conditioning_from_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                                       tokenizer) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """
        Extract conditioning sequence from input batch and decode to string.
        
        Convenience wrapper around extract_conditioning_sequence that also
        decodes the result to a SMILES string.
        
        Args:
            input_ids: Input token IDs, shape [seq_len] or [batch_size, seq_len]
            attention_mask: Attention mask, shape [seq_len] or [batch_size, seq_len]
            tokenizer: Tokenizer with sep_token_id, eos_token_id, and decode()
            
        Returns:
            Tuple of (unmasked_ids, decoded_string) or (None, None) if invalid
        """
        cond_ids, cond_mask = SmilesUtils.extract_conditioning_sequence(
            input_ids, attention_mask, tokenizer.sep_token_id, tokenizer.eos_token_id
        )
        if cond_ids is None or len(cond_ids) == 0:
            return None, None
        
        unmasked_ids = cond_ids[cond_mask.bool()]
        if len(unmasked_ids) == 0:
            return None, None
            
        cond_str = tokenizer.decode(unmasked_ids.tolist())
        return unmasked_ids, cond_str
    
    @staticmethod
    def get_cls_token_id(tokenizer) -> int:
        """
        Get the CLS token ID from tokenizer.
        
        Args:
            tokenizer: Tokenizer with cls_token or bos_token
            
        Returns:
            CLS token ID
        """
        if hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None:
            return tokenizer.cls_token_id
        elif hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            return tokenizer.bos_token_id
        else:
            raise ValueError("Tokenizer has neither cls_token_id nor bos_token_id")

