# Bundled assets

This directory holds the data files, oracles, baseline generations, and model
checkpoints that the code references via `pairit._assets.ASSETS` (or the
`${pairit_assets:...}` Hydra resolver). **Nothing here is committed to git**
(see `.gitignore`); the tree has to be materialised on local disk before the
code that consumes it will run. Set `PAIRIT_ASSETS_DIR` to keep it outside the
checkout.

The tree will be published on Hugging Face together with a download script.
Instructions will be added here when the hosting is finalised.

## Layout

```
assets/
├── data/              dataset molecule + pair tables
├── oracles/           RF joblibs + MiniMol heads
├── generations/       pre-selection 64-gen baseline pools (per method × dataset)
└── checkpoints/       model weights
```

## Contents

| Local path | Size | Notes |
|---|---|---|
| `oracles/moleculenet/evaluators/{bace,bbbp,hiv}_b.joblib` | 10.5 MB | Train-only RF oracles (test AUC 0.906/0.913/0.800). See §Oracles. |
| `oracles/moleculenet/minimol/{BACE,BBBP,HIV}/head.joblib` | ~10 KB | MiniMol classifier heads (second-oracle re-scoring). Heads only; training-time artefacts are not shipped. |
| `oracles/tdc/evaluators/herg_b.joblib` | 1.7 MB | TDC hERG RF oracle (test ROC-AUC 0.826, blocker classification). `caco2_wang` and `freesolv` oracles are not shipped — the paper's TDC scope is hERG. `LazyPropertyEvaluator` in `property_registry.py` tolerates the missing files (they only raise `FileNotFoundError` if `compute_caco2_papp`/`compute_freesolv_dg` is actually called). |
| `oracles/tdc/minimol/hERG/head.joblib` | ~3 KB | TDC hERG MiniMol head (test ROC-AUC 0.821). |
| `data/zinc/{train,valid}/molecules.parquet` | 31 MB total (28 MB train, 3.3 MB valid) | ZINC-250K pretraining molecule table with all-property columns (logp/qed/sa/etc + MoleculeNet labels), split per subdirectory to match the moleculenet/tdc layout. No `test/molecules.parquet` — ZINC's downstream test is the 5k ICML subset (next rows). |
| `data/zinc/{train,valid,test}/pairs.parquet` | 118 MB total (109 MB train, 5.4 MB valid, 783 KB test) | ZINC molecular pair tables. **`valid/pairs.parquet` excludes the 5k ICML test molecules** to prevent leakage during early stopping. |
| `data/zinc/test/test_subset_5k.csv` | 605 KB | The 5000-molecule ZINC test subset from the ICML paper. Used by `DEFAULT_ZINC250K_TEST_CSV` for `sim_gt_mean` when zinc250k is the target dataset in an eval. |
| `data/zinc/distance_distributions.parquet` | 8.9 MB | Precomputed per-molecule distance stats used by distance-prompting configs. `MoleculeImprovementLightningDataModule` loads it via `distribution_file`; recomputes lazily if missing (~minutes on 250k mols). |
| `data/zinc/pair_build_config.yaml` | 599 B | Provenance record — how the ZINC pair dataset was built (filter type, max pairs per mol, splits). Not code-referenced. |
| `generations/pairit/moleculenet/{bace,bbbp,hiv}/all_generations.parquet` | 954 KB total | PairIT single-pass best-of-64 raw candidate pools. Rows = seeds×64: BACE 8640 (135 seeds), BBBP 5312 (83 seeds — 5 unencodable dropped), HIV 13696 (214 seeds — 17 dropped). |
| `generations/pairit-beam/moleculenet/{bace,bbbp,hiv}/all_generations.parquet` | 788 KB total | PairIT iterative-beam-search raw pools. Variable length (beam terminates when candidates stop improving) — rows < seeds×64. BACE 4929 (132 seeds), BBBP 4871 (83), HIV 9674 (212). |
| `generations/pairit-beam/tdc/herg/all_generations.parquet` | 341 KB | hERG PairIT-Beam (k=10), single training seed (seed0). 80/85 seeds (5 hERG test molecules unencodable). |
| `generations/pairit-rl/moleculenet/{bace,bbbp,hiv}/generations.csv` | 3.3 MB total | PairIT after GRPO post-training with the train-only RF oracle as reward. Full seed coverage: BACE 135/135, BBBP 88/88, HIV 231/231 — RL's tokenization path doesn't drop the disconnected/unk seeds that base PairIT does. BACE uses the sim1 reward-weight variant; BBBP/HIV use sim5. |
| `generations/mmp/moleculenet/{bace,bbbp,hiv}/generations.csv` | ~10 MB total | MMP-prop (property-improvement-filtered mmpdb enumeration) baseline. |
| `generations/nfbo/moleculenet/{bace,bbbp,hiv}/generations.csv` | ~2 MB total | Normalizing-flow BO. **BACE was sampled with the leaky RF oracle** — the train-only RF drove the BO's GP surrogate to numerical failure on all 135 BACE class-0 seeds (uniform p_class1=0 → degenerate GP → NaN cholesky). BBBP and HIV were sampled with the train-only oracle. Scoring uses the train-only oracle for all three; the asymmetry affects sampling only. |
| `generations/nfbo/tdc/herg/generations.csv` | ~500 KB | NFBO on hERG (85 blocker seeds). |
| `generations/ddsbm-bridge/moleculenet/{bace,bbbp,hiv}/all_generations.parquet` + `generations/ddsbm-bridge/tdc/herg/all_generations.parquet` | ~5 MB total | Diffusion Schrödinger Bridge, forward path, merged from 64 per-generation files into one parquet per dataset with `seed_idx, gen_idx, smiles_seed, smiles_generated, is_valid, smiles_generated_relaxed, is_valid_relaxed` columns matching PairIT format. hERG restricted to the 85 blocker seeds (Y=1); the 26 non-blocker test molecules don't fit the improvement task. |
| `generations/ddsbm-sb/moleculenet/{bace,bbbp,hiv}/all_generations.parquet` + `generations/ddsbm-sb/tdc/herg/all_generations.parquet` | ~5 MB total | Diffusion Schrödinger Bridge, SB variant. Same merge; hERG also restricted to 85 blocker seeds. |
| `generations/hierg2g/moleculenet/{bace,bbbp,hiv}/translations.txt` + `generations/hierg2g/tdc/herg/generations.csv` | ~50 MB total | Hierarchical graph-to-graph translation, logp06 variant, epoch 11. Text format for MoleculeNet (`seed_smi gen_smi` per line, 64 lines per seed), CSV for hERG. |
| `checkpoints/pairit/zinc_pretrain/CLM_68M_logp_eps1.ckpt` | ~800 MB | PairIT ZINC-pairs pretrain, CLM_68M, logP, eps=1, best (lowest val_loss) ckpt. |
| `checkpoints/pairit/moleculenet/{bace,bbbp,hiv}/finetuned.ckpt` | ~2.4 GB total | PairIT MoleculeNet SFT fine-tunes on top of the ZINC pretrain (best-by-val-loss ckpt per dataset). |
| `checkpoints/pairit/moleculenet/{bace,bbbp,hiv}/rl.ckpt` | ~2.4 GB total | PairIT after GRPO with the train-only RF oracle as reward. BACE uses the `sim1` reward variant; BBBP/HIV use `sim5`. Final RL ckpt (GRPO produces no per-epoch val_loss to select on). |
| `checkpoints/pairit/tdc/herg/finetuned.ckpt` | ~800 MB | PairIT hERG SFT fine-tune (best-by-val-loss). eps=0.5, KNN-pairs. |
| `checkpoints/nfbo/zinc/seqflow_zinc_pairs_best.pt` + `checkpoints/nfbo/moleculenet/{bace,bbbp,hiv}/seqflow_{ds}_best.pt` | ~500 MB total | NFBO checkpoints — 1 ZINC-pair fine-tune + 3 MoleculeNet real-only fine-tunes. The initial NFBO pretrain (`seqflow_1.27M_3.pt`) comes from the upstream NFBO repository and is not shipped. |
| `checkpoints/ddsbm-bridge/moleculenet/{bace,bbbp,hiv}/{forward,backward}/last.ckpt` + `checkpoints/ddsbm-bridge/zinc/last.ckpt` | ~500 MB total | DDSBM (Bridge variant). Only one iteration exists, so `last.ckpt` is the definitive ckpt. |
| `checkpoints/ddsbm-sb/moleculenet/{bace,bbbp,hiv}/{forward,backward}/last.ckpt` + `checkpoints/ddsbm-sb/zinc/backward_last.ckpt` | ~500 MB total | DDSBM (SB variant). SB training does 5 iterations; the paper uses iterate 4 (the last). |

### Not included

- **DDSBM-Bridge and DDSBM-SB checkpoints for hERG.**
- **NFBO checkpoints for hERG.**
- **HierG2G checkpoints** — third-party model; regenerate from the upstream repository.
- **Pretraining-ablation checkpoints** — not needed for the main paper tables.
- **Non-beam PairIT and PairIT-RL raw candidate pools for hERG** — only PairIT-Beam (k=10) is shipped for hERG.

## Oracles

Only the train-only RF oracles are shipped. A *leaky* variant (RF trained on
the full split, test AUC = 1.000) exists upstream and is deliberately excluded,
so `assets/oracles/moleculenet/evaluators/` always means the train-only oracles
in this repo.

The NF-BO **bace** baseline generations were sampled with the leaky RF (see the
table above). Those generations are re-scored with the train-only oracle
downstream, so the leaky joblib itself is not needed — only the pre-computed
generation CSV is.
