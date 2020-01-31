# parameter-factorization
Factorization of the neural parameter space for zero-shot multi-lingual and multi-task transfer. Code for the paper:

Edoardo M. Ponti, Ivan Vulić, Ryan Cotterell, Marinela Parovic, Roi Reichart and Anna Korhonen. 2020. **Parameter Space Factorization for Zero-Shot Learning across Tasks and Languages**. [[arXiv]](https://arxiv.org/pdf/2001.11453.pdf)

If you use this software for academic research, please cite the paper in question:
```
@misc{ponti2020parameter,
    title={Parameter Space Factorization for Zero-Shot Learning across Tasks and Languages},
    author={Edoardo M. Ponti and Ivan Vulić and Ryan Cotterell and Marinela Parovic and Roi Reichart and Anna Korhonen},
    year={2020},
    eprint={2001.11453},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Dependencies

- python 3.5.2
- pytorch 1.1.0

## Data

Obtain the data from Universal Dependencies (for POS tagging) and Wikiann (for NER):

```
 ./tools/get_data.sh
```

## Train and evaluate

Run the model and baselines. For instance, to train and evaluate parameter space factorization with low-rank factor covariance:

```
python src/run_matrix_completion.py --mode lrcmeta --rank_cov 10 
```

## Acknowledgements

The part of the code for multilingual BERT has been taken from in [HuggingFace's Transformers](https://github.com/huggingface/transformers). The link contains a copy of the original license and the citation for the library.