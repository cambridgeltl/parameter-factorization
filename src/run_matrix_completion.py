from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import itertools
import gc

import torch
from transformers import BertConfig, BertTokenizer, MultiTaskBert

from commons import set_seed, load_and_cache_examples, find_nearest_languages
from utils_conll import CLASSES_PER_TASK
from commons import train, evaluate

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--mode", default='transfer', choices=['transfer', 'meta', 'lrcmeta', 'svimeta'],
                        help="Modes.")
    parser.add_argument("--meta_emb_dim", default=100, type=int,
                        help="Dimensionality of task and language embeddings.")
    parser.add_argument("--n_samples", default=3, type=int,
                        help="Number of samples in the Bayesian mode.")
    parser.add_argument("--scaling", default='uniform', type=str, choices=['uniform', 'linear_annealing', 'logistic_annealing'],
                        help="Scaling for KL term in VI.")
    parser.add_argument("--max_patience", default=10, type=int,
                        help="Maximum patience for early stopping.")
    parser.add_argument("--weight_by_size", action='store_true',
                        help="Sample task-language example according to data size, weight lr accordingly")
    parser.add_argument("--num_hidden_layers", default=6, type=int,
                        help="Number of hidden layers for the functions psi and phi")
    parser.add_argument("--rank_cov", default=0, type=int,
                        help="Rank of the factored covariance matrix. Diagonal if < 1")
    parser.add_argument("--typ_dist", default="src/typ_feats.tab", type=str,
                        help="File containing pre-computed typological distances between languages")
    parser.add_argument("--largest_source", action='store_true',
                        help="Always choose the source language with the largest number of examples for transfer")
    parser.add_argument("--model_averaging", action='store_true',
                        help="Predict through model averaging rather than pluggin in the mean")

    # Experiment
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--no_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--no_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--debug", action='store_true',
                        help="Whether to debug gradient flow.")

    parser.add_argument("--max_seq_length", default=250, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0., type=float,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=2500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=2500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()

    if args.largest_source:
        assert args.mode == "transfer"

    args.tasks = ['pos', 'ner']
    args.languages = sorted(['aii', 'am', 'ar', 'bm', 'cy', 'et', 'eu', 'fi', 'fo', 'gl', 'gun', 'he', 'hsb', 'hu', 'hy', 'id', 'kk',
                             'kmr', 'ko', 'kpv', 'mt', 'myv', 'sme', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'vi', 'wo', 'yo', 'yue'])
    # NER changes zh-yue -> yue, gn -> gun, sme -> sm, arc -> aii, ku -> kmr, kv -> kpv,
    # find_overlapping_languages(args.tasks)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.no_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)

    # Creating partitions of observed and held-out data for each iteration
    cartesian_product = list(itertools.product(sorted(args.tasks), sorted(args.languages)))
    partitions = {ti: sorted([cartesian_product[pi] for pi in range(ti, len(cartesian_product), len(args.tasks))]) for ti in range(len(args.tasks))}

    for partition, heldout_pairs in partitions.items():
        logger.info("Partition: {}".format(partition))
        logger.info("Held-out task-language pairs: {}".format(heldout_pairs))
        observed_pairs = sorted(list(set(cartesian_product) - set(heldout_pairs)))
        lang_nns = find_nearest_languages(heldout_pairs, observed_pairs, args.typ_dist)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        config_class, tokenizer_class, model_class = BertConfig, BertTokenizer, MultiTaskBert

        # Data and model
        config = config_class.from_pretrained(args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        dataloaders, num_batches = load_and_cache_examples(args, tokenizer, observed_pairs if not args.largest_source
                                                           else [('pos', 'en'), ('ner', 'en_{}'.format(partition))], partition)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
                                            mode=args.mode, languages=args.languages, n_classes={t: len(c) for t, c in CLASSES_PER_TASK.items()},
                                            emb_dim=args.meta_emb_dim, n_samples=args.n_samples, num_hidden_layers=args.num_hidden_layers,
                                            rank_cov=args.rank_cov, largest_source=args.largest_source)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
        model.to(args.device)

        logger.info("Training/evaluation parameters %s", args)

        # Training
        if not args.no_train:
            global_step, tr_loss = train(args, dataloaders, model, tokenizer, num_batches, observed_pairs, heldout_pairs, partition, lang_nns)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        model = model_class.from_pretrained(os.path.join(args.output_dir, 'checkpoint-best-{}'.format(partition)),
                                            from_tf=bool('.ckpt' in args.model_name_or_path), config=config,
                                            mode=args.mode, languages=args.languages, n_classes={t: len(c) for t, c in CLASSES_PER_TASK.items()},
                                            emb_dim=args.meta_emb_dim, n_samples=args.n_samples, num_hidden_layers=args.num_hidden_layers,
                                            rank_cov=args.rank_cov, largest_source=args.largest_source)

        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
        if not args.no_eval and args.local_rank in [-1, 0]:
            # Evaluate
            with torch.no_grad():
                result = evaluate(args, dataloaders, model, tokenizer, 'test', heldout_pairs, lang_nns, partition,
                                  prefix=partition, sample=args.model_averaging)
            logger.info("Results: {}".format(result))

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
