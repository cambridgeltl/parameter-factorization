import os
import random
import numpy as np
import logging
from collections import defaultdict
import pandas

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from transformers import AdamW, WarmupLinearSchedule
from utils_conll import (read_conll_examples, read_xnli_examples, convert_examples_to_features, compute_metrics,
                         glue_convert_examples_to_features, CLASSES_PER_TASK)
from plots import plot_grad_flow

logger = logging.getLogger(__name__)

TASK_DIRS = {'ner': 'data/wikiann', 'pos': 'data/ud-treebanks-v2.4-symlinked', 'nli': 'data/XNLI'}


def find_nearest_languages(heldout_pairs, observed_pairs, typ_dist):
    pair2nn = {}
    df = pandas.read_csv(typ_dist, sep='\t')
    df = df.set_index('LANGUAGE')
    for task, heldout_language in heldout_pairs:
        observed_languages = [l for t, l in observed_pairs if t == task]
        row = df.loc[heldout_language, observed_languages]
        pair2nn[(task, heldout_language)] = row.idxmin()
    return pair2nn


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def checkpoint(model, args, affix='', results=None):
    output_dir = os.path.join(args.output_dir, 'checkpoint{}'.format(affix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    if results is not None:
        torch.save(results, os.path.join(output_dir, 'results{}.bin'.format(affix)))
    logger.info("Saving model checkpoint to %s", output_dir)


def train(args, dataloaders, model, tokenizer, num_batches, observed_pairs, heldout_pairs, partition, lang_nns):
    """ Train the model """
    _, ex_counts = zip(*[x for x in sorted(num_batches.items())])
    if args.weight_by_size:
        sample_probs = ex_counts / np.linalg.norm(ex_counts, ord=1)
        ex_avg = sum(ex_counts) / len(ex_counts)
        lr_weights = {key: ex_avg / value for key, value in sorted(num_batches.items())}
    else:
        sample_probs = None

    num_batches = sum(ex_counts)

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (num_batches // args.gradient_accumulation_steps) + 1
    else:
        t_total = num_batches // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight',
                'mean', 'logvar']  # No decay in Gaussian weights
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.warmup_proportion * t_total), t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_batches * args.train_batch_size)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_kl, logging_kl = 0.0, 0.0
    best_valid = 0.0
    patience = 0
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch in range(int(args.num_train_epochs)):
        step = 0
        while step < num_batches:
            if not args.largest_source:
                which = np.random.choice(len(observed_pairs), p=sample_probs)
                task, language = observed_pairs[which]
            else:
                task = random.choice(args.tasks)
                language = 'en_{}'.format(partition) if task == "ner" else 'en'
            bank = random.choice([k for k in dataloaders[task].keys() if k.startswith(language)])
            batch = next(dataloaders[task][bank]['train'])
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            outputs = model(batch, task, language if not args.largest_source else 'en')
            loss, _, kl_term = outputs

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                kl_term = kl_term.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                kl_term = kl_term / args.gradient_accumulation_steps

            tr_loss += loss.item()

            if args.mode in ['lrcmeta', 'svimeta']:
                if args.scaling == "uniform":
                    scaling = 1. / t_total
                elif args.scaling == "linear_annealing":
                    scaling = ((t_total - step - 1) * 2. + 1.) / t_total**2
                elif args.scaling == "logistic_annealing":
                    steepness = 0.0025
                    scaling = 1. / (1 + np.exp(-steepness * (step - t_total / 2.)))
                loss = loss + scaling * kl_term
                tr_kl += kl_term.item()

            if args.local_rank in [-1, 0] and global_step % 1000 == 0 and global_step:
                logger.info("Epoch {} seen examples {} log-lik {}".format(epoch, step * args.train_batch_size, (tr_loss - logging_loss) / 1000.))
                tb_writer.add_scalar('lr_{}'.format(partition), scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('log-lik_{}'.format(partition), (tr_loss - logging_loss) / 1000., global_step)
                if args.mode in ['lrcmeta', 'svimeta']:
                    logger.info("Epoch {} seen examples {} kl {}".format(epoch, step * args.train_batch_size, (tr_kl - logging_kl) / 1000.))
                    tb_writer.add_scalar('kl_{}'.format(partition), (tr_kl - logging_kl) / 1000., global_step)
                logging_loss = tr_loss
                logging_kl = tr_kl

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.debug:
                plot_grad_flow(model.named_parameters())

            if args.weight_by_size:
                lr2s = []
                for param_group in optimizer.param_groups:
                    lr2 = param_group['lr']
                    param_group['lr'] = lr2 * lr_weights[(task, language)]
                    lr2s.append(lr2)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, dataloaders, model, tokenizer, 'dev', heldout_pairs, lang_nns, partition, prefix=partition)
                        for task, value1 in results.items():
                            for language, value2 in value1.items():
                                for metric, value3 in value2.items():
                                    tb_writer.add_scalar('eval_{}'.format("-".join([task, language, metric])), value3, global_step)
                        overall_valid = np.mean([results[task]['all']['f1'] for task in args.tasks])
                        if overall_valid > best_valid:
                            logger.info("New best validation! Average f1 {}".format(overall_valid))
                            checkpoint(model, args, affix='-best-{}'.format(partition), results=results)
                            best_valid = overall_valid
                            patience = 0
                        else:
                            patience += 1

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    checkpoint(model, args, affix='-latest-{}'.format(partition), results=None)

                if args.weight_by_size:
                    for param_group, lr2 in zip(optimizer.param_groups, lr2s):
                        param_group['lr'] = lr2

            if (args.max_steps > 0 and global_step > args.max_steps) or patience > args.max_patience:
                break
            step += 1
        if (args.max_steps > 0 and global_step > args.max_steps) or patience > args.max_patience:
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, dataloaders, model, tokenizer, split, heldout_pairs, lang_nns, partition, prefix='', sample=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    results = {}
    for eval_task in args.tasks:
        heldout_languages = [l for t, l in heldout_pairs if t == eval_task]
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Eval!
        logger.info("***** Running {} evaluation {} *****".format(split, prefix))
        logger.info("  Batch size = %d", args.eval_batch_size)
        results[eval_task] = {}
        for bank, dataset in dataloaders[eval_task].items():
            language = bank.split('-')[0] if eval_task == 'pos' else bank
            if language not in heldout_languages or split not in dataset:
                continue
            nearest_language = 'en' if args.largest_source else lang_nns[(eval_task, language)]
            eval_loss = 0.0
            nb_eval_steps = 0
            preds = None
            out_label_ids = None
            results[eval_task][bank] = {}

            for batch in dataset[split]:
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    tmp_eval_loss, logits, _ = model(batch, eval_task, language if args.mode != 'transfer' else nearest_language,
                                                     sample=sample, calculate_log_probs=False)
                    eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                raw_labels = batch[1][:, :batch[2].max().item()]
                raw_labels = raw_labels[(raw_labels != -1).to(torch.bool)] if eval_task != 'nli' else raw_labels
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    out_label_ids = raw_labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, raw_labels.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(eval_task, preds, out_label_ids.squeeze())
            results[eval_task][bank].update(result)

    output_eval_file = os.path.join(args.output_dir, "eval_results_{}.txt".format(prefix))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for task, value1 in sorted(results.items()):
            tot_acc = 0.
            tot_f1 = 0.
            tot_n = 0.
            for bank, value2 in sorted(value1.items()):
                tot_n += 1.
                for metric, value3 in sorted(value2.items()):
                    if metric == "acc":
                        tot_acc += value3
                    elif metric == "f1":
                        tot_f1 += value3
                    logger.info("%s %s %s = %s", task, bank, metric, str(value3))
                    writer.write("%s %s %s = %s\n" % (task, bank, metric, str(value3)))
            logger.info("%s all acc = %s", task, str(tot_acc / tot_n))
            logger.info("%s all f1 = %s", task, str(tot_f1 / tot_n))
            writer.write("%s all acc = %s\n" % (task, str(tot_acc / tot_n)))
            writer.write("%s all f1 = %s\n" % (task, str(tot_f1 / tot_n)))
            results[task]['all'] = {'acc': tot_acc / tot_n}
            results[task]['all'] = {'f1': tot_f1 / tot_n}

    return results


def find_overlapping_languages(tasks):
    task2lang = defaultdict(set)
    for task in tasks:
        basedir = TASK_DIRS[task]
        for subdir in os.listdir(basedir):
            if os.path.isdir(os.path.join(basedir, subdir)):
                if subdir not in ['qhe-hiencs', 'fr-ftb', 'en-esl', 'ar-nyuad', 'ja-bccwj']:  # Treebanks without tokens
                    task2lang[task].add(subdir.split("-")[0] if task == 'pos' else subdir)
    languages = set.intersection(*task2lang.values())
    logger.info("Languages found in data directory:")
    for k, v in task2lang.items():
        logger.info(k + ": " + ", ".join(list(v)))
    logger.info("Overlapping languages:")
    logger.info(", ".join(languages))
    return languages


def infinite_generator(dataloader):
    while True:
        for i, batch in enumerate(dataloader):
            yield(batch)


def load_and_cache_examples(args, tokenizer, observed_pairs, partition):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    features = {task: {} for task in args.tasks}
    label_maps = {k: {vv: i for i, vv in enumerate(sorted(v))} for k, v in CLASSES_PER_TASK.items()}
    for task in args.tasks:
        # Load data features from cache or dataset file
        cached_features_dir = 'data/cached/{}'.format(task)
        if os.path.exists(cached_features_dir) and not args.overwrite_cache:
            logger.info("Loading features from cached direcory %s", cached_features_dir)
            for subdir in os.listdir(cached_features_dir):
                filepath = os.path.join(cached_features_dir, subdir)
                if os.path.isfile(filepath):
                    subdir = subdir.split(".")[0]
                    language = subdir.split("-")[0] if task == "pos" else subdir
                    to_load = args.languages + (['en_{}'.format(partition)] if task == "ner" else ['en'])
                    if language in (to_load if args.largest_source else args.languages):
                        features[task][subdir] = torch.load(filepath)
        else:
            if not os.path.exists(cached_features_dir):
                os.makedirs(cached_features_dir)
            basedir = TASK_DIRS[task]
            label_map = label_maps[task]
            for subdir in os.listdir(basedir):
                if os.path.isdir(os.path.join(basedir, subdir)):
                    if task == 'pos':
                        input_file_pattern = subdir + '-ud-{}.conllu'
                        # Treebanks without tokens or splits
                        if subdir in ['qhe-hiencs', 'fr-ftb', 'en-esl', 'ar-nyuad', 'ja-bccwj']:
                                # or not all([os.path.exists(os.path.join(basedir, subdir, input_file_pattern.format(split))) for split in ['train', 'dev', 'test']]):
                            continue
                    elif task == 'ner':
                        if subdir in ['ja', 'zh']:  # files too large
                            continue
                        input_file_pattern = subdir + '.{}.bio'
                    elif task == 'nli':
                        input_file_pattern = 'xnli.{}.' + subdir + '.tsv'

                    splits = {}
                    for split in ['test', 'dev', 'train']:
                        input_file = os.path.join(basedir, subdir, input_file_pattern.format(split))
                        if not os.path.exists(input_file):
                            continue
                        logger.info("Creating features from dataset file {}".format(input_file))
                        if task in ['ner', 'pos']:
                            examples = read_conll_examples(input_file=input_file, task=task)
                            cur_features = convert_examples_to_features(examples=examples,
                                                                        label_map=label_map,
                                                                        tokenizer=tokenizer,
                                                                        max_seq_length=args.max_seq_length)

                        else:
                            examples = read_xnli_examples(input_file=input_file)
                            cur_features = glue_convert_examples_to_features(examples=examples,
                                                                             label_map=label_map,
                                                                             tokenizer=tokenizer,
                                                                             max_seq_length=args.max_seq_length)
                        ids_dict = {'input_ids': np.array([f.input_ids for f in cur_features], dtype=np.int32),
                                    'input_lengths': np.array([f.input_lengths for f in cur_features], dtype=np.int16),
                                    'label_ids': np.array([f.label_ids for f in cur_features], dtype=np.int8),
                                    }

                        splits[split] = ids_dict
                        # If a treebank has no train set, we use the test set for cross-lingual transfer
                        if split == 'test' and not os.path.exists(os.path.join(basedir, subdir, input_file_pattern.format('train'))):
                            logger.warning("{} has no train set. Test set employed in its stead.".format(subdir))
                            splits['train'] = ids_dict
                    if args.local_rank in [-1, 0]:
                        lang_dir = os.path.join(cached_features_dir, subdir + '.pth')
                        logger.info("Saving features for language %s into cached directory %s", subdir, cached_features_dir)
                        torch.save(splits, lang_dir)

                    to_load = args.languages + (['en_{}'.format(partition)] if task == "ner" else ['en'])
                    if subdir.split("-")[0] in (to_load if args.largest_source else args.languages):
                        features[task][subdir] = splits

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    dataloaders = {task: {} for task in args.tasks}
    num_batches = defaultdict(int)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    for task in args.tasks:
        for language in features[task]:
            dataloaders[task][language] = {}
            for split in ['train', 'dev', 'test']:
                if split in features[task][language]:
                    feature_dict = features[task][language][split]
                    dataset = TensorDataset(torch.from_numpy(np.int32(feature_dict['input_ids'])[:, :args.max_seq_length]).long(),
                                            torch.from_numpy(np.int8(feature_dict['label_ids'])[:, :(args.max_seq_length if task != 'nli' else None)]).long(),
                                            torch.min(torch.LongTensor([args.max_seq_length]), torch.from_numpy(np.int16(feature_dict['input_lengths'])).long())
                                            )
                    if split == 'train':
                        sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
                    else:
                        sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
                    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size if split == 'train' else args.eval_batch_size)
                    if split == 'train':
                        langname = language.split("-")[0] if task == "pos" else language
                        if (task, langname) in observed_pairs:
                            num_batches[(task, langname)] += len(dataloader)
                        dataloader = infinite_generator(dataloader)
                    dataloaders[task][language][split] = dataloader
    return dataloaders, num_batches
