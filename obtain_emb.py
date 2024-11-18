import os
import json
import argparse
import numpy as np
import torch
from transformers import BertTokenizer

from data_retriever import get_embeddings, make_single_loader, EntitySet
from run_retriever import load_model
from tqdm import tqdm


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='/nfs/yding4/EntQA/data/retriever.pt',
                        help='model path')
    parser.add_argument('--pretrained_path', type=str, default='/nfs/yding4/EntQA/data/blink_model/models/',
                        help='the directory of the wikipedia pretrained models')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training from checkpoint?')
    parser.add_argument('--type_loss', type=str, default='sum_log_nce',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of multi-label loss ?')
    parser.add_argument('--use_title', action='store_true',
                        help='use title or topic?')
    parser.add_argument('--add_topic', action='store_true',
                        help='add topic information?')
    parser.add_argument('--blink', action='store_true',
                        help='use BLINK pretrained model?')
    parser.add_argument('--max_len', type=int, default=42,
                        help='max length of the mention input ')
    parser.add_argument('--data_dir', type=str, default='/nfs/yding4/EntQA/data/retriever_input/',
                        help='the  data directory')
    parser.add_argument('--kb_dir', type=str, default='/nfs/yding4/EntQA/data/kb',
                        help='the knowledge base directory')
    parser.add_argument('--out_dir', type=str, default='/nfs/yding4/EntQA/data/reader_input',
                        help='the output saving directory')
    parser.add_argument('--B', type=int, default=4,
                        help='the batch size per gpu')
    parser.add_argument('--lr', type=float, default=2e-6,
                        help='the learning rate')
    parser.add_argument('--epochs', type=int, default=4,
                        help='the number of training epochs')
    parser.add_argument('--k', type=int, default=100,
                        help='recall@k when evaluate')
    parser.add_argument('--warmup_proportion', type=float, default=0.2,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--rands_ratio', default=0.9, type=float,
                        help='the ratio of random candidates and hard')
    parser.add_argument('--num_cands', default=64, type=int,
                        help='the total number of candidates')
    parser.add_argument('--mention_bsz', type=int, default=4096,
                        help='the batch size')
    parser.add_argument('--entity_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--use_gpu_index', action='store_true',
                        help='use gpu index?')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) "
             "instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', "
             "'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument('--cands_embeds_path', type=str,default='/nfs/yding4/EntQA/data/candidate_embeds.npy',
                        help='the directory of candidates embeddings')
    parser.add_argument('--use_cached_embeds', action='store_true',
                        help='use cached candidates embeddings ?')
    args = parser.parse_args()

    # https://github.com/WenzhengZhang/EntQA/tree/main?tab=readme-ov-file#train-retriever
    args.blink = True
    args.add_topic = True
    return args


def load_entities(kb_dir='/nfs/yding4/EntQA/data/kb'):
        entities = []
        with open(os.path.join(kb_dir, 'entities_kilt.json')) as f:
            for line in tqdm(f):
                entities.append(json.loads(line))

        return entities


# device = 'cpu'
device = 'cuda:0'
device = torch.device(device)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

args = parse_arg()
# BLINK config

config = {
    "top_k": 100,
    "biencoder_model": args.pretrained_path + "biencoder_wiki_large.bin",
    "biencoder_config": args.pretrained_path + "biencoder_wiki_large.json"
}


# https://github.com/WenzhengZhang/EntQA/blob/main/run_retriever.py#L329
model = load_model(
    False, 
    config['biencoder_config'],
    args.model, 
    device, 
    args.type_loss,
    args.blink,
)

model.eval()


print('start to load entities!')
entities = load_entities(kb_dir=args.kb_dir)
print('load entities done!')
entity_set = EntitySet(entities)
entity_loader = make_single_loader(entity_set, args.entity_bsz, False)
all_cands_embeds = get_embeddings(entity_loader, model, False, device)
np.save(args.cands_embeds_path, all_cands_embeds)



