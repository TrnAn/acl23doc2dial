from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import \
    DocumentGroundedDialogRerankTrainer
from modelscope.utils.constant import DownloadMode
# from modelscope.hub.snapshot_download import snapshot_download
import argparse

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--use-extended-dataset", help= "Run experiments on English and Chinese dataset", type= bool, default= False)
    parser.add_argument("--test-size", help= "Set test split", type= float, default= 0.1)
    parser.add_argument("--lang-token", help= "Add language token <lang> to input", action=argparse.BooleanOptionalAction)
    # parser.add_argument("--use-batch-accumulation", help= "Use batch accumulation to maintain baseline results", type= bool, default= False)
    parser.add_argument("--cache-dir", help= "Specifiy cache dir to save model to", type= str, default= "./")
    args = vars(parser.parse_args())

    args.update({
        'device': 'gpu',
        'tokenizer_name': '',
        'cache_dir': args["cache_dir"],
        'instances_size': 1,
        'output_dir': f'{args["cache_dir"]}/output',
        'max_num_seq_pairs_per_device': 32,
        'full_train_batch_size': 32,
        'gradient_accumulation_steps': 32,
        'per_gpu_train_batch_size': 1,
        'num_train_epochs': 10,
        'train_instances': -1,
        'learning_rate': 2e-5,
        'max_seq_length': 512,
        'num_labels': 2,
        'fold': '',  # IofN
        'doc_match_weight': 0.0,
        'query_length': 195,
        'resume_from': '',  # to resume training from a checkpoint
        'config_name': '',
        'do_lower_case': True,
        'weight_decay': 0.0,  # previous default was 0.01
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_instances': 0,  # previous default was 0.1 of total
        'warmup_fraction': 0.0,  # only applies if warmup_instances <= 0
        'no_cuda': False,
        'n_gpu': 1,
        'seed': 42,
        'fp16': False,
        'fp16_opt_level': 'O1',  # previous default was O2
        'per_gpu_eval_batch_size': 8,
        'log_on_all_nodes': False,
        'world_size': 1,
        'global_rank': 0,
        'local_rank': -1,
        'tokenizer_resize': True,
        'model_resize': True
    })

    args[
        'gradient_accumulation_steps'] = args['full_train_batch_size'] // (
            args['per_gpu_train_batch_size'] * args['world_size'])
    train_dataset = MsDataset.load(
        'DAMO_ConvAI/FrDoc2BotRerank',
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
        split='train')
    
    # cache_path = snapshot_download('DAMO_ConvAI/nlp_convai_ranking_pretrain', cache_dir=args["cache_dir"])
    trainer = DocumentGroundedDialogRerankTrainer(
        model=f'DAMO_ConvAI/nlp_convai_ranking_pretrain', dataset=train_dataset, args=args)
    trainer.train()


if __name__ == '__main__':
    main()
