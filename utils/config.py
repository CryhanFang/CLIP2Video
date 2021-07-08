#coding:utf-8
# @Time : 2021/4/25 9:56 下午
# @Author : Han Fang
# @File: config.py 
# @Version: 2021/4/25 9:56 下午 config.py
import argparse

def get_args(description='CLIP2Video on Dideo-Text Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)


    # arugment based on CLIP4clip:
    # https://github.com/ArrowLuo/CLIP4Clip/blob/668334707c493a4eaee7b4a03b2dae04915ce170/main_task_retrieval.py#L457
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='frame rate for uniformly sampling the video')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")


    # important extra argument for training and testing CLIP2Video
    parser.add_argument('--sim_type', type=str, default="meanP", choices=["meanP", "seqTransf"],
                        help="choice a similarity header.")

    # argument for testing
    parser.add_argument('--checkpoint', type=str, default='', help="checkpoint dir")
    parser.add_argument('--model_num', type=str, default='', help="model id")
    parser.add_argument('--local_rank', default=0, type=int, help='shard_id: node rank for distributed training')
    parser.add_argument("--datatype", default="msrvtt", type=str, help="msvd | msrvtt | vatexEnglish | msrvttfull")

    # for different vocab size
    parser.add_argument('--vocab_size', type=int, default=49408, help="the number of vocab size")

    # for TDB block
    parser.add_argument('--temporal_type', type=str, default='', help="TDB type")
    parser.add_argument('--temporal_proj', type=str, default='', help="sigmoid_mlp | sigmoid_selfA")

    # for TAB block
    parser.add_argument('--center_type', type=str, default='', help="TAB")
    parser.add_argument('--centerK', type=int, default=5, help='center number for clustering.')
    parser.add_argument('--center_weight', type=float, default=0.5, help='the weight to adopt the main simiarility')
    parser.add_argument('--center_proj', type=str, default='', help='TAB | TAB_TDB')

    # model path of clip
    parser.add_argument('--clip_path', type=str,
                        default='/data/ceph_11015/ssd/howiefang/videoCLIP/CLIP2Clip/ViT-B-32.pt',
                        help="model path of CLIP")

    args = parser.parse_args()
    return args
