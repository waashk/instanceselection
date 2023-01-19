
from src.main.python.utils.general import str2bool
from datetime import datetime
import argparse
import os
import random


def check_if_split_exists(args):

    # if args.sel == "":
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_idxinfold.csv"
    # else:
    #	saida = args.outputdir+"split_"+str(args.folds)+"_"+args.method+"_"+args.sel+"_idxinfold.csv"

    saida = args.filename+".json"

    if os.path.exists(saida):
        print("Already exists selection output")
        exit()


def arguments():
    # datasets/webkb/tfidf/ --splitdir datasets/webkb/ --outputdir output/webkb/cnn/
    parser = argparse.ArgumentParser(description='Generate baseline splits.')
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-m', "--method", type=str, help='selection method')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument(
        '--save', type=lambda x: bool(str2bool(x)), default=True)
    parser.add_argument("--inputrep", type=str, default="tfidf")
    parser.add_argument("--out", required=True)
    parser.add_argument("--datain", required=True)
    parser.add_argument("--overwrite", default=0)

    args = parser.parse_args()

    # args.inputdir=f'datasets/{args.dataset}/tfidf/'
    args.inputdir = f'{args.datain}/{args.dataset}/{args.inputrep}/'
    # args.splitdir=f'datasets/{args.dataset}/'
    args.splitdir = f'{args.datain}/{args.dataset}/splits/'
    # args.outputdir=f'outselection2/{args.dataset}/'
    args.outputdir = f'{args.out}/selection/{args.dataset}/'

    args.filename = f"{args.outputdir}/saida_{args.method}"

    args.start_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    print(args)
    check_if_split_exists(args)

    if not os.path.exists(args.outputdir):
        print(f"Criando saida {args.outputdir}")
        os.system("mkdir -p {}".format(args.outputdir))

    with open(args.filename, "w") as arq:
        arq.write(f"{args.method}\n{args}\n")

    random.seed(1608637542)

    info = {
        "reducion": [],
        "time_for_reduce": [],
        "original_len": [],
        "reduced_len": [],
    }

    return args, info
