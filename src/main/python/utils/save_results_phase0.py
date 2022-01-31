
from src.main.python.utils.general import print_in_file
import numpy as np
import socket
from datetime import datetime
import json


def save_results(args, info):
    # saving results in file
    string = f"""times: {info['time_for_reduce']}\n
				reducion: {info['reducion']}\n
				time mean: {np.mean(info['time_for_reduce'])}\n
				reducion mean: {np.mean(info['reducion'])}\n
				reduced_len_mean: {np.mean(info['reduced_len'])}\n
				original_len_mean: {np.mean(info['original_len'])}"""

    print_in_file(string, args.filename)

    info["time_mean"] = np.mean(info['time_for_reduce'])
    info["reduction_mean"] = np.mean(info['reducion'])
    info["reduced_len_mean"] = np.mean(info['reduced_len'])
    info["original_len_mean"] = np.mean(info['original_len'])

    args.end_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    # Saving results in json
    data = {
        "hiperparams": str(args),
        "dataset": args.dataset,
        "method": args.method,
        "info": info,
        "machine": socket.gethostname(),
    }

    if args.method == 'rus':
        data["sel"] = args.sel

    with open(args.filename+".json", 'w') as outfile:
        json.dump(data, outfile, indent=4)
