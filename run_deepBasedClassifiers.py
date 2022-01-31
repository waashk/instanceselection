#from inout import *
from src.main.python.utils.general import *
from src.main.python.tClassifiers.deepClassifiers import DeepClassifier
from sklearn.metrics import f1_score
import time
import socket
import gzip
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import confusion_matrix
import gc
import argparse
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from os import path
import json

def arguments():
	parser = argparse.ArgumentParser(description='xlnet.')
	parser.add_argument("--dataset", type=str)
	parser.add_argument("--nfolds", type=int)
	parser.add_argument("--foldesp", type=int, default=0)
	parser.add_argument("--ismethod", type=str)
	parser.add_argument("--out", type=str, default="out/")
	parser.add_argument("--machine", type=str, default=socket.gethostname())
	parser.add_argument("--max_len", type=int, default=150)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--path_selection", type=str , default="../instance-based/out/selection/")
	parser.add_argument("--path_datasets", type=str ,  default="../instance-based/datasets/")
	parser.add_argument("--pretrained_models_path", type=str ,  default="pretrained_models_path")
	parser.add_argument("--base_or_large", type=str ,  default="base")
	parser.add_argument("--deepmethod", type=str ,  default="bert")
	parser.add_argument("--max_iter", type=int, default=20)
	parser.add_argument("--save_model", type=int, default=0)
	parser.add_argument("--load_model", type=int, default=0)
	parser.add_argument("--save_proba", type=int, default=0)
	args = parser.parse_args()
	print(args)
	random.seed(1608637542)
	
	return args

def check_if_result_exists(args, f):

	outdir = f"{args.out}/{args.dataset}/{args.ismethod}/"
	#filename = f"{outdir}/out.fold={f}.json"
	try:
		os.makedirs(outdir)
	except:
		print("Out dir already exist")
	filename = f"{outdir}/out.fold={f}.json"

	if os.path.exists(filename):
		print(f"Already exists results output for {filename}")
		exit()

def save_proba(file, X, y):
	with gzip.open(file, 'w') as filout:
		dump_svmlight_file(X, y, filout, zero_based=False)


from datetime import datetime
#Example of usage: python xlnet.py --dataset aisopos_ntua --nfolds 10 --foldesp 0 --ismethod cnn
if __name__ == '__main__':

	gc.collect()
	args = arguments()

	DATAIN = f"{args.path_datasets}/{args.dataset}/"
	
	if args.ismethod:
		splits = load_splits_ids_for_is(args, DATAIN)
	else: 
		splits = load_splits_ids(f"{DATAIN}/split_{args.nfolds}.csv")

	y = LabelEncoder().fit_transform(np.array(list(map(int, readfile(path.join(DATAIN, 'score.txt'))))))
	X = readfile(path.join(DATAIN, 'texts.txt'))

	for f in [args.foldesp]:

		check_if_result_exists(args, f)

		train_idx, test_idx = splits[f]
		X_train, y_train = get_array(X, train_idx), get_array(y, train_idx)
		X_test, y_test = get_array(X, test_idx), get_array(y, test_idx)

		clf = DeepClassifier(deepmethod=args.deepmethod, batch_num = args.batch_size, max_len= args.max_len,
							 base_or_large=args.base_or_large, max_iter = args.max_iter, learning_rate = args.lr,
							 save_model=args.save_model, load_model=args.load_model, out_dir = f"{args.out}/{args.dataset}/{args.ismethod}/{f}/",
							 pretrained_models_path=args.pretrained_models_path)

		t0_train = time.time()
		clf.fit(X_train, y_train)
		t0_train = time.time() - t0_train

		t0_test = time.time()
		y_pred = clf.predict(X_test)
		t0_test = time.time() - t0_test

		if args.save_proba:
			print("Saving proba...")
			all_proba = clf.predict_proba(X_test)
			outdir = f"{args.out}/{args.dataset}/{args.ismethod}/"

			save_proba(f"{outdir}/proba{f}.gz", all_proba, y_test)

		micro = f1_score(y_true=y_test, y_pred=y_pred, average='micro')
		macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')

		print(micro, macro)

	cm = confusion_matrix(y_test, y_pred).tolist()

	#Saving results in json
	data = {
		"hiperparams": str(args),

		"method": args.deepmethod,
		"machine": args.machine,
		"ismethod": args.ismethod,
		
		"epochs": clf.epoch_id,
		"micro": micro,
		"macro": macro,
		"time_train": t0_train,
		"time_test": t0_test,
		"time_test_avg": t0_test/len(y_test),
		
		"max_patience": clf.max_patience,
		"validation_loss": clf.validation_loss,
		"training_loss": clf.training_loss,

		"cm": cm,
		"y_pred": y_pred.tolist(),
	}

		# current date and time
	now = datetime.now()

	timestamp = datetime.timestamp(now)
	timestamp = str(int(timestamp))
	print("timestamp =", timestamp)


	outdir = f"{args.out}/{args.dataset}/{args.ismethod}/"
	createPath(outdir)
	filename = f"{outdir}/out"
	#with open(f"{filename}.fold={f}.json.{timestamp}", 'w') as outfile:
	with open(f"{filename}.fold={f}.json", 'w') as outfile:
		json.dump(data, outfile, indent=4)



