"""
utilities helper file for main file running model benchmarks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import os
import sys
import csv
import json
import subprocess
from google.cloud import storage
from utilization_helper import FindJSONKeyValue, FindOverviewFiles, DrawUtilization, UtilizationHelper
from similarity_helper import SimilarityHelper
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
os.dup2(sys.stdout.fileno(), 1) # dup stdout to system 
os.dup2(sys.stdout.fileno(), 2) # dup stderr to system

FLAGS = tf.flags.FLAGS


def clear_flags(FLAGS, ignore_flags):
	flags_dict = FLAGS._flags()
	keys_list = flags_dict.keys()
	for keys in keys_list:
		if(keys not in ignore_flags):
			FLAGS.__delattr__(keys)

def try_flag_string(name, default, help_str):
	try:
		tf.flags.DEFINE_string(name, default, help_str)
	except:
		# Already exists from imports
		pass

def try_flag_bool(name, default, help_str):
	try:
		tf.flags.DEFINE_bool(name, default, help_str)
	except:
		# Already exists from imports
		pass

def PrintFlags(flag_obj):
	tf.logging.info("==== BENCH_FLAGS: ====")
	for f in flag_obj:
		tf.logging.info(str(f) + ":\t" + str(getattr(flag_obj, f)) )
	tf.logging.info("======================")

def CleanUp(logdir, added_path=[]):
  """
  CleanUp() 
  uploads any file in the cwd with 'TPUPoint' in its name to the logdir specified
  and removes them from the cwd
  """	
  # gs://bucket_name/path --> 'gs:','','bucket_name','path'
  bucket_name = str(logdir).split('/')[2]
  bucket_dir = os.path.join( *str(logdir).split('/')[3:] )

  cwd = os.getcwd()
  stored_files = []
  for r,d,f in os.walk(cwd):
    for file in f:
      if("TPUPoint" in file):
        stored_files.append( os.path.join(r,file) )

  storage_client = storage.Client()
  bucket = storage_client.bucket(bucket_name)

  for stored_file in stored_files:
    stored_file_dir = stored_file
    stored_file_basename = os.path.basename(stored_file)

    destination_blob_name_dir = [bucket_dir, 'TPUPoint'] + added_path + [stored_file_basename]
    # destination_blob_name  = os.path.join( bucket_dir , 'TPUPoint', stored_file_basename )
    destination_blob_name  = os.path.join( *destination_blob_name_dir )
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(stored_file_dir)

    os.remove( stored_file_dir )

class Tee(object):
	def __init__(self):
		self.process = None
		self.old_fd_stdout = None
		self.old_fd_stderr = None
		self.filename = None

	def start(self, filename):

		self.old_fd_stdout = os.dup( sys.stdout.fileno() ) # store original stdout
		self.old_fd_stderr = os.dup( sys.stderr.fileno() ) # store original stdout
		self.process = subprocess.Popen(["tee", filename], stdin=subprocess.PIPE ) # tee pipe subprocess
		os.dup2( self.process.stdin.fileno() , 1 ) # stdout --> subprocess.stdin_fd
		os.dup2( self.process.stdin.fileno() , 2 ) # stderr --> subprocess.stdin_fd
		print(filename)
		time.sleep(2)
		self.filename = filename

	def stop(self):
		os.dup2( self.old_fd_stdout , 1 ) # stdin --> terminal.stdout_fd
		os.dup2( self.old_fd_stderr , 2 ) # stderr --> terminal.stderr_fd

		self.process.terminate()
		self.process = None
		self.old_fd_stdout = None
		self.old_fd_stderr = None

	def stop_and_upload(self, path):
		time.sleep(2) # Finish final print outs
		os.dup2( self.old_fd_stdout , 1 ) # stdin --> terminal.stdout_fd
		os.dup2( self.old_fd_stderr , 2 ) # stderr --> terminal.stderr_fd

		self.process.terminate()
		self.process = None
		self.old_fd_stdout = None
		self.old_fd_stderr = None

		bucket_name = path.split('/')[2]
		storage_client = storage.Client()
		bucket = storage_client.bucket(bucket_name)

		source_file_name = self.filename
		destination_blob_name  = os.path.join(path , source_file_name)
		destination_blob_name  = destination_blob_name.replace('gs://','')
		destination_blob_name  = destination_blob_name.replace(bucket_name + '/','')

		blob = bucket.blob(destination_blob_name)
		blob.upload_from_filename(source_file_name)
		self.filename = None




def run_bert(FLAGS, ignore_flags):
	"""
	Setup and run of the BERT model on the MPRC/SQuAD dataset

	python run_classifier.py \
	--task_name=${TASK_NAME} \
	--do_train=true \
	--do_eval=true \
	--data_dir=${GLUE_DIR}/${TASK_NAME} \
	--vocab_file=${BERT_BASE_DIR}/vocab.txt \
	--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
	--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=3.0 \
	--output_dir=${MODEL_DIR}/\
	--use_tpu=True \
	--tpu_name=${TPU_NAME} | tee output.txt

	python run_squad.py \
	--vocab_file=$BERT_LARGE_DIR/vocab.txt \
	--bert_config_file=$BERT_LARGE_DIR/bert_config.json \
	--init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
	--do_train=True \
	--train_file=$SQUAD_DIR/train-v1.1.json \
	--do_predict=True \
	--predict_file=$SQUAD_DIR/dev-v1.1.json \
	--train_batch_size=24 \
	--learning_rate=3e-5 \
	--num_train_epochs=2.0 \
	--max_seq_length=384 \
	--doc_stride=128 \
	--output_dir=gs://some_bucket/squad_large/ \
	--use_tpu=True \
	--tpu_name=$TPU_NAME
	"""
	clear_flags(FLAGS, ignore_flags)
	from bert_script import *
	BERT_BASE_DIR="gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12"

	FLAGS.do_train = True
	FLAGS.do_eval = True
	FLAGS.vocab_file = os.path.join( BERT_BASE_DIR, 'vocab.txt' )
	FLAGS.bert_config_file = os.path.join( BERT_BASE_DIR , 'bert_config.json' )
	FLAGS.init_checkpoint = os.path.join( BERT_BASE_DIR, 'bert_model.ckpt')
	# max_seq_length
	# train_batch_size
	# learning_rate
	# FLAGS.num_train_epochs = 3.0 
	FLAGS.use_tpu = FLAGS.bench_use_tpu
	FLAGS.tpu_name = FLAGS.bench_tpu_name
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone
	FLAGS.gcp_project = FLAGS.bench_gcp_project


	tee = Tee()

	### MRPC , COLA, XNLI

	for task_name in ["MRPC", "COLA", "MNLI"]:
		task_name_lower = task_name.lower()
		task_name_upper = task_name.upper()

		FLAGS.task_name = task_name
		data_local = {"cola": 'CoLA' , 'mrpc':'MRPC', 'mnli': "MNLI"}[task_name_lower]
		FLAGS.data_dir = os.path.join( FLAGS.bench_data_location , "BERT" , data_local )

		if(FLAGS.bench_tpupoint_baseline):
			if(FLAGS.bench_run_train):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "baseline" )
				tee.start("bert_"+task_name_lower+"_baseline_train.txt")
				PrintFlags(FLAGS)
				# bert_run_baseline()
				p = multiprocessing.Process( target=bert_run_baseline , args=() )
				p.start()
				p.join()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_run_eval):
				tee.start("bert_"+task_name_lower+"_baseline_eval.txt")
				bert_run_eval()
				tee.stop_and_upload(FLAGS.output_dir)

		if(FLAGS.bench_tpupoint_profile):
			if(FLAGS.bench_run_train):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "profile" )
				tee.start("bert_"+task_name_lower+"_profile_train.txt")
				PrintFlags(FLAGS)
				# bert_run_profile()
				p = multiprocessing.Process( target=bert_run_profile , args=() )
				p.start()
				p.join()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_run_eval):
				tee.start("bert_"+task_name_lower+"_profile_eval.txt")
				bert_run_eval()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_csv_utilization):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "profile" )
				UtilizationHelper(dir_path=FLAGS.output_dir, name="BERT_"+task_name_upper)
				CleanUp(FLAGS.output_dir, added_path=['utilization'])
			if(FLAGS.bench_csv_similarity):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "profile" )
				SimilarityHelper(dir_path=FLAGS.output_dir, name="BERT_"+task_name_upper)
				CleanUp(FLAGS.output_dir, added_path=['similarity'])

		if(FLAGS.bench_tpupoint_optimize):
			if(FLAGS.bench_run_train):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "optimize" )
				tee.start("bert_"+task_name_lower+"_optimize_train.txt")
				PrintFlags(FLAGS)
				# bert_run_optimize()
				p = multiprocessing.Process( target=bert_run_optimize , args=() )
				p.start()
				p.join()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_csv_utilization):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "optimize" )
				UtilizationHelper(dir_path=FLAGS.output_dir, name="BERT_"+task_name_upper)
				CleanUp(FLAGS.output_dir, added_path=['utilization'])
			if(FLAGS.bench_run_eval):
				tee.start("bert_"+task_name_lower+"_optimize_eval.txt")
				bert_run_eval()
				tee.stop_and_upload(FLAGS.output_dir)

		if(FLAGS.bench_tpupoint_dynamic):
			if(FLAGS.bench_run_train):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "dynamic" )
				tee.start("bert_"+task_name_lower+"_dynamic_train.txt")
				PrintFlags(FLAGS)
				# bert_run_dynamic()
				p = multiprocessing.Process( target=bert_run_dynamic , args=() )
				p.start()
				p.join()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_csv_utilization):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "dynamic" )
				UtilizationHelper(dir_path=FLAGS.output_dir, name="BERT_"+task_name_upper)
				CleanUp(FLAGS.output_dir, added_path=['utilization'])
			if(FLAGS.bench_run_eval):
				tee.start("bert_"+task_name_lower+"_dynamic_eval.txt")
				bert_run_eval()
				tee.stop_and_upload(FLAGS.output_dir)

		if(FLAGS.bench_naive_wo_tpupoint):
			if(FLAGS.bench_run_train):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "naive_wo_tpupoint" )
				tee.start("bert_"+task_name_lower+"_naive_wo_tpupoint_train.txt")
				PrintFlags(FLAGS)
				p = multiprocessing.Process( target=bert_run_naive_wo_tpupoint , args=() )
				p.start()
				p.join()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_run_eval):
				tee.start("bert_"+task_name_lower+"_naive_wo_tpupoint_eval.txt")
				bert_run_eval()
				tee.stop_and_upload(FLAGS.output_dir)

		if(FLAGS.bench_naive_w_tpupoint):
			if(FLAGS.bench_run_train):
				FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_"+task_name_upper , "naive_w_tpupoint" )
				tee.start("bert_"+task_name_lower+"_naive_w_tpupoint_train.txt")
				PrintFlags(FLAGS)
				p = multiprocessing.Process( target=bert_run_naive_w_tpupoint , args=() )
				p.start()
				p.join()
				tee.stop_and_upload(FLAGS.output_dir)
			if(FLAGS.bench_run_eval):
				tee.start("bert_"+task_name_lower+"_naive_w_tpupoint_eval.txt")
				bert_run_eval()
				tee.stop_and_upload(FLAGS.output_dir)	



	## SQuAD

	clear_flags(FLAGS, ignore_flags)
	from bert_squad_script import *

	FLAGS.vocab_file = os.path.join( BERT_BASE_DIR, 'vocab.txt' )
	FLAGS.bert_config_file = os.path.join( BERT_BASE_DIR , 'bert_config.json' )
	FLAGS.init_checkpoint = os.path.join( BERT_BASE_DIR, 'bert_model.ckpt')
	FLAGS.train_file = os.path.join( FLAGS.bench_data_location , "SQUAD" , "train-v1.1.json" )
	FLAGS.predict_file = os.path.join( FLAGS.bench_data_location , "SQUAD" , "dev-v1.1.json" )
	# FLAGS.train_batch_size=24
	# FLAGS.learning_rate=3e-5
	# FLAGS.num_train_epochs=2.0
	# FLAGS.max_seq_length=384
	FLAGS.use_tpu=FLAGS.bench_use_tpu
	FLAGS.tpu_name=FLAGS.bench_tpu_name
	FLAGS.do_train=True
	FLAGS.do_predict=True

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "baseline" )
			tee.start("bert_squad_baseline_train.txt")
			PrintFlags(FLAGS)
			# bert_run_squad_baseline()
			p = multiprocessing.Process( target=bert_run_squad_baseline , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_run_eval):
			tee.start("bert_squad_baseline_eval.txt")
			bert_run_squad_eval()
			tee.stop_and_upload(FLAGS.output_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "profile" )
			tee.start("bert_squad_profile_train.txt")
			PrintFlags(FLAGS)
			# bert_run_squad_profile()
			p = multiprocessing.Process( target=bert_run_squad_profile , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_run_eval):
			tee.start("bert_squad_profile_eval.txt")
			bert_run_squad_eval()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "profile" )
			UtilizationHelper(dir_path=FLAGS.output_dir, name="BERT_SQUAD")
			CleanUp(FLAGS.output_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "profile" )
			SimilarityHelper(dir_path=FLAGS.output_dir, name="BERT_SQUAD")
			CleanUp(FLAGS.output_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "optimize" )
			tee.start("bert_squad_optimize_train.txt")
			PrintFlags(FLAGS)
			# bert_run_squad_optimize()
			p = multiprocessing.Process( target=bert_run_squad_optimize , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "optimize" )
			UtilizationHelper(dir_path=FLAGS.output_dir, name="BERT_SQUAD")
			CleanUp(FLAGS.output_dir, added_path=['utilization'])
		if(FLAGS.bench_run_eval):
			tee.start("bert_squad_optimize_eval.txt")
			bert_run_squad_eval()
			tee.stop_and_upload(FLAGS.output_dir)

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "dynamic" )
			tee.start("bert_squad_dynamic_train.txt")
			PrintFlags(FLAGS)
			# bert_run_squad_dynamic()
			p = multiprocessing.Process( target=bert_run_squad_dynamic , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "dynamic" )
			UtilizationHelper(dir_path=FLAGS.output_dir, name="BERT_SQUAD")
			CleanUp(FLAGS.output_dir, added_path=['utilization'])
		if(FLAGS.bench_run_eval):
			tee.start("bert_squad_dynamic_eval.txt")
			bert_run_squad_eval()
			tee.stop_and_upload(FLAGS.output_dir)

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "naive_wo_tpupoint" )
			tee.start("bert_squad_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=bert_run_squad_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_run_eval):
			tee.start("bert_squad_naive_wo_tpupoint_eval.txt")
			bert_run_squad_eval()
			tee.stop_and_upload(FLAGS.output_dir)

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.output_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "BERT_SQUAD" , "naive_w_tpupoint" )
			tee.start("bert_squad_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=bert_run_squad_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.output_dir)
		if(FLAGS.bench_run_eval):
			tee.start("bert_squad_naive_w_tpupoint_eval.txt")
			bert_run_squad_eval()
			tee.stop_and_upload(FLAGS.output_dir)


def run_dcgan(FLAGS, ignore_flags):
	"""
	Setup and run of the DCGAN model on the MNIST dataset

	python dcgan_main.py \
	--tpu=${TPU_NAME} \
	--gcp_project=${GCP_PROJECT} \
	--tpu_zone=${TPU_ZONE} \
	--dataset='cifar'/'mnist' \
	--model_dir=${MODEL_DIR} \
	--batch_size=1024 \
	--num_shards=8 \
	--train_steps=10000 \
	--train_steps_per_eval=1000 \
	--iterations_per_loop=100 \
	--learning_rate=0.0002 \
	--eval_loss=False \
	--use_tpu=True \
	--cifar_train_data_file='gs://bucket/CIFAR-10/train.tfrecords' \
	--cifar_test_data_file='gs://bucket/CIFAR-10/eval.tfrecords' \
	--mnist_train_data_file='gs://bucket/MNIST/mnist_train.tfrecord' \
	--mnist_test_data_file='gs://bucket/MNIST/mnist_test.tfrecord' \
	"""
	clear_flags( FLAGS, ignore_flags)
	from dcgan_script import *

	FLAGS.tpu = FLAGS.bench_tpu_name
	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.tpu_zone= FLAGS.bench_tpu_zone
	# FLAGS.batch_size=1024 
	FLAGS.num_shards = 8 
	# FLAGS.train_steps=10000 
	# FLAGS.train_steps_per_eval=1000 
	# FLAGS.iterations_per_loop=100 
	# FLAGS.learning_rate=0.0002 
	# FLAGS.eval_loss=False 
	FLAGS.use_tpu= FLAGS.bench_use_tpu

	### MNIST

	FLAGS.dataset='mnist' 
	FLAGS.mnist_train_data_file = os.path.join( FLAGS.bench_data_location , "MNIST" , "mnist_train.tfrecord" )
	FLAGS.mnist_test_data_file  = os.path.join( FLAGS.bench_data_location , "MNIST" , "mnist_test.tfrecord" )

	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "baseline" )
			tee.start("dcgan_mnist_baseline_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_baseline()
			p = multiprocessing.Process( target=dcgan_run_baseline , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_mnist_baseline_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "profile" )
			tee.start("dcgan_mnist_profile_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_profile()
			p = multiprocessing.Process( target=dcgan_run_profile , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_mnist_profile_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "profile" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="DCGAN_MNIST")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "profile" )
			SimilarityHelper(dir_path=FLAGS.model_dir, name="DCGAN_MNIST")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "optimize" )
			tee.start("dcgan_mnist_optimize_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_optimize()
			p = multiprocessing.Process( target=dcgan_run_optimize , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "optimize" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="DCGAN_MNIST")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_mnist_optimize_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "dynamic" )
			tee.start("dcgan_mnist_dynamic_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=dcgan_run_dynamic , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "dynamic" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="DCGAN_MNIST")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_mnist_dynamic_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "naive_wo_tpupoint" )
			tee.start("dcgan_mnist_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=dcgan_run_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_mnist_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_MNIST" , "naive_w_tpupoint" )
			tee.start("dcgan_mnist_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=dcgan_run_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_mnist_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		


	### CIFAR10

	FLAGS.dataset='cifar' 
	FLAGS.cifar_train_data_file = os.path.join( FLAGS.bench_data_location , "CIFAR10" , "cifar10_train.tfrecord" )
	FLAGS.cifar_test_data_file  = os.path.join( FLAGS.bench_data_location , "CIFAR10" , "cifar10_test.tfrecord" )

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "baseline" )
			tee.start("dcgan_cifar10_baseline_train.txt")
			# dcgan_run_baseline()
			p = multiprocessing.Process( target=dcgan_run_baseline , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_cifar10_baseline_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "profile" )
			tee.start("dcgan_cifar10_profile_train.txt")
			# dcgan_run_profile()
			p = multiprocessing.Process( target=dcgan_run_profile , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_cifar10_profile_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "profile" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="DCGAN_CIFAR10")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "profile" )
			SimilarityHelper(dir_path=FLAGS.model_dir, name="DCGAN_CIFAR10")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "optimize" )
			tee.start("dcgan_cifar10_optimize_train.txt")
			# dcgan_run_optimize()
			p = multiprocessing.Process( target=dcgan_run_optimize , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "optimize" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="DCGAN_CIFAR10")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_cifar10_optimize_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "dynamic" )
			tee.start("dcgan_cifar10_dynamic_train.txt")
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=dcgan_run_dynamic , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "dynamic" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="DCGAN_CIFAR10")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_cifar10_dynamic_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "naive_wo_tpupoint" )
			tee.start("dcgan_cifar10_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=dcgan_run_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_cifar10_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "DCGAN_CIFAR10" , "naive_w_tpupoint" )
			tee.start("dcgan_cifar10_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=dcgan_run_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("dcgan_cifar10_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		


def run_qanet(FLAGS, ignore_flags):
	"""
	Setup and run of the QaNET model on the SQuAD dataset
	run.py \
	--tpu=$TPU_NAME \
	--data_path=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--config=dataset.train_batch_size=32,steps_per_epoch=20000,num_epochs=5 \
	"""
	clear_flags(FLAGS, ignore_flags)
	from qanet_script import *

	FLAGS.tpu = FLAGS.bench_tpu_name
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone
	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.data_path = os.path.join( FLAGS.bench_data_location , "SQUAD" )
	FLAGS.config = "dataset.train_batch_size=32,steps_per_epoch=20000,num_epochs=5"

	### SQuAD


	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "baseline" )
			tee.start("qanet_squad_baseline_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_baseline()
			p = multiprocessing.Process( target=qanet_run_baseline , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_baseline_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "profile" )
			tee.start("qanet_squad_profile_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_profile()
			p = multiprocessing.Process( target=qanet_run_profile , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_profile_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "profile" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "profile" )
			SimilarityHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "optimize" )
			tee.start("qanet_squad_optimize_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_optimize()
			p = multiprocessing.Process( target=qanet_run_optimize , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "optimize" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])			
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_optimize_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "dynamic" )
			tee.start("qanet_squad_dynamic_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_dynamic()
			p = multiprocessing.Process( target=qanet_run_dynamic , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "dynamic" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])			
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_dynamic_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "naive_wo_tpupoint" )
			tee.start("qanet_squad_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=qanet_run_squad_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD" , "naive_w_tpupoint" )
			tee.start("qanet_squad_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=qanet_run_squad_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		


def run_resnet(FLAGS, ignore_flags):
	"""
	Setup and run of the ResNet model on the ImageNet dataset
	python resnet_main.py \
	--tpu=$TPU_NAME \
	--data_dir=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--gcp_project=$GCP_PROJECT \
	--tpu_zone=$TPU_ZONE \
	--mode="train" \
	--use_tpu=True
	"""
	clear_flags(FLAGS, ignore_flags)
	from resnet_script import *

	FLAGS.tpu = FLAGS.bench_tpu_name
	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone
	FLAGS.use_tpu = FLAGS.bench_use_tpu
	# FLAGS.train_steps=1 # For testing

	### ImageNet

	FLAGS.data_dir = FLAGS.bench_imagenet_dir
	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "baseline" )
			tee.start("resnet_imagenet_baseline_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_baseline()
			p = multiprocessing.Process( target=resnet_run_baseline , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_imagenet_baseline_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "profile" )
			tee.start("resnet_imagenet_profile_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_profile()
			p = multiprocessing.Process( target=resnet_run_profile , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_imagenet_profile_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "profile" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RESNET_IMAGENET")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "profile" )			
			SimilarityHelper(dir_path=FLAGS.model_dir, name="RESNET_IMAGENET")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "optimize" )
			tee.start("resnet_imagenet_optimize_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_optimize()
			p = multiprocessing.Process( target=resnet_run_optimize , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_imagenet_optimize_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "optimize" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RESNET_IMAGENET")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "dynamic" )
			tee.start("resnet_imagenet_dynamic_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_dynamic()
			p = multiprocessing.Process( target=resnet_run_dynamic , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_imagenet_dynamic_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "dynamic" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RESNET_IMAGENET")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "naive_wo_tpupoint" )
			tee.start("resnet_imagenet_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=resnet_run_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_imagenet_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_IMAGENET" , "naive_w_tpupoint" )
			tee.start("resnet_imagenet_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=resnet_run_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_imagenet_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		


def run_retinanet(FLAGS, ignore_flags):
	"""
	Setup and run of the retinanet model on the COCO dataset

	python tpu/models/official/retinanet/retinanet_main.py \
	--tpu=${TPU_NAME} \
	--train_batch_size=64 \
	--training_file_pattern=${DATA_DIR}/train-* \
	--resnet_checkpoint=${RESNET_CHECKPOINT} \
	--model_dir=${MODEL_DIR} \
	--hparams=image_size=640 \
	--num_examples_per_epoch=100 \
	--num_epochs=1
	"""
	clear_flags( FLAGS, ignore_flags)
	from retinanet_script import *

	## TESTING ###
	# FLAGS.train_batch_size=64
	# FLAGS.num_examples_per_epoch=100
	# FLAGS.num_epochs=1
	##############

	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone

	FLAGS.tpu=FLAGS.bench_tpu_name
	FLAGS.training_file_pattern=os.path.join(FLAGS.bench_coco_dir , 'train-*' )
	FLAGS.validation_file_pattern=os.path.join(FLAGS.bench_coco_dir , 'val-*' )
	FLAGS.eval_batch_size=8
	FLAGS.hparams='image_size=640'

	val_json = 'instances_val2017.json'
	local_file = os.path.join( os.getcwd() , val_json ) 

	if( not os.path.exists( local_file ) ):
		val_json_file = os.path.join(FLAGS.bench_coco_dir , val_json )
		with tf.io.gfile.GFile(val_json_file, 'r') as gfile:
			local = open(local_file , 'w')
			local.write( gfile.read() )

	FLAGS.val_json_file=local_file


	FLAGS.resnet_checkpoint=os.path.join( FLAGS.bench_data_location , "RESNETCKPT" )
	FLAGS.use_tpu=FLAGS.bench_use_tpu

	### coco

	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "baseline" )
			tee.start("retinanet_coco_baseline_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_baseline()
			p = multiprocessing.Process( target=retinanet_run_baseline , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_baseline_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "profile" )
			tee.start("retinanet_coco_profile_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_profile()
			p = multiprocessing.Process( target=retinanet_run_profile , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_profile_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "profile" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "profile" )			
			SimilarityHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "optimize" )
			tee.start("retinanet_coco_optimize_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_optimize()
			p = multiprocessing.Process( target=retinanet_run_optimize , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_optimize_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "optimize" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "dynamic" )
			tee.start("retinanet_coco_dynamic_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_dynamic()
			p = multiprocessing.Process( target=retinanet_run_dynamic , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_dynamic_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "dynamic" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "naive_wo_tpupoint" )
			tee.start("retinanet_coco_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=retinanet_run_squad_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO" , "naive_w_tpupoint" )
			tee.start("retinanet_coco_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=retinanet_run_squad_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		


def run_qanet_small(FLAGS, ignore_flags):
	"""
	Setup and run of the QaNET model on the SQuAD dataset
	run.py \
	--tpu=$TPU_NAME \
	--data_path=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--config=dataset.train_batch_size=32,steps_per_epoch=20000,num_epochs=5 \
	"""
	clear_flags(FLAGS, ignore_flags)
	from qanet_script import *

	FLAGS.tpu = FLAGS.bench_tpu_name
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone
	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.data_path = os.path.join( FLAGS.bench_data_location , "SQUAD" )
	FLAGS.config = "dataset.train_batch_size=32,steps_per_epoch=10000,num_epochs=5"

	### SQuAD


	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "baseline" )
			tee.start("qanet_squad_small_baseline_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_baseline()
			p = multiprocessing.Process( target=qanet_run_baseline , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_small_baseline_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "profile" )
			tee.start("qanet_squad_small_profile_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_profile()
			p = multiprocessing.Process( target=qanet_run_profile , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_small_profile_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "profile" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "profile" )
			SimilarityHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "optimize" )
			tee.start("qanet_squad_small_optimize_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_optimize()
			p = multiprocessing.Process( target=qanet_run_optimize , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "optimize" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])			
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_small_optimize_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "dynamic" )
			tee.start("qanet_squad_small_dynamic_train.txt")
			PrintFlags(FLAGS)
			# qanet_run_dynamic()
			p = multiprocessing.Process( target=qanet_run_dynamic , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "dynamic" )
			UtilizationHelper(dir_path=FLAGS.model_dir, name="QANET_SQUAD_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])			
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_small_dynamic_eval.txt")
			qanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "naive_wo_tpupoint" )
			tee.start("qanet_squad_small_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=qanet_run_squad_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_small_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "QANET_SQUAD_SMALL" , "naive_w_tpupoint" )
			tee.start("qanet_squad_small_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			# dcgan_run_dynamic()
			p = multiprocessing.Process( target=qanet_run_squad_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("qanet_squad_small_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)


def run_retinanet_small(FLAGS, ignore_flags):
	"""
	Setup and run of the retinanet model on the COCO dataset

	python tpu/models/official/retinanet/retinanet_main.py \
	--tpu=${TPU_NAME} \
	--train_batch_size=64 \
	--training_file_pattern=${DATA_DIR}/train-* \
	--resnet_checkpoint=${RESNET_CHECKPOINT} \
	--model_dir=${MODEL_DIR} \
	--hparams=image_size=640 \
	--num_examples_per_epoch=100 \
	--num_epochs=1
	"""
	clear_flags( FLAGS, ignore_flags)
	from retinanet_script import *

	# FLAGS.train_batch_size=64
	FLAGS.num_examples_per_epoch=60000
	# FLAGS.num_epochs=1

	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone

	FLAGS.tpu=FLAGS.bench_tpu_name
	FLAGS.training_file_pattern=os.path.join(FLAGS.bench_coco_dir , 'train-*' )
	FLAGS.validation_file_pattern=os.path.join(FLAGS.bench_coco_dir , 'val-*' )
	FLAGS.eval_batch_size=8
	FLAGS.hparams='image_size=640'

	val_json = 'instances_val2017.json'
	local_file = os.path.join( os.getcwd() , val_json ) 

	if( not os.path.exists( local_file ) ):
		val_json_file = os.path.join(FLAGS.bench_coco_dir , val_json )
		with tf.io.gfile.GFile(val_json_file, 'r') as gfile:
			local = open(local_file , 'w')
			local.write( gfile.read() )

	FLAGS.val_json_file=local_file


	FLAGS.resnet_checkpoint=os.path.join( FLAGS.bench_data_location , "RESNETCKPT" )
	FLAGS.use_tpu=FLAGS.bench_use_tpu

	### coco

	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "baseline" )
			tee.start("retinanet_coco_small_baseline_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_baseline()
			p = multiprocessing.Process( target=retinanet_run_baseline , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_small_baseline_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "profile" )
			tee.start("retinanet_coco_small_profile_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_profile()
			p = multiprocessing.Process( target=retinanet_run_profile , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_small_profile_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "profile" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "profile" )			
			SimilarityHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "optimize" )
			tee.start("retinanet_coco_small_optimize_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_optimize()
			p = multiprocessing.Process( target=retinanet_run_optimize , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_small_optimize_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "optimize" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "dynamic" )
			tee.start("retinanet_coco_small_dynamic_train.txt")
			PrintFlags(FLAGS)
			# retinanet_run_dynamic()
			p = multiprocessing.Process( target=retinanet_run_dynamic , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_small_dynamic_eval.txt")
			retinanet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "dynamic" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RETINANET_COCO_SMALL")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "naive_wo_tpupoint" )
			tee.start("retinanet_coco_small_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=retinanet_run_squad_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_small_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RETINANET_COCO_SMALL" , "naive_w_tpupoint" )
			tee.start("retinanet_coco_small_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=retinanet_run_squad_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("retinanet_coco_small_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)	


def run_resnet_cifar(FLAGS, ignore_flags):
	"""
	Setup and run of the ResNet model on the ImageNet dataset
	python resnet_main.py \
	--tpu=$TPU_NAME \
	--data_dir=$DATA_DIR \
	--model_dir=$MODEL_DIR \
	--gcp_project=$GCP_PROJECT \
	--tpu_zone=$TPU_ZONE \
	--mode="train" \
	--use_tpu=True
	"""
	clear_flags(FLAGS, ignore_flags)
	from resnet_cifar_script import *

	FLAGS.tpu = FLAGS.bench_tpu_name
	FLAGS.gcp_project = FLAGS.bench_gcp_project
	FLAGS.tpu_zone = FLAGS.bench_tpu_zone
	FLAGS.use_tpu = FLAGS.bench_use_tpu
	# FLAGS.train_steps=1 # For testing

	### ImageNet
	# FLAGS.data_dir = FLAGS.bench_imagenet_dir
	FLAGS.data_dir = os.path.join( FLAGS.bench_data_location , "CIFAR10" , "cifar10_train.tfrecord" )


	### CIFAR10
	# FLAGS.cifar_train_data_file = os.path.join( FLAGS.bench_data_location , "CIFAR10" , "cifar10_train.tfrecord" )
	# FLAGS.cifar_test_data_file  = os.path.join( FLAGS.bench_data_location , "CIFAR10" , "cifar10_test.tfrecord" )
	# FLAGS.noise_dim = 64
	FLAGS.image_size = 32
	FLAGS.num_label_classes = 10
	FLAGS.base_learning_rate = 0.000005

	tee = Tee()

	if(FLAGS.bench_tpupoint_baseline):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "baseline" )
			tee.start("resnet_cifar_baseline_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_baseline()
			p = multiprocessing.Process( target=resnet_run_baseline , args=() )
			p.start()
			p.join()	
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_cifar_baseline_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)

	if(FLAGS.bench_tpupoint_profile):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "profile" )
			tee.start("resnet_cifar_profile_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_profile()
			p = multiprocessing.Process( target=resnet_run_profile , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_cifar_profile_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "profile" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RESNET_CIFAR")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])
		if(FLAGS.bench_csv_similarity):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "profile" )			
			SimilarityHelper(dir_path=FLAGS.model_dir, name="RESNET_CIFAR")
			CleanUp(FLAGS.model_dir, added_path=['similarity'])

	if(FLAGS.bench_tpupoint_optimize):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "optimize" )
			tee.start("resnet_cifar_optimize_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_optimize()
			p = multiprocessing.Process( target=resnet_run_optimize , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_cifar_optimize_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "optimize" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RESNET_CIFAR")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_tpupoint_dynamic):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "dynamic" )
			tee.start("resnet_cifar_dynamic_train.txt")
			PrintFlags(FLAGS)
			# resnet_run_dynamic()
			p = multiprocessing.Process( target=resnet_run_dynamic , args=() )
			p.start()
			p.join() 
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_cifar_dynamic_eval.txt")
			resnet_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_csv_utilization):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "dynamic" )			
			UtilizationHelper(dir_path=FLAGS.model_dir, name="RESNET_CIFAR")
			CleanUp(FLAGS.model_dir, added_path=['utilization'])

	if(FLAGS.bench_naive_wo_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "naive_wo_tpupoint" )
			tee.start("resnet_cifar_naive_wo_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=resnet_run_naive_wo_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_cifar_naive_wo_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		

	if(FLAGS.bench_naive_w_tpupoint):
		if(FLAGS.bench_run_train):
			FLAGS.model_dir = os.path.join( FLAGS.bench_model_location , FLAGS.bench_tpu_version , "RESNET_CIFAR" , "naive_w_tpupoint" )
			tee.start("resnet_cifar_naive_w_tpupoint_train.txt")
			PrintFlags(FLAGS)
			p = multiprocessing.Process( target=resnet_run_naive_w_tpupoint , args=() )
			p.start()
			p.join()
			tee.stop_and_upload(FLAGS.model_dir)
		if(FLAGS.bench_run_eval):
			tee.start("resnet_cifar_naive_w_tpupoint_eval.txt")
			dcgan_run_eval()
			tee.stop_and_upload(FLAGS.model_dir)		


