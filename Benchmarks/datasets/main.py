"""
main file for downloading datasets into the 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import os
import sys
from google.cloud import storage
import requests
import zipfile
import shutil

import download_and_convert_mnist_raw as mnist_raw
import preprocess as squad
import download_glue_data as bert
import download_and_convert_cifar10 as cifar10
import movielens

tf.flags.DEFINE_string("data_dir", None, "GCS bucket to place datasets in the form of 'gs://bucket_name'")
tf.flags.DEFINE_bool("mnist", False, "Download mnist dataset and upload to GCS.")
tf.flags.DEFINE_bool("SQUAD", False, "Download SQuAD dataset and upload to GCS.")
tf.flags.DEFINE_bool("BERT", False, "Download BERT dataset and upload to GCS.")
tf.flags.DEFINE_bool("CIFAR10", False, "Download CIFAR10 dataset and upload to GCS.")
tf.flags.DEFINE_bool("ML20M", False, "Download ML20M dataset and upload to GCS.")
tf.flags.DEFINE_bool("ML1M", False, "Download ML1M dataset and upload to GCS.")
tf.flags.DEFINE_bool("ResNetCheckpoint", False, "Download ResNet-50 checkpoint for object detection models.")
tf.flags.DEFINE_bool("InstallPycocotools", False, "Install Pycocotools required for object detection models.")

FLAGS = tf.flags.FLAGS

ML20M = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
ML1M =  "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

RESNET_BASE_URL="https://storage.googleapis.com/cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07"



def run_mnist():
	print("Starting mnist")
	local_dir = '/tmp/mnist'
	try:
		os.mkdir(local_dir)
	except:
		pass
	mnist_raw.run( local_dir )

	files = ['labels.txt','mnist_test.tfrecord','mnist_train.tfrecord']

	bucket_name = str(FLAGS.data_dir).replace('gs://','')
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	for file in files:
		destination_blob_name  = os.path.join('MNIST', file) 
		source_file_name = os.path.join(local_dir, file)
		blob = bucket.blob(destination_blob_name)
		blob.upload_from_filename(source_file_name)

	try:
		shutil.rmtree(local_dir, ignore_errors=True)
	except:
		pass
	print("Finished Downloading & upliading mnist")

def run_squad():
	local_dir = '/tmp/squad_data'
	try:
		os.mkdir(local_dir)
	except:
		pass

	# curl https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json > $LOCAL_SQUAD_DATA/train-v1.1.json
	out_file = os.path.join(local_dir, 'train-v1.1.json')
	if(not os.path.exists(out_file) ):
		print("Downloading train-v1.1.json")
		url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json'
		r = requests.get(url, allow_redirects=True)
		open(out_file, 'wb').write(r.content)

	# curl https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json > $LOCAL_SQUAD_DATA/dev-v1.1.json
	out_file = os.path.join(local_dir, 'dev-v1.1.json')
	if(not os.path.exists(out_file) ):
		print("Downloading dev-v1.1.json")
		url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json'
		r = requests.get(url, allow_redirects=True)
		open(out_file, 'wb').write(r.content)

	# curl https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip > $LOCAL_SQUAD_DATA/
	out_file = os.path.join(local_dir, 'crawl-300d-2M.vec.zip')
	if(not os.path.exists(out_file) ):
		print("Downloading crawl-300d-2M.vec.zip")
		url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
		r = requests.get(url, allow_redirects=True)
		open(out_file, 'wb').write(r.content)

		print("Unzipping crawl-300d-2M.vec.zip")
		with zipfile.ZipFile(out_file, 'r') as zipObj:
			zipObj.extractall(local_dir)


	"""
	python preprocess.py \
	--input_path $LOCAL_SQUAD_DATA/train-v1.1.json,$LOCAL_SQUAD_DATA/dev-v1.1.json \
	--embedding_path $LOCAL_SQUAD_DATA/crawl-300d-2M.vec \
	--output_path $DATA_DIR
	"""
	FLAGS.input_path = str(os.path.join(local_dir, 'train-v1.1.json'))+","+str(os.path.join(local_dir,'dev-v1.1.json'))
	FLAGS.embedding_path = str(os.path.join(local_dir, 'crawl-300d-2M.vec'))
	FLAGS.max_shard_size = 11000
	FLAGS.output_path = os.path.join( FLAGS.data_dir , 'SQUAD')

	squad.main(sys.argv)


	# gsutil cp dev-v1.1.json train-v1.1.json $DATA_DIR
	print("Uploading dev-v1.1.json & train-v1.1.json to " + str( os.path.join( FLAGS.data_dir , 'SQUAD') ) )
	bucket_name = str(FLAGS.data_dir).replace('gs://','')
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	for file in ['dev-v1.1.json', 'train-v1.1.json']:
		destination_blob_name  = os.path.join('SQUAD', file) 
		source_file_name = os.path.join(local_dir, file)
		blob = bucket.blob(destination_blob_name)
		blob.upload_from_filename(source_file_name)

	try:
		shutil.rmtree(local_dir, ignore_errors=True)
	except:
		pass
	print("Finished Downloading & upliading squad")

def run_bert():
	local_dir = '/tmp/bert'
	try:
		os.mkdir(local_dir)
	except:
		pass
	args = []
	args.append('--data_dir=' + local_dir)
	args.append('--tasks=all')
	bert.main(args)

	stored_files = []
	for r,d,f in os.walk(local_dir):
		for file in f:
			stored_files.append( os.path.join(r,file) )

	print("Uploading bert to " + str(FLAGS.data_dir) )

	bucket_name = str(FLAGS.data_dir).replace('gs://','')
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)

	for stored_file in stored_files:
		source_file_name = stored_file

		file = stored_file.replace(local_dir, '')
		if(file[0] == '/'): file = file[1:]
		destination_blob_name  = os.path.join('BERT', file) 

		blob = bucket.blob(destination_blob_name)
		blob.upload_from_filename(source_file_name)

	try:
		shutil.rmtree(local_dir, ignore_errors=True)
	except:
		pass
	print("Finished Downloading & upliading BERT dataset")

def run_cifar10():
	local_dir = '/tmp/cifar10'
	try:
		os.mkdir(local_dir)
	except:
		pass


	cifar10.run(local_dir)

	print("Uploading CIFAR10 to " + str(FLAGS.data_dir) )

	stored_files = []
	for r,d,f in os.walk(local_dir):
		for file in f:
			stored_files.append( os.path.join(r,file) )

	bucket_name = str(FLAGS.data_dir).replace('gs://','')
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)

	for stored_file in stored_files:
		source_file_name = stored_file

		file = stored_file.replace(local_dir, '')
		if(file[0] == '/'): file = file[1:]
		destination_blob_name  = os.path.join('CIFAR10', file) 

		blob = bucket.blob(destination_blob_name)
		blob.upload_from_filename(source_file_name)

	try:
		shutil.rmtree(local_dir, ignore_errors=True)
	except:
		pass

	print("Finished Downloading & upliading CIFAR10 dataset")

def run_ml20m():
	gcs_dir = os.path.join( str(FLAGS.data_dir) , 'ML20M')
	movielens.download('ml-20m', gcs_dir)

def run_ml1m():
	gcs_dir = os.path.join( str(FLAGS.data_dir) , 'ML1M')
	movielens.download('ml-1m', gcs_dir)

def run_resnet_checkpoint():
	local_dir = '/tmp/resnetckpt'
	try:
		os.mkdir(local_dir)
	except:
		pass

	# wget -N ${BASE_URL}/checkpoint -P ${DEST_DIR}
	# wget -N ${BASE_URL}/model.ckpt-112603.data-00000-of-00001 -P ${DEST_DIR}
	# wget -N ${BASE_URL}/model.ckpt-112603.index  -P ${DEST_DIR}
	# wget -N ${BASE_URL}/model.ckpt-112603.meta -P ${DEST_DIR}

	files = [
		os.path.join( RESNET_BASE_URL , 'checkpoint')  ,
		os.path.join( RESNET_BASE_URL , 'model.ckpt-112603.data-00000-of-00001') ,
		os.path.join( RESNET_BASE_URL , 'model.ckpt-112603.index') ,
		os.path.join( RESNET_BASE_URL , 'model.ckpt-112603.meta')
	]

	local_files = []

	for file in files:
		url = file
		base_name = os.path.basename(file)
		out_file = os.path.join( local_dir,  base_name )
		print("Downloading " + str( base_name ) )
		r = requests.get(url, allow_redirects=True)
		open(out_file, 'wb').write(r.content)
		local_files.append(out_file) 

	

	bucket_name = str(FLAGS.data_dir).replace('gs://','')
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)

	for stored_file in local_files:
		source_file_name = stored_file
		destination_blob_name  = os.path.join( 'RESNETCKPT', os.path.basename(stored_file) )
		
		print("Uploading " + str(destination_blob_name) )

		blob = bucket.blob(destination_blob_name)
		blob.upload_from_filename(source_file_name)

	# out_file = os.path.join(local_dir, 'ml-20m.zip')
	# print("Downloading ml-20m.zip")
	# url = ML20M
	# r = requests.get(url, allow_redirects=True)
	# open(out_file, 'wb').write(r.content)
	# print("Unzipping ml-20m.zip")
	# with zipfile.ZipFile(out_file, 'r') as zipObj:
	# 	zipObj.extractall(local_dir)
	# os.remove(out_file)

	# print("Uploading ml-20m to " + str(FLAGS.data_dir) )

	# stored_files = []
	# for r,d,f in os.walk(local_dir):
	# 	for file in f:
	# 		stored_files.append( os.path.join(r,file) )

	# bucket_name = str(FLAGS.data_dir).replace('gs://','')
	# storage_client = storage.Client()
	# bucket = storage_client.bucket(bucket_name)

	# for stored_file in stored_files:
	# 	source_file_name = stored_file
	# 	destination_blob_name  = os.path.join('ML20M', 'ml-20m', os.path.basename(stored_file) )
	# 	blob = bucket.blob(destination_blob_name)
	# 	blob.upload_from_filename(source_file_name)

	try:
		shutil.rmtree(local_dir, ignore_errors=True)
	except:
		pass

def run_install_pytools():
	cwd = os.getcwd()
	cmd = "pip uninstall pycocotools"
	os.system(cmd)
	cmd = "bash ../models/research/object_detection/dataset_tools/create_pycocotools_package.sh " + cwd
	os.system(cmd)
	cmd = "tar xvzf pycocotools-2.0.tar.gz"
	os.system(cmd)
	os.chdir('./PythonAPI')
	cmd = "cp ../pycocotools/* ./pycocotools"
	os.system(cmd)
	cmd = "sudo python setup.py install"
	ret = os.system(cmd)
	if(ret != 0):
		cmd = "python setup.py install"
		os.system(cmd)
	os.chdir(cwd)

def main(unused_argv):
	
	if(FLAGS.mnist):
		run_mnist()
	if(FLAGS.SQUAD):
		run_squad()
	if(FLAGS.BERT):
		run_bert()
	if(FLAGS.CIFAR10):
		run_cifar10()
	if(FLAGS.ML20M):
		run_ml20m()
	if(FLAGS.ML1M):
		run_ml1m()
	if(FLAGS.ResNetCheckpoint):
		run_resnet_checkpoint()
	if(FLAGS.InstallPycocotools):
		run_install_pytools()

if __name__ == "__main__":
	tf.app.run()