"""
main file for running model benchmarks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
import os
import sys

import main_util

CWD = os.getcwd()

sys.path.append( os.path.join( os.getcwd() , '..') )


# Basic Running Running Information
tf.flags.DEFINE_string("bench_data_location", None, "GCS bucket where the datasets are located. Written as 'gs://bucket_name'")
tf.flags.DEFINE_string("bench_model_location", None, "GCS bucket to place models, checkpoints, and profiles. Written as 'gs://bucket_name'")
tf.flags.DEFINE_string("bench_tpu_version", None, "tpu version. Either 'TPUv2' or 'TPUv3'")

# What TPUPoint Test to run
main_util.try_flag_bool("bench_tpupoint_baseline", True, "Whether to run the baseline test")
main_util.try_flag_bool("bench_tpupoint_profile", True, "Whether to run the TPUPoint profile test")
main_util.try_flag_bool("bench_tpupoint_optimize", True, "Whether to run the TPUPoint optimization test")
main_util.try_flag_bool("bench_tpupoint_dynamic", True, "Whether to run the TPUPoint dynamic test")
main_util.try_flag_bool("bench_naive_wo_tpupoint", True, "Whether to run naive test without TPUPoint")
main_util.try_flag_bool("bench_naive_w_tpupoint", True, "Whether to run naive test with TPUPoint")
main_util.try_flag_bool("bench_run_train", True, "Whether to run train for each of the baseline/profile/optimize/dynamic tests")
main_util.try_flag_bool("bench_run_eval", True, "Whether to run eval after each of the baseline/profile/optimize/dynamic tests")

# Create CSV for Plots
main_util.try_flag_bool("bench_csv_utilization", True, "Whether to create csv for plot of utilization for profile tests")
main_util.try_flag_bool("bench_csv_similarity", True, "Whether tocreate csv for plot of similarity v.s. #phases for profile tests")



# Flags that may have already been defined
main_util.try_flag_string("bench_tpu_name", None, "The Cloud TPU to use for training. Either TPU name or a grpc://ip.address.of.tpu:8470")
main_util.try_flag_string( "bench_tpu_zone", None, "GCE zone where the Cloud TPU is located in.")
main_util.try_flag_string("bench_gcp_project", None, "Project name for the Cloud TPU-enabled project.")
main_util.try_flag_bool("bench_use_tpu", True, "Whether to use TPU or GPU/CPU.")
main_util.try_flag_string("bench_imagenet_dir", "gs://cloud-tpu-test-datasets/fake_imagenet", "Path to GCS ImageNet directory. Else using randomly generated fake dataset at 'gs://cloud-tpu-test-datasets/fake_imagenet'")
main_util.try_flag_string("bench_coco_dir", None, "Path to GCS COCO directory. This is required to run a benchmark using object detection")



# Model to benchmark and run
tf.flags.DEFINE_bool("BERT", False, "Run BERT Benchmark tests")
tf.flags.DEFINE_bool("DCGAN", False, "Run DCGAN Benchmark tests")
tf.flags.DEFINE_bool("QaNET", False, "Run QaNET Benchmark tests")
tf.flags.DEFINE_bool("ResNet", False, "Run ResNet Benchmark tests")
tf.flags.DEFINE_bool("RetinaNet", False, "Run RetinaNet Benchmark tests")
tf.flags.DEFINE_bool("QaNET_Small", False, "Run QaNET Benchmark tests with decrease dataset size")
tf.flags.DEFINE_bool("RetinaNet_Small", False, "Run RetinaNet Benchmark tests with decrease dataset size")
tf.flags.DEFINE_bool("ResNet_CIFAR10", False, "Run ResNet Benchmark tests with CIFAR10")


FLAGS = tf.flags.FLAGS




def main(unused_argv):
	tf.flags.mark_flag_as_required("bench_data_location")
	tf.flags.mark_flag_as_required("bench_model_location")
	tf.flags.mark_flag_as_required("bench_tpu_version")

	tf.flags.mark_flag_as_required("bench_tpu_name")
	tf.flags.mark_flag_as_required("bench_tpu_zone")
	tf.flags.mark_flag_as_required("bench_gcp_project")

	BENCH_FLAGS = FLAGS._flags().keys()

	models = []
	if(FLAGS.BERT):
		models.append('BERT')
	if(FLAGS.DCGAN):
		models.append('DCGAN')
	if(FLAGS.QaNET):
		models.append('QaNET')
	if(FLAGS.ResNet):
		models.append('ResNet')
	if(FLAGS.RetinaNet):
		models.append('RetinaNet')

	print("Running models: " + str(models) + " on " + str(FLAGS.bench_tpu_version) )


	if(FLAGS.BERT):
		main_util.run_bert(FLAGS, BENCH_FLAGS)

	if(FLAGS.DCGAN):
		main_util.run_dcgan(FLAGS, BENCH_FLAGS)

	if(FLAGS.QaNET):
		main_util.run_qanet(FLAGS, BENCH_FLAGS)

	if(FLAGS.ResNet):
		main_util.run_resnet(FLAGS, BENCH_FLAGS)

	if(FLAGS.RetinaNet):
		main_util.run_retinanet(FLAGS, BENCH_FLAGS)

	if(FLAGS.QaNET_Small):
		main_util.run_qanet_small(FLAGS, BENCH_FLAGS)

	if(FLAGS.RetinaNet_Small):
		main_util.run_retinanet_small(FLAGS, BENCH_FLAGS)

	if(FLAGS.ResNet_CIFAR10):
		main_util.run_resnet_cifar(FLAGS, BENCH_FLAGS)

if __name__ == "__main__":
	tf.app.run()