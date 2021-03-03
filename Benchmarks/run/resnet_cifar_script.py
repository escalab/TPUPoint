from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
try:
	from tensorflow.contrib.tpu import TPUPoint
except:
	pass
import time
import os
import sys

sys.path.append( os.path.join( os.getcwd() , '..', 'tpu', 'models', 'official', 'resnet' ) )
sys.path.append( os.path.join( os.getcwd() , '..', 'tpu', 'models' ) )

from resnet_main import *
from cifar10_input import *
import cifar10_input

# flags.DEFINE_string('cifar_train_data_file', '',
#                     'Path to CIFAR10 training data.')
# flags.DEFINE_string('cifar_test_data_file', '', 'Path to CIFAR10 test data.')



def resnet_run_baseline():
	bench_total_start = time.time()
	params = params_dict.ParamsDict(
			resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
	params = params_dict.override_params_dict(
			params, FLAGS.config_file, is_strict=True)
	params = params_dict.override_params_dict(
			params, FLAGS.params_override, is_strict=True)

	params = flags_to_params.override_params_from_input_flags(params, FLAGS)

	params.validate()
	params.lock()

	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	if params.use_async_checkpointing:
		save_checkpoints_steps = None
	else:
		save_checkpoints_steps = max(5000, params.iterations_per_loop)
	config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			save_checkpoints_steps=save_checkpoints_steps,
			log_step_count_steps=FLAGS.log_step_count_steps,
			session_config=tf.ConfigProto(
					graph_options=tf.GraphOptions(
							rewrite_options=rewriter_config_pb2.RewriterConfig(
									disable_meta_optimizer=True))),
			tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=params.iterations_per_loop,
					num_shards=params.num_cores,
					per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
					.PER_HOST_V2))	# pylint: disable=line-too-long

	resnet_classifier = tf.contrib.tpu.TPUEstimator(
			use_tpu=params.use_tpu,
			model_fn=resnet_model_fn,
			config=config,
			params=params.as_dict(),
			train_batch_size=params.train_batch_size,
			eval_batch_size=params.eval_batch_size,
			export_to_tpu=FLAGS.export_to_tpu)

	assert (params.precision == 'bfloat16' or
					params.precision == 'float32'), (
							'Invalid value for precision parameter; '
							'must be bfloat16 or float32.')
	tf.logging.info('Precision: %s', params.precision)
	use_bfloat16 = params.precision == 'bfloat16'


	cifar10_train, cifar10_eval = [
			cifar10_input.CIFAR10Input(
					is_training=is_training,
					data_dir=FLAGS.data_dir,
					transpose_input=params.transpose_input,
					cache=params.use_cache and is_training,
					image_size=params.image_size,
					num_parallel_calls=params.num_parallel_calls,
					include_background_label=(params.num_label_classes == 1001),
					use_bfloat16=use_bfloat16) for is_training in [True, False]
	]

	steps_per_epoch = params.num_train_images // params.train_batch_size
	eval_steps = params.num_eval_images // params.eval_batch_size

	hooks = []
	if params.use_async_checkpointing:
		hooks.append(
				async_checkpoint.AsyncCheckpointSaverHook(
						checkpoint_dir=FLAGS.model_dir,
						save_steps=max(5000, params.iterations_per_loop)))
	if FLAGS.profile_every_n_steps > 0:
		hooks.append(
				tpu_profiler_hook.TPUProfilerHook(
						save_steps=FLAGS.profile_every_n_steps,
						output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
				)
	bench_start = time.time()
	resnet_classifier.train( input_fn=cifar10_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")




def resnet_run_eval():
	print("Not Implemented")
	return



def resnet_run_profile():
	bench_total_start = time.time()
	params = params_dict.ParamsDict(
			resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
	params = params_dict.override_params_dict(
			params, FLAGS.config_file, is_strict=True)
	params = params_dict.override_params_dict(
			params, FLAGS.params_override, is_strict=True)

	params = flags_to_params.override_params_from_input_flags(params, FLAGS)

	params.validate()
	params.lock()

	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	if params.use_async_checkpointing:
		save_checkpoints_steps = None
	else:
		save_checkpoints_steps = max(5000, params.iterations_per_loop)
	config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			save_checkpoints_steps=save_checkpoints_steps,
			log_step_count_steps=FLAGS.log_step_count_steps,
			session_config=tf.ConfigProto(
					graph_options=tf.GraphOptions(
							rewrite_options=rewriter_config_pb2.RewriterConfig(
									disable_meta_optimizer=True))),
			tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=params.iterations_per_loop,
					num_shards=params.num_cores,
					per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
					.PER_HOST_V2))	# pylint: disable=line-too-long

	resnet_classifier = tf.contrib.tpu.TPUEstimator(
			use_tpu=params.use_tpu,
			model_fn=resnet_model_fn,
			config=config,
			params=params.as_dict(),
			train_batch_size=params.train_batch_size,
			eval_batch_size=params.eval_batch_size,
			export_to_tpu=FLAGS.export_to_tpu)

	assert (params.precision == 'bfloat16' or
					params.precision == 'float32'), (
							'Invalid value for precision parameter; '
							'must be bfloat16 or float32.')
	tf.logging.info('Precision: %s', params.precision)
	use_bfloat16 = params.precision == 'bfloat16'


	cifar10_train, cifar10_eval = [
			cifar10_input.CIFAR10Input(
					is_training=is_training,
					data_dir=FLAGS.data_dir,
					transpose_input=params.transpose_input,
					cache=params.use_cache and is_training,
					image_size=params.image_size,
					num_parallel_calls=params.num_parallel_calls,
					include_background_label=(params.num_label_classes == 1001),
					use_bfloat16=use_bfloat16) for is_training in [True, False]
	]

	steps_per_epoch = params.num_train_images // params.train_batch_size
	eval_steps = params.num_eval_images // params.eval_batch_size

	hooks = []
	if params.use_async_checkpointing:
		hooks.append(
				async_checkpoint.AsyncCheckpointSaverHook(
						checkpoint_dir=FLAGS.model_dir,
						save_steps=max(5000, params.iterations_per_loop)))
	if FLAGS.profile_every_n_steps > 0:
		hooks.append(
				tpu_profiler_hook.TPUProfilerHook(
						save_steps=FLAGS.profile_every_n_steps,
						output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
				)
	
	tpupoint = TPUPoint( 
		estimator = resnet_classifier,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	bench_start = time.time()
	tpupoint.Start()
	resnet_classifier.train( input_fn=cifar10_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")


def resnet_run_optimize():
	bench_total_start = time.time()
	params = params_dict.ParamsDict(
			resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
	params = params_dict.override_params_dict(
			params, FLAGS.config_file, is_strict=True)
	params = params_dict.override_params_dict(
			params, FLAGS.params_override, is_strict=True)

	params = flags_to_params.override_params_from_input_flags(params, FLAGS)

	params.validate()
	params.lock()

	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	if params.use_async_checkpointing:
		save_checkpoints_steps = None
	else:
		save_checkpoints_steps = max(5000, params.iterations_per_loop)
	config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			save_checkpoints_steps=save_checkpoints_steps,
			log_step_count_steps=FLAGS.log_step_count_steps,
			session_config=tf.ConfigProto(
					graph_options=tf.GraphOptions(
							rewrite_options=rewriter_config_pb2.RewriterConfig(
									disable_meta_optimizer=True))),
			tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=params.iterations_per_loop,
					num_shards=params.num_cores,
					per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
					.PER_HOST_V2))	# pylint: disable=line-too-long

	resnet_classifier = tf.contrib.tpu.TPUEstimator(
			use_tpu=params.use_tpu,
			model_fn=resnet_model_fn,
			config=config,
			params=params.as_dict(),
			train_batch_size=params.train_batch_size,
			eval_batch_size=params.eval_batch_size,
			export_to_tpu=FLAGS.export_to_tpu)

	assert (params.precision == 'bfloat16' or
					params.precision == 'float32'), (
							'Invalid value for precision parameter; '
							'must be bfloat16 or float32.')
	tf.logging.info('Precision: %s', params.precision)
	use_bfloat16 = params.precision == 'bfloat16'


	cifar10_train, cifar10_eval = [
			cifar10_input.CIFAR10Input(
					is_training=is_training,
					data_dir=FLAGS.data_dir,
					transpose_input=params.transpose_input,
					cache=params.use_cache and is_training,
					image_size=params.image_size,
					num_parallel_calls=params.num_parallel_calls,
					include_background_label=(params.num_label_classes == 1001),
					use_bfloat16=use_bfloat16) for is_training in [True, False]
	]

	steps_per_epoch = params.num_train_images // params.train_batch_size
	eval_steps = params.num_eval_images // params.eval_batch_size

	hooks = []
	if params.use_async_checkpointing:
		hooks.append(
				async_checkpoint.AsyncCheckpointSaverHook(
						checkpoint_dir=FLAGS.model_dir,
						save_steps=max(5000, params.iterations_per_loop)))
	if FLAGS.profile_every_n_steps > 0:
		hooks.append(
				tpu_profiler_hook.TPUProfilerHook(
						save_steps=FLAGS.profile_every_n_steps,
						output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
				)
	
	tpupoint = TPUPoint( 
		estimator = resnet_classifier,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	input_fn = tpupoint.optimize_input_fn(cifar10_train.input_fn, blocking=True)

	bench_start = time.time()
	tpupoint.train(estimator=resnet_classifier, input_fn=input_fn, max_steps=params.train_steps, hooks=hooks)
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def resnet_run_dynamic():
	bench_total_start = time.time()
	params = params_dict.ParamsDict(
			resnet_config.RESNET_CFG, resnet_config.RESNET_RESTRICTIONS)
	params = params_dict.override_params_dict(
			params, FLAGS.config_file, is_strict=True)
	params = params_dict.override_params_dict(
			params, FLAGS.params_override, is_strict=True)

	params = flags_to_params.override_params_from_input_flags(params, FLAGS)

	params.validate()
	params.lock()

	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	if params.use_async_checkpointing:
		save_checkpoints_steps = None
	else:
		save_checkpoints_steps = max(5000, params.iterations_per_loop)
	config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			save_checkpoints_steps=save_checkpoints_steps,
			log_step_count_steps=FLAGS.log_step_count_steps,
			session_config=tf.ConfigProto(
					graph_options=tf.GraphOptions(
							rewrite_options=rewriter_config_pb2.RewriterConfig(
									disable_meta_optimizer=True))),
			tpu_config=tf.contrib.tpu.TPUConfig(
					iterations_per_loop=params.iterations_per_loop,
					num_shards=params.num_cores,
					per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
					.PER_HOST_V2))	# pylint: disable=line-too-long

	resnet_classifier = tf.contrib.tpu.TPUEstimator(
			use_tpu=params.use_tpu,
			model_fn=resnet_model_fn,
			config=config,
			params=params.as_dict(),
			train_batch_size=params.train_batch_size,
			eval_batch_size=params.eval_batch_size,
			export_to_tpu=FLAGS.export_to_tpu)

	assert (params.precision == 'bfloat16' or
					params.precision == 'float32'), (
							'Invalid value for precision parameter; '
							'must be bfloat16 or float32.')
	tf.logging.info('Precision: %s', params.precision)
	use_bfloat16 = params.precision == 'bfloat16'


	cifar10_train, cifar10_eval = [
			cifar10_input.CIFAR10Input(
					is_training=is_training,
					data_dir=FLAGS.data_dir,
					transpose_input=params.transpose_input,
					cache=params.use_cache and is_training,
					image_size=params.image_size,
					num_parallel_calls=params.num_parallel_calls,
					include_background_label=(params.num_label_classes == 1001),
					use_bfloat16=use_bfloat16) for is_training in [True, False]
	]

	steps_per_epoch = params.num_train_images // params.train_batch_size
	eval_steps = params.num_eval_images // params.eval_batch_size

	hooks = []
	if params.use_async_checkpointing:
		hooks.append(
				async_checkpoint.AsyncCheckpointSaverHook(
						checkpoint_dir=FLAGS.model_dir,
						save_steps=max(5000, params.iterations_per_loop)))
	if FLAGS.profile_every_n_steps > 0:
		hooks.append(
				tpu_profiler_hook.TPUProfilerHook(
						save_steps=FLAGS.profile_every_n_steps,
						output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
				)
	
	tpupoint = TPUPoint( 
		estimator = resnet_classifier,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	bench_start = time.time()
	tpupoint.train_dynamic(model_fn=resnet_model_fn , estimator=resnet_classifier, input_fn=cifar10_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def resnet_run_naive_wo_tpupoint():
	print("Not Implemented")
	return


def resnet_run_naive_wo_tpupoint1():
	print("Not Implemented")
	return



def resnet_run_naive_w_tpupoint():
	print("Not Implemented")
	return
