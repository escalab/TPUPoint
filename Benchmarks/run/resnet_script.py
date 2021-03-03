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
from imagenet_input import *

class Naive_ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               image_size=224,
               num_parallel_calls=2,
               cache=False,
               dataset_split=None,
               shuffle_shards=False,
               include_background_label=False):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data; if
        'null' (the literal string 'null') or implicitly False then construct a
        null pipeline, consisting of empty images and blank labels.
      image_size: `int` image height and width.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
      dataset_split: If provided, must be one of 'train' or 'validation' and
        specifies the dataset split to read, overriding the default set by
        is_training. In this case, is_training specifies whether the data is
        augmented.
      shuffle_shards: Whether to shuffle the dataset shards.
      include_background_label: Whether to include the background label. If
        this is True, then num_label_classes should be 1001. If False, then
        num_label_classes should be 1000.
    """
    super(Naive_ImageNetInput, self).__init__(
        is_training=is_training,
        image_size=image_size,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input,
        include_background_label=include_background_label)
    self.data_dir = data_dir
    # TODO(b/112427086):  simplify the choice of input source
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache
    self.dataset_split = dataset_split
    self.shuffle_shards = shuffle_shards

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces the
        same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([self.image_size, self.image_size, 3],
                    tf.bfloat16 if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(Naive_ImageNetInput, self).dataset_parser(value)

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      # return tf.data.Dataset.range(1).repeat().map(self._get_null_input)
      dataset = tf.data.Dataset.range(1)
      dataset = dataset.repeat()
      dataset = dataset.map(self._get_null_input, num_parallel_calls=self.num_parallel_calls)
      return dataset

    # Shuffle the filenames to ensure better randomization.
    if not self.dataset_split:
      file_pattern = os.path.join(
          self.data_dir, 'train-*' if self.is_training else 'validation-*')
    else:
      if self.dataset_split not in ['train', 'validation']:
        raise ValueError(
            "If provided, dataset_split must be 'train' or 'validation', was %s"
            % self.dataset_split)
      file_pattern = os.path.join(self.data_dir, self.dataset_split + '-*')

    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.shuffle_shards)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply( tf.contrib.data.parallel_interleave( fetch_dataset, cycle_length=64, sloppy=True))
    # dataset = dataset.apply( tf.contrib.data.parallel_interleave( fetch_dataset, cycle_length=self.num_parallel_calls, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply( tf.contrib.data.shuffle_and_repeat(1024 * 16))
      # dataset = dataset.cache()
      # dataset = dataset.shuffle( 1024 * 16 )
      # dataset = dataset.repeat()
    else:
      dataset = dataset.shuffle(1024)
    return dataset




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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				imagenet_input.ImageNetInput(
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
	resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def resnet_run_eval():
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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				imagenet_input.ImageNetInput(
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

	eval_results = resnet_classifier.evaluate( input_fn=imagenet_eval.input_fn, steps=eval_steps)
	"""	
	if FLAGS.mode == 'eval':

		# Run evaluation when there's a new checkpoint
		for ckpt in evaluation.checkpoints_iterator(
				FLAGS.model_dir, timeout=FLAGS.eval_timeout):
			tf.logging.info('Starting to evaluate.')
			try:
				start_timestamp = time.time()	# This time will include compilation time
				eval_results = resnet_classifier.evaluate(
						input_fn=imagenet_eval.input_fn,
						steps=eval_steps,
						checkpoint_path=ckpt)
				elapsed_time = int(time.time() - start_timestamp)
				tf.logging.info('Eval results: %s. Elapsed seconds: %d',
												eval_results, elapsed_time)

				# Terminate eval job when final checkpoint is reached
				current_step = int(os.path.basename(ckpt).split('-')[1])
				if current_step >= params.train_steps:
					tf.logging.info(
							'Evaluation finished after training step %d', current_step)
					break

			except tf.errors.NotFoundError:
				# Since the coordinator is on a different job than the TPU worker,
				# sometimes the TPU worker does not finish initializing until long after
				# the CPU job tells it to start evaluating. In this case, the checkpoint
				# file could have been deleted already.
				tf.logging.info(
						'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

	else:	 # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
		current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	# pylint: disable=protected-access,line-too-long
		steps_per_epoch = params.num_train_images // params.train_batch_size
		tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
										' step %d.',
										params.train_steps,
										params.train_steps / steps_per_epoch,
										current_step)

		start_timestamp = time.time()	# This time will include compilation time

		if FLAGS.mode == 'train':
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
			resnet_classifier.train(
					input_fn=imagenet_train.input_fn,
					max_steps=params.train_steps,
					hooks=hooks)

		else:
			assert FLAGS.mode == 'train_and_eval'
			while current_step < params.train_steps:
				# Train for up to steps_per_eval number of steps.
				# At the end of training, a checkpoint will be written to --model_dir.
				next_checkpoint = min(current_step + FLAGS.steps_per_eval,
															params.train_steps)
				resnet_classifier.train(
						input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
				current_step = next_checkpoint

				tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
												next_checkpoint, int(time.time() - start_timestamp))

				# Evaluate the model on the most recent model in --model_dir.
				# Since evaluation happens in batches of --eval_batch_size, some images
				# may be excluded modulo the batch size. As long as the batch size is
				# consistent, the evaluated images are also consistent.
				tf.logging.info('Starting to evaluate.')
				eval_results = resnet_classifier.evaluate(
						input_fn=imagenet_eval.input_fn,
						steps=params.num_eval_images // params.eval_batch_size)
				tf.logging.info('Eval results at step %d: %s',
												next_checkpoint, eval_results)

			elapsed_time = int(time.time() - start_timestamp)
			tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
											params.train_steps, elapsed_time)

		if FLAGS.export_dir is not None:
			# The guide to serve a exported TensorFlow model is at:
			#		https://www.tensorflow.org/serving/serving_basic
			tf.logging.info('Starting to export model.')
			export_path = resnet_classifier.export_saved_model(
					export_dir_base=FLAGS.export_dir,
					serving_input_receiver_fn=imagenet_input.image_serving_input_fn)
			if FLAGS.add_warmup_requests:
				inference_warmup.write_warmup_requests(
						export_path,
						FLAGS.model_name,
						params.image_size,
						batch_sizes=FLAGS.inference_batch_sizes,
						image_format='JPEG')
	"""



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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				imagenet_input.ImageNetInput(
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
	resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				imagenet_input.ImageNetInput(
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

	tpupoint.optimize_input_fn(imagenet_train.input_fn, blocking=True)

	bench_start = time.time()
	# tpupoint.Start()
	# resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.train(estimator=resnet_classifier, input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				imagenet_input.ImageNetInput(
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

	tpupoint.optimize_input_fn(imagenet_train.input_fn)

	bench_start = time.time()
	# tpupoint.Start()
	# resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.train_dynamic(model_fn=resnet_model_fn ,estimator=resnet_classifier, input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def resnet_run_naive_wo_tpupoint():
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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				# imagenet_input.ImageNetInput(
				Naive_ImageNetInput(
						is_training=is_training,
						data_dir=FLAGS.data_dir,
						transpose_input=params.transpose_input,
						cache=params.use_cache and is_training,
						image_size=params.image_size,
						num_parallel_calls=2, # params.num_parallel_calls,
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
	resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")


def resnet_run_naive_wo_tpupoint1():
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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				imagenet_input.ImageNetInput(
				# Naive_ImageNetInput(
						is_training=is_training,
						data_dir=FLAGS.data_dir,
						transpose_input=params.transpose_input,
						cache=params.use_cache and is_training,
						image_size=params.image_size,
						num_parallel_calls=2, # params.num_parallel_calls,
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
	tpupoint.optimize_input_fn(imagenet_train.input_fn, blocking=True, worst=True)

	bench_start = time.time()
	
	# resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.train_naive(estimator=resnet_classifier, input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)

	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")




def resnet_run_naive_w_tpupoint():
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

	# Input pipelines are slightly different (with regards to shuffling and
	# preprocessing) between training and evaluation.
	if FLAGS.bigtable_instance:
		tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
		select_train, select_eval = _select_tables_from_flags()
		imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
				is_training=is_training,
				use_bfloat16=use_bfloat16,
				transpose_input=params.transpose_input,
				selection=selection) for (is_training, selection) in
																		 [(True, select_train),
																			(False, select_eval)]]
	else:
		if FLAGS.data_dir == FAKE_DATA_DIR:
			tf.logging.info('Using fake dataset.')
		else:
			tf.logging.info('Using dataset: %s', FLAGS.data_dir)
		imagenet_train, imagenet_eval = [
				# imagenet_input.ImageNetInput(
				Naive_ImageNetInput(
						is_training=is_training,
						data_dir=FLAGS.data_dir,
						transpose_input=params.transpose_input,
						cache=params.use_cache and is_training,
						image_size=params.image_size,
						num_parallel_calls=2, # params.num_parallel_calls,
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
	tpupoint.optimize_input_fn(imagenet_train.input_fn, blocking=False, worst=False)

	bench_start = time.time()
	
	# resnet_classifier.train( input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	# tpupoint.train_naive(estimator=resnet_classifier, input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)
	tpupoint.train_dynamic(model_fn=resnet_model_fn ,estimator=resnet_classifier, input_fn=imagenet_train.input_fn, max_steps=params.train_steps, hooks=hooks)

	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()


