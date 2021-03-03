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

sys.path.append( os.path.join( os.getcwd() , '..', 'tpu', 'models', 'experimental', 'qanet' ) )

from run import *
from run_lib import *


def new_build_dataset(cfg, is_tpu):
  """Construct train and eval inputs_fn."""
  load_tfrecord = cfg.load_tfrecord
  if is_tpu:
    load_tfrecord = True
  # TODO(ddohan): Share the common args more clearly
  train_input = new_get_input_fn(
      split=cfg.train_split,
      max_length=cfg.max_length,
      # TPUs don't handle OutOfRange exceptions from data pipelines, so we
      # repeat indefinitely and handle setting number of training steps
      # manually. This is handled by the tpu.steps_per_epoch setting.
      # On a GPU, we are able to be more exact about the exact boundary between
      # epochs and avoid reasoning in terms of step counts.
      # If 0, repeat indefinitely. Otherwise repeat N times.
      num_repeats=0 if is_tpu else cfg.num_repeats,
      shuffle=cfg.train_shuffle,
      cache=cfg.cache,
      limit=None,
      data_path=cfg.data_path,
      vocab_path=cfg.vocab_path,
      is_tpu=is_tpu,
      use_generator=not load_tfrecord,
      resample_too_long=cfg.resample_too_long,
      is_training=True)
  eval_input = new_get_input_fn(
      split=cfg.eval_split,
      max_length=None,  # Never do any filtering at eval
      limit=None,
      num_repeats=1,
      shuffle=False,
      cache=cfg.cache,
      data_path=cfg.data_path,
      vocab_path=cfg.vocab_path,
      is_tpu=False, # Never eval on TPU because of py_func
      use_generator=not load_tfrecord,
      is_training=False)
  return train_input, eval_input

def new_get_input_fn(split='dev', shuffle=False, num_repeats=False, limit=None, do_embedding=True, cache=True, max_length=None, resample_too_long=True, data_path=None, vocab_path=None, is_tpu=False, use_generator=True, is_training=False):
  """Build input function."""
  if is_tpu:
    assert max_length

  # Do the GLOVE embedding lookups in the data loader
  if do_embedding:
    # Load and package into the graph directly
    # Vocab is about ~200MB total once filtered down
    embedding_words, embedding_vectors = data.get_pretrained_embeddings_cache(
        embeddings_path=vocab_path)

  def _input_fn(params=None):
    """Input function compatible with `Experiment` object.

    Args:
      params: Params passed to the estimator. Contains 'batch_size'.

    Returns:
      A tuple of feature tensors and target tensors.

    Raises:
      ValueError: If filtering by length is set during eval mode.
    """
    if not is_training:
      assert not is_tpu
    tf.logging.info('Data pipeline given params:\n%s' % params)
    if is_training:
      batch_size = params.dataset.train_batch_size
    else:
      batch_size = params.dataset.eval_batch_size

    if use_generator:
      tf.logging.info('Building generator data pipeline.')
      ds = data.build_generator_pipeline(
          data_path=data_path,
          split=split,
          tokenizer_fn=word_tokenize)
    else:
      tf.logging.info('Loading TFRecords from %s' % data_path)
      filenames = tf.gfile.Glob(os.path.join(data_path, '%s_*' % split))
      tf.logging.info(filenames)
      ds = data.build_tfrecord_pipeline(filenames=filenames)

    if max_length:
      if not is_training:
        raise ValueError('Unable to filter or resample examples at eval time.')
      if resample_too_long:

        tf.logging.info('Resampling with max length %s', max_length)
        def _resample(x):
          return data.resample_example(x, max_length=max_length)

        ds = ds.map(_resample, num_parallel_calls=16)
      else:
        # Filter out examples over our max length to avoid an error downstream.
        tf.logging.info('Filtering out examples over max length %s', max_length)
        def _not_too_long(x):
          return tf.greater_equal(
              tf.to_int32(max_length), tf.to_int32(x['context_length']))

        ds = ds.filter(_not_too_long)

    if limit:
      # Take the first N examples
      ds = ds.take(limit)

    if cache:
      # Cache dataset to avoid hitting the python generator after first epoch
      ds = ds.cache()

    # Subset that we should actually pass back to the caller
    # This is required to filter out tf.string fields which are not TPU
    # compatible
    # Specifically: id, context, question, context_tokens and question_tokens
    # are all string fields that will be removed.
    shapes, _ = data.get_shapes_and_types(is_tpu=is_tpu, max_length=max_length)

    if do_embedding:
      # Embed tokens with pretrained word vectors

      # Add in shape info before batching
      shapes['context_vecs'] = [max_length if is_tpu else None, 300]
      shapes['question_vecs'] = [max_length if is_tpu else None, 300]

      vocab_table = tf.contrib.lookup.index_table_from_tensor(
          embedding_words, default_value=0)
      vocab_vectors = tf.constant(embedding_vectors, dtype=tf.float32)

      def lookup(words):
        ids = vocab_table.lookup(words)
        result = tf.nn.embedding_lookup(params=vocab_vectors, ids=ids)
        return result

      def lookup_fields(d):
        d['context_vecs'] = lookup(d['context_tokens'])
        d['question_vecs'] = lookup(d['question_tokens'])
        return d

      ds = ds.map(lookup_fields, num_parallel_calls=16)

    buffer_size = 5000  # Magic number TUNE ME
    repeats = num_repeats if num_repeats else None
    if shuffle and repeats != 1:
      tf.logging.info('Shuffle and repeat size: %s' % buffer_size)
      ds = ds.apply(
          tf.contrib.data.shuffle_and_repeat(
              buffer_size=buffer_size,
              count=repeats))
    elif repeats != 1:
      tf.logging.info('Repeating')
      ds = ds.repeat(count=repeats)
    elif shuffle:
      tf.logging.info('Shuffle size: %s' % buffer_size)
      ds = ds.shuffle(buffer_size=buffer_size)

    def filter_fields(example):
      out = {}
      for k in shapes:
        out[k] = example[k]
      return out

    ds = ds.map(filter_fields, num_parallel_calls=16)

    if is_training:
      ds = ds.apply(
          tf.contrib.data.padded_batch_and_drop_remainder(
              batch_size, padded_shapes=shapes))
    else:
      # Never want to ignore values at eval time
      ds = ds.padded_batch(batch_size, padded_shapes=shapes)
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)  # Buffer a few batches ahead
    # if do_embedding:
    #  iterator = ds.make_initializable_iterator()
    #  # Must be initialized when the graph is initialized and before the
    #  # dataset tensors are evaluated.
    #  # Run `tf.tables_initializer()` before getting first batch
    #  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
    #            iterator.initializer)
    # else:
    #  iterator = ds.make_one_shot_iterator()
    # batch = iterator.get_next()
    # return batch, batch

    def MapFn(batch):
      return batch, batch
    ds = ds.map(MapFn)
    if do_embedding:
    	iterator = ds.make_initializable_iterator()
    	tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return ds 
  
  return _input_fn


def Naive_build_tfrecord_pipeline(filenames):
  """Read TFRecords from disk to create data pipeline."""
  sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.int64, allow_missing=True)
  str_sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.string, allow_missing=True)
  int_feature = tf.FixedLenFeature([], tf.int64)
  str_feature = tf.FixedLenFeature([], tf.string)
  features = {
      'id': str_feature,
      'num_answers': int_feature,
      'answers': str_sequence_feature,
      'answers_start_token': sequence_feature,
      'answers_end_token': sequence_feature,
      'context': str_feature,
      'context_length': int_feature,
      'context_tokens': str_sequence_feature,
      'question': str_feature,
      'question_length': int_feature,
      'question_tokens': str_sequence_feature,
  }

  def _parse(proto):
    return tf.parse_single_example(proto, features=features)

  ds = tf.data.TFRecordDataset(
      filenames,
      # 1 GB
      buffer_size=1024 * 1024 * 1024,
      num_parallel_reads=8)

  ds = ds.map(_parse, num_parallel_calls=1)
  return ds

def Naive_new_build_dataset(cfg, is_tpu):
  """Construct train and eval inputs_fn."""
  load_tfrecord = cfg.load_tfrecord
  if is_tpu:
    load_tfrecord = True
  # TODO(ddohan): Share the common args more clearly
  train_input = Naive_new_get_input_fn(
      split=cfg.train_split,
      max_length=cfg.max_length,
      # TPUs don't handle OutOfRange exceptions from data pipelines, so we
      # repeat indefinitely and handle setting number of training steps
      # manually. This is handled by the tpu.steps_per_epoch setting.
      # On a GPU, we are able to be more exact about the exact boundary between
      # epochs and avoid reasoning in terms of step counts.
      # If 0, repeat indefinitely. Otherwise repeat N times.
      num_repeats=0 if is_tpu else cfg.num_repeats,
      shuffle=cfg.train_shuffle,
      cache=cfg.cache,
      limit=None,
      data_path=cfg.data_path,
      vocab_path=cfg.vocab_path,
      is_tpu=is_tpu,
      use_generator=not load_tfrecord,
      resample_too_long=cfg.resample_too_long,
      is_training=True)
  eval_input = Naive_new_get_input_fn(
      split=cfg.eval_split,
      max_length=None,  # Never do any filtering at eval
      limit=None,
      num_repeats=1,
      shuffle=False,
      cache=cfg.cache,
      data_path=cfg.data_path,
      vocab_path=cfg.vocab_path,
      is_tpu=False, # Never eval on TPU because of py_func
      use_generator=not load_tfrecord,
      is_training=False)
  return train_input, eval_input

def Naive_new_get_input_fn(split='dev', shuffle=False, num_repeats=False, limit=None, do_embedding=True, cache=True, max_length=None, resample_too_long=True, data_path=None, vocab_path=None, is_tpu=False, use_generator=True, is_training=False):
  """Build input function."""
  if is_tpu:
    assert max_length

  # Do the GLOVE embedding lookups in the data loader
  if do_embedding:
    # Load and package into the graph directly
    # Vocab is about ~200MB total once filtered down
    embedding_words, embedding_vectors = data.get_pretrained_embeddings_cache(
        embeddings_path=vocab_path)

  def _input_fn(params=None):
    """Input function compatible with `Experiment` object.

    Args:
      params: Params passed to the estimator. Contains 'batch_size'.

    Returns:
      A tuple of feature tensors and target tensors.

    Raises:
      ValueError: If filtering by length is set during eval mode.
    """
    if not is_training:
      assert not is_tpu
    tf.logging.info('Data pipeline given params:\n%s' % params)
    if is_training:
      batch_size = params.dataset.train_batch_size
    else:
      batch_size = params.dataset.eval_batch_size

    if use_generator:
      tf.logging.info('Building generator data pipeline.')
      ds = data.build_generator_pipeline(
          data_path=data_path,
          split=split,
          tokenizer_fn=word_tokenize)
    else:
      tf.logging.info('Loading TFRecords from %s' % data_path)
      filenames = tf.gfile.Glob(os.path.join(data_path, '%s_*' % split))
      tf.logging.info(filenames)
      # ds = data.build_tfrecord_pipeline(filenames=filenames)
      ds = Naive_build_tfrecord_pipeline(filenames=filenames)


    if max_length:
      if not is_training:
        raise ValueError('Unable to filter or resample examples at eval time.')
      if resample_too_long:

        tf.logging.info('Resampling with max length %s', max_length)
        def _resample(x):
          return data.resample_example(x, max_length=max_length)

        # ds = ds.map(_resample, num_parallel_calls=16)
        ds = ds.map(_resample, num_parallel_calls=1)
      else:
        # Filter out examples over our max length to avoid an error downstream.
        tf.logging.info('Filtering out examples over max length %s', max_length)
        def _not_too_long(x):
          return tf.greater_equal(
              tf.to_int32(max_length), tf.to_int32(x['context_length']))

        ds = ds.filter(_not_too_long)

    if limit:
      # Take the first N examples
      ds = ds.take(limit)

    if cache:
      # Cache dataset to avoid hitting the python generator after first epoch
      ds = ds.cache()

    # Subset that we should actually pass back to the caller
    # This is required to filter out tf.string fields which are not TPU
    # compatible
    # Specifically: id, context, question, context_tokens and question_tokens
    # are all string fields that will be removed.
    shapes, _ = data.get_shapes_and_types(is_tpu=is_tpu, max_length=max_length)

    if do_embedding:
      # Embed tokens with pretrained word vectors

      # Add in shape info before batching
      shapes['context_vecs'] = [max_length if is_tpu else None, 300]
      shapes['question_vecs'] = [max_length if is_tpu else None, 300]

      vocab_table = tf.contrib.lookup.index_table_from_tensor(
          embedding_words, default_value=0)
      vocab_vectors = tf.constant(embedding_vectors, dtype=tf.float32)

      def lookup(words):
        ids = vocab_table.lookup(words)
        result = tf.nn.embedding_lookup(params=vocab_vectors, ids=ids)
        return result

      def lookup_fields(d):
        d['context_vecs'] = lookup(d['context_tokens'])
        d['question_vecs'] = lookup(d['question_tokens'])
        return d

      ds = ds.map(lookup_fields, num_parallel_calls=16)

    buffer_size = 5000  # Magic number TUNE ME
    repeats = num_repeats if num_repeats else None
    if shuffle and repeats != 1:
      tf.logging.info('Shuffle and repeat size: %s' % buffer_size)
      # ds = ds.apply(
      #     tf.contrib.data.shuffle_and_repeat(
      #         buffer_size=buffer_size,
      #         count=repeats))
      ds = ds.shuffle(buffer_size)
      ds = ds.repeat(repeats)

    elif repeats != 1:
      tf.logging.info('Repeating')
      ds = ds.repeat(count=repeats)
    elif shuffle:
      tf.logging.info('Shuffle size: %s' % buffer_size)
      ds = ds.shuffle(buffer_size=buffer_size)

    def filter_fields(example):
      out = {}
      for k in shapes:
        out[k] = example[k]
      return out

    # ds = ds.map(filter_fields, num_parallel_calls=16)
    ds = ds.map(filter_fields, num_parallel_calls=1)

    if is_training:
      ds = ds.apply(
          tf.contrib.data.padded_batch_and_drop_remainder(
              batch_size, padded_shapes=shapes))
    else:
      # Never want to ignore values at eval time
      ds = ds.padded_batch(batch_size, padded_shapes=shapes)
    # ds = ds.prefetch(tf.contrib.data.AUTOTUNE)  # Buffer a few batches ahead
    ds = ds.prefetch(1)  # Buffer a few batches ahead
    # if do_embedding:
    #  iterator = ds.make_initializable_iterator()
    #  # Must be initialized when the graph is initialized and before the
    #  # dataset tensors are evaluated.
    #  # Run `tf.tables_initializer()` before getting first batch
    #  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
    #            iterator.initializer)
    # else:
    #  iterator = ds.make_one_shot_iterator()
    # batch = iterator.get_next()
    # return batch, batch

    def MapFn(batch):
      return batch, batch
    # ds = ds.map(MapFn)
    ds = ds.map(MapFn, num_parallel_calls=1)
    if do_embedding:
    	iterator = ds.make_initializable_iterator()
    	tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    return ds 
  
  return _input_fn



def qanet_run_baseline():
	bench_total_start = time.time()
	# tf.logging.set_verbosity(tf.logging.INFO)
	cfg = create_config(model_dir=FLAGS.model_dir)

	if FLAGS.tpu:
		cfg.tpu.name = FLAGS.tpu
		cfg.tpu.zone = FLAGS.tpu_zone
		cfg.tpu.gcp_project = FLAGS.gcp_project
		cfg.tpu.enable = True
	else:
		# Toggle TPU relevant settings
		if FLAGS.enable_tpu:
			cfg.tpu.enable = True
		else:
			cfg.tpu.enable = False
	# train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

	tf.logging.info("cfg.model_dir = " + cfg.model_dir)
	# Save out config to model directory
	# assert "train" in FLAGS.mode
	tf.gfile.MakeDirs(cfg.model_dir)
	with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
		json.dump(cfg, f)

	if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
		raise ValueError("Must have a fixed num repeats or epoch step size.")

	# Construct inputs and estimator
	# train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	estimator = model.get_estimator(**cfg)

	# if do_eval:
	# 	eval_metrics = None
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch or None)
	# 		tf.logging.info(pprint.pformat(train_metrics))
	# 		eval_metrics = estimator.evaluate(input_fn=eval_input)
	# 		tf.logging.info(pprint.pformat(eval_metrics))
	# 		if report_fn:
	# 			report_fn(eval_metrics)
	# 	return eval_metrics
	# else:
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch)
	# 		tf.logging.info(pprint.pformat(train_metrics))

	tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
	bench_start = time.time()
	train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info(pprint.pformat(train_metrics))
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def qanet_run_eval():
	# tf.logging.set_verbosity(tf.logging.INFO)
	cfg = create_config(model_dir=FLAGS.model_dir)

	if FLAGS.tpu:
		cfg.tpu.name = FLAGS.tpu
		cfg.tpu.zone = FLAGS.tpu_zone
		cfg.tpu.gcp_project = FLAGS.gcp_project
		cfg.tpu.enable = True
	else:
		# Toggle TPU relevant settings
		if FLAGS.enable_tpu:
			cfg.tpu.enable = True
		else:
			cfg.tpu.enable = False
	# train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))
	eval_metrics = evaluate(override_cfg=cfg, model_dir=FLAGS.model_dir, continuous=False)
	tf.logging.info("eval_metrics: " + str(eval_metrics))



def qanet_run_profile():
	bench_total_start = time.time()
	# tf.logging.set_verbosity(tf.logging.INFO)
	cfg = create_config(model_dir=FLAGS.model_dir)

	if FLAGS.tpu:
		cfg.tpu.name = FLAGS.tpu
		cfg.tpu.zone = FLAGS.tpu_zone
		cfg.tpu.gcp_project = FLAGS.gcp_project
		cfg.tpu.enable = True
	else:
		# Toggle TPU relevant settings
		if FLAGS.enable_tpu:
			cfg.tpu.enable = True
		else:
			cfg.tpu.enable = False
	# train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

	tf.logging.info("cfg.model_dir = " + cfg.model_dir)
	# Save out config to model directory
	# assert "train" in FLAGS.mode
	tf.gfile.MakeDirs(cfg.model_dir)
	with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
		json.dump(cfg, f)

	if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
		raise ValueError("Must have a fixed num repeats or epoch step size.")

	# Construct inputs and estimator
	# train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	estimator = model.get_estimator(**cfg)

	# if do_eval:
	# 	eval_metrics = None
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch or None)
	# 		tf.logging.info(pprint.pformat(train_metrics))
	# 		eval_metrics = estimator.evaluate(input_fn=eval_input)
	# 		tf.logging.info(pprint.pformat(eval_metrics))
	# 		if report_fn:
	# 			report_fn(eval_metrics)
	# 	return eval_metrics
	# else:
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch)
	# 		tf.logging.info(pprint.pformat(train_metrics))

	tpupoint = TPUPoint( 
		estimator = estimator,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
	bench_start = time.time()
	tpupoint.Start()
	train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info(pprint.pformat(train_metrics))
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def qanet_run_optimize():
	bench_total_start = time.time()
	# tf.logging.set_verbosity(tf.logging.INFO)
	cfg = create_config(model_dir=FLAGS.model_dir)

	if FLAGS.tpu:
		cfg.tpu.name = FLAGS.tpu
		cfg.tpu.zone = FLAGS.tpu_zone
		cfg.tpu.gcp_project = FLAGS.gcp_project
		cfg.tpu.enable = True
	else:
		# Toggle TPU relevant settings
		if FLAGS.enable_tpu:
			cfg.tpu.enable = True
		else:
			cfg.tpu.enable = False
	# train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

	tf.logging.info("cfg.model_dir = " + cfg.model_dir)
	# Save out config to model directory
	# assert "train" in FLAGS.mode
	tf.gfile.MakeDirs(cfg.model_dir)
	with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
		json.dump(cfg, f)

	if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
		raise ValueError("Must have a fixed num repeats or epoch step size.")

	# Construct inputs and estimator
	# train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	estimator = model.get_estimator(**cfg)

	# if do_eval:
	# 	eval_metrics = None
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch or None)
	# 		tf.logging.info(pprint.pformat(train_metrics))
	# 		eval_metrics = estimator.evaluate(input_fn=eval_input)
	# 		tf.logging.info(pprint.pformat(eval_metrics))
	# 		if report_fn:
	# 			report_fn(eval_metrics)
	# 	return eval_metrics
	# else:
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch)
	# 		tf.logging.info(pprint.pformat(train_metrics))

	tpupoint = TPUPoint( 
		estimator = estimator,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tpupoint.optimize_input_fn(train_input, blocking=True)

	tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
	bench_start = time.time()
	# tpupoint.Start()
	# train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	tpupoint.train(estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	# tf.logging.info(pprint.pformat(train_metrics))
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def qanet_run_dynamic():
	bench_total_start = time.time()
	# tf.logging.set_verbosity(tf.logging.INFO)
	cfg = create_config(model_dir=FLAGS.model_dir)

	if FLAGS.tpu:
		cfg.tpu.name = FLAGS.tpu
		cfg.tpu.zone = FLAGS.tpu_zone
		cfg.tpu.gcp_project = FLAGS.gcp_project
		cfg.tpu.enable = True
	else:
		# Toggle TPU relevant settings
		if FLAGS.enable_tpu:
			cfg.tpu.enable = True
		else:
			cfg.tpu.enable = False
	# train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

	tf.logging.info("cfg.model_dir = " + cfg.model_dir)
	# Save out config to model directory
	# assert "train" in FLAGS.mode
	tf.gfile.MakeDirs(cfg.model_dir)
	with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
		json.dump(cfg, f)

	if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
		raise ValueError("Must have a fixed num repeats or epoch step size.")

	# Construct inputs and estimator
	# train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	estimator = model.get_estimator(**cfg)

	# if do_eval:
	# 	eval_metrics = None
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch or None)
	# 		tf.logging.info(pprint.pformat(train_metrics))
	# 		eval_metrics = estimator.evaluate(input_fn=eval_input)
	# 		tf.logging.info(pprint.pformat(eval_metrics))
	# 		if report_fn:
	# 			report_fn(eval_metrics)
	# 	return eval_metrics
	# else:
	# 	for i in range(cfg.num_epochs):
	# 		tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
	# 		train_metrics = estimator.train(
	# 				input_fn=train_input, steps=cfg.steps_per_epoch)
	# 		tf.logging.info(pprint.pformat(train_metrics))

	tpupoint = TPUPoint( 
		estimator = estimator,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tpupoint.optimize_input_fn(train_input)

	tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
	bench_start = time.time()
	# tpupoint.Start()
	# train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	tpupoint.train_dynamic(model_fn=model.model_fn , estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	# tf.logging.info(pprint.pformat(train_metrics))
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def qanet_run_squad_naive_wo_tpupoint():
	bench_total_start = time.time()
	# tf.logging.set_verbosity(tf.logging.INFO)
	cfg = create_config(model_dir=FLAGS.model_dir)

	if FLAGS.tpu:
		cfg.tpu.name = FLAGS.tpu
		cfg.tpu.zone = FLAGS.tpu_zone
		cfg.tpu.gcp_project = FLAGS.gcp_project
		cfg.tpu.enable = True
	else:
		# Toggle TPU relevant settings
		if FLAGS.enable_tpu:
			cfg.tpu.enable = True
		else:
			cfg.tpu.enable = False
	# train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

	tf.logging.info("cfg.model_dir = " + cfg.model_dir)
	# Save out config to model directory
	# assert "train" in FLAGS.mode
	tf.gfile.MakeDirs(cfg.model_dir)
	with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
		json.dump(cfg, f)

	if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
		raise ValueError("Must have a fixed num repeats or epoch step size.")

	# Construct inputs and estimator
	# train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	# train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	train_input, eval_input = Naive_new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
	estimator = model.get_estimator(**cfg)
	tpupoint = TPUPoint( 
		estimator = estimator,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 
	tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
	bench_start = time.time()
	tpupoint.Start()
	train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info(pprint.pformat(train_metrics))
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def qanet_run_squad_naive_wo_tpupoint1():
  bench_total_start = time.time()
  # tf.logging.set_verbosity(tf.logging.INFO)
  cfg = create_config(model_dir=FLAGS.model_dir)

  if FLAGS.tpu:
    cfg.tpu.name = FLAGS.tpu
    cfg.tpu.zone = FLAGS.tpu_zone
    cfg.tpu.gcp_project = FLAGS.gcp_project
    cfg.tpu.enable = True
  else:
    # Toggle TPU relevant settings
    if FLAGS.enable_tpu:
      cfg.tpu.enable = True
    else:
      cfg.tpu.enable = False
  # train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

  tf.logging.info("cfg.model_dir = " + cfg.model_dir)
  # Save out config to model directory
  # assert "train" in FLAGS.mode
  tf.gfile.MakeDirs(cfg.model_dir)
  with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
    json.dump(cfg, f)

  if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
    raise ValueError("Must have a fixed num repeats or epoch step size.")

  # Construct inputs and estimator
  # train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
  train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
  estimator = model.get_estimator(**cfg)

  # if do_eval:
  #   eval_metrics = None
  #   for i in range(cfg.num_epochs):
  #     tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
  #     train_metrics = estimator.train(
  #         input_fn=train_input, steps=cfg.steps_per_epoch or None)
  #     tf.logging.info(pprint.pformat(train_metrics))
  #     eval_metrics = estimator.evaluate(input_fn=eval_input)
  #     tf.logging.info(pprint.pformat(eval_metrics))
  #     if report_fn:
  #       report_fn(eval_metrics)
  #   return eval_metrics
  # else:
  #   for i in range(cfg.num_epochs):
  #     tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
  #     train_metrics = estimator.train(
  #         input_fn=train_input, steps=cfg.steps_per_epoch)
  #     tf.logging.info(pprint.pformat(train_metrics))

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu,
    logdir=FLAGS.model_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  tpupoint.optimize_input_fn(train_input, blocking=True, worst=True)

  tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
  bench_start = time.time()
  # tpupoint.Start()
  # train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  tpupoint.train_naive(estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  # tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  # tf.logging.info(pprint.pformat(train_metrics))
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()



def qanet_run_squad_naive_w_tpupoint():
  bench_total_start = time.time()
  # tf.logging.set_verbosity(tf.logging.INFO)
  cfg = create_config(model_dir=FLAGS.model_dir)

  if FLAGS.tpu:
    cfg.tpu.name = FLAGS.tpu
    cfg.tpu.zone = FLAGS.tpu_zone
    cfg.tpu.gcp_project = FLAGS.gcp_project
    cfg.tpu.enable = True
  else:
    # Toggle TPU relevant settings
    if FLAGS.enable_tpu:
      cfg.tpu.enable = True
    else:
      cfg.tpu.enable = False
  # train_and_eval(cfg, do_eval=("eval" in FLAGS.mode))

  tf.logging.info("cfg.model_dir = " + cfg.model_dir)
  # Save out config to model directory
  # assert "train" in FLAGS.mode
  tf.gfile.MakeDirs(cfg.model_dir)
  with tf.gfile.GFile(os.path.join(cfg.model_dir, "config.json"), "w") as f:
    json.dump(cfg, f)

  if not cfg.dataset.num_repeats and not cfg.steps_per_epoch:
    raise ValueError("Must have a fixed num repeats or epoch step size.")

  # Construct inputs and estimator
  # train_input, eval_input = data.build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
  # train_input, eval_input = new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
  train_input, eval_input = Naive_new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
  estimator = model.get_estimator(**cfg)

  # if do_eval:
  #   eval_metrics = None
  #   for i in range(cfg.num_epochs):
  #     tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
  #     train_metrics = estimator.train(
  #         input_fn=train_input, steps=cfg.steps_per_epoch or None)
  #     tf.logging.info(pprint.pformat(train_metrics))
  #     eval_metrics = estimator.evaluate(input_fn=eval_input)
  #     tf.logging.info(pprint.pformat(eval_metrics))
  #     if report_fn:
  #       report_fn(eval_metrics)
  #   return eval_metrics
  # else:
  #   for i in range(cfg.num_epochs):
  #     tf.logging.info("Starting epoch %s/%s" % (i + 1, cfg.num_epochs))
  #     train_metrics = estimator.train(
  #         input_fn=train_input, steps=cfg.steps_per_epoch)
  #     tf.logging.info(pprint.pformat(train_metrics))

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu,
    logdir=FLAGS.model_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  # tpupoint.optimize_input_fn(train_input, blocking=True, worst=False)
  tpupoint.optimize_input_fn(train_input, blocking=False, worst=False)
  # naive_params = {
  #   'auto_vectorize_map': ['False'],
  #   'auto_prefetch_buffer_size': ['Autotune'],
  #   'auto_map_parallel': ['Autotune'],
  #   'auto_map_and_batch':['FALSE'],
  #   'auto_model_fn': tpupoint.autoadjustclass.train_test_model_fn,
  # }
  # tpupoint.autoadjustclass.train_params_results[float('inf')] = tpupoint.autoadjustclass.GetAutoParams(**naive_params)
  # train_input = tpupoint.autoadjustclass.GetWorstModifiedDataset
  # train_input, eval_input = Naive_new_build_dataset( cfg.dataset, is_tpu=cfg.tpu.enable)
  # tpupoint.autoadjustclass.train_params_results[float('-inf')] = tpupoint.autoadjustclass.GetAutoParams(**naive_params)
  # train_input = tpupoint.autoadjustclass.GetModifiedDataset

  tf.logging.info("Starting training for  %s steps" % (cfg.steps_per_epoch * cfg.num_epochs))
  bench_start = time.time()
  # tpupoint.Start()
  # train_metrics = estimator.train( input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  # tpupoint.train(estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  # tpupoint.train_naive(estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  # tpupoint.train_dynamic(estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  tpupoint.train_dynamic(model_fn=model.model_fn , estimator=estimator, input_fn=train_input, steps=(cfg.steps_per_epoch * cfg.num_epochs))
  # tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  # tf.logging.info(pprint.pformat(train_metrics))
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()







