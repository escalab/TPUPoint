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

sys.path.append( os.path.join( os.getcwd() , '..', 'tpu', 'models', 'experimental', 'dcgan' ) )

from dcgan_main import *
import dcgan_main


class CIFARInputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.cifar_train_data_file if is_training else FLAGS.cifar_test_data_file)

  def __call__(self, params):
    batch_size = params['batch_size']
    dataset = tf.data.TFRecordDataset([self.data_file])
    dataset = dataset.map(cifar_input.parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)

    def MapFunc(images, labels):
      images = tf.reshape(images, [batch_size, 32, 32, 3])
      random_noise = tf.random_normal([batch_size, self.noise_dim])
      return random_noise, images
    dataset = dataset.map(MapFunc)
    return dataset

    # images, labels = dataset.make_one_shot_iterator().get_next()
    # # Reshape to give inputs statically known shapes.
    # images = tf.reshape(images, [batch_size, 32, 32, 3])
    # random_noise = tf.random_normal([batch_size, self.noise_dim])
    # features = { 'real_images': images, 'random_noise': random_noise}
    # return features, labels


class MNISTInputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.mnist_train_data_file if is_training else FLAGS.mnist_test_data_file)

  def __call__(self, params):
    """Creates a simple Dataset pipeline."""
    batch_size = params['batch_size']
    dataset = tf.data.TFRecordDataset(self.data_file)
    dataset = dataset.map(mnist_input.parser).cache()
    if self.is_training:
      dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)    # Prefetch overlaps in-feed with training

    def MapFunc(images, labels):
      random_noise = tf.random_normal([batch_size, self.noise_dim])
      return random_noise, images
    dataset = dataset.map(MapFunc)
    return dataset

    # images, labels = dataset.make_one_shot_iterator().get_next()
    # random_noise = tf.random_normal([batch_size, self.noise_dim])
    # features = { 'real_images': images, 'random_noise': random_noise}
    # return features, labels


class NaiveCIFARInputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.cifar_train_data_file if is_training else FLAGS.cifar_test_data_file)

  def __call__(self, params):
    batch_size = params['batch_size']
    dataset = tf.data.TFRecordDataset([self.data_file])
    dataset = dataset.map(cifar_input.parser)
    # dataset = dataset.prefetch(batch_size).cache().repeat()
    dataset = dataset.prefetch( int(batch_size / 2) ).repeat() #.cache().repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    def MapFunc(images, labels):
      images = tf.reshape(images, [batch_size, 32, 32, 3])
      random_noise = tf.random_normal([batch_size, self.noise_dim])
      return random_noise, images
    dataset = dataset.map(MapFunc, num_parallel_calls=1)
    return dataset

    # images, labels = dataset.make_one_shot_iterator().get_next()
    # # Reshape to give inputs statically known shapes.
    # images = tf.reshape(images, [batch_size, 32, 32, 3])
    # random_noise = tf.random_normal([batch_size, self.noise_dim])
    # features = { 'real_images': images, 'random_noise': random_noise}
    # return features, labels


class NaiveMNISTInputFunction(object):
  """Wrapper class that is passed as callable to Estimator."""

  def __init__(self, is_training, noise_dim):
    self.is_training = is_training
    self.noise_dim = noise_dim
    self.data_file = (FLAGS.mnist_train_data_file if is_training else FLAGS.mnist_test_data_file)

  def __call__(self, params):
    """Creates a simple Dataset pipeline."""
    batch_size = params['batch_size']
    dataset = tf.data.TFRecordDataset(self.data_file)
    dataset = dataset.map(mnist_input.parser) #.cache()
    if self.is_training:
      dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    # dataset = dataset.prefetch(batch_size)
    dataset = dataset.prefetch( batch_size )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)    # Prefetch overlaps in-feed with training

    def MapFunc(images, labels):
      random_noise = tf.random_normal([batch_size, self.noise_dim])
      return random_noise, images
    dataset = dataset.map(MapFunc, num_parallel_calls=1)
    return dataset

    # images, labels = dataset.make_one_shot_iterator().get_next()
    # random_noise = tf.random_normal([batch_size, self.noise_dim])
    # features = { 'real_images': images, 'random_noise': random_noise}
    # return features, labels






def dcgan_run_baseline():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.

	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, MNISTInputFunction),
			'cifar': (cifar_model, CIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)

	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)
	bench_start = time.time()
	est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	#		est.train(input_fn=train_eval_input_fn, max_steps=next_checkpoint)
	#	 current_step = next_checkpoint
	#	 tf.logging.info('Finished training step %d' % current_step)

	#	 if FLAGS.eval_loss:
	#		 # Evaluate loss on test set
	#		 metrics = est.evaluate(train_eval_input_fn,
	#														steps=dataset.NUM_EVAL_IMAGES // FLAGS.batch_size)
	#		 tf.logging.info('Finished evaluating')
	#		 tf.logging.info(metrics)

	#	 # Render some generated images
	#	 generated_iter = est.predict(input_fn=noise_input_fn)
	#	 images = [p['generated_data'][:, :, :] for p in generated_iter]
	#	 assert len(images) == _NUM_VIZ_IMAGES
	#	 image_rows = [np.concatenate(images[i:i+10], axis=0)
	#								 for i in range(0, _NUM_VIZ_IMAGES, 10)]
	#	 tiled_image = np.concatenate(image_rows, axis=1)

	#	 img = dataset.convert_array_to_image(tiled_image)

	#	 step_string = str(current_step).zfill(5)
	#	 file_obj = tf.gfile.Open(
	#			 os.path.join(FLAGS.model_dir,
	#										'generated_images', 'gen_%s.png' % (step_string)), 'w')
	#	 img.save(file_obj, format='png')
	#	 tf.logging.info('Finished generating images')



def dcgan_run_eval():
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.
	model, dataset = {
			'mnist': (mnist_model, mnist_input),
			'cifar': (cifar_model, cifar_input),
	}[FLAGS.dataset]

	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)
	# est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	#		est.train(input_fn=train_eval_input_fn, max_steps=next_checkpoint)
	#	 current_step = next_checkpoint
	#	 tf.logging.info('Finished training step %d' % current_step)

	#	 if FLAGS.eval_loss:
	#		 # Evaluate loss on test set
	try:

		metrics = est.evaluate(train_eval_input_fn, steps=dataset.NUM_EVAL_IMAGES // FLAGS.batch_size)
		tf.logging.info('Finished evaluating')
		tf.logging.info(metrics)
	except:
		# CIFAR10 has 50000 training images and 10000 test
		NUM_TRAIN_IMAGES = 50000
		NUM_EVAL_IMAGES = 10000
		metrics = est.evaluate(train_eval_input_fn, steps=NUM_TRAIN_IMAGES // FLAGS.batch_size)
		tf.logging.info('Finished evaluating')
		tf.logging.info(metrics)

	#	 # Render some generated images
	#	 generated_iter = est.predict(input_fn=noise_input_fn)
	#	 images = [p['generated_data'][:, :, :] for p in generated_iter]
	#	 assert len(images) == _NUM_VIZ_IMAGES
	#	 image_rows = [np.concatenate(images[i:i+10], axis=0)
	#								 for i in range(0, _NUM_VIZ_IMAGES, 10)]
	#	 tiled_image = np.concatenate(image_rows, axis=1)

	#	 img = dataset.convert_array_to_image(tiled_image)

	#	 step_string = str(current_step).zfill(5)
	#	 file_obj = tf.gfile.Open(
	#			 os.path.join(FLAGS.model_dir,
	#										'generated_images', 'gen_%s.png' % (step_string)), 'w')
	#	 img.save(file_obj, format='png')
	#	 tf.logging.info('Finished generating images')



def dcgan_run_profile():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.
	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, MNISTInputFunction),
			'cifar': (cifar_model, CIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)

	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)

	tpupoint = TPUPoint( 
		estimator = est,
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
	est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def dcgan_run_optimize():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.
	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, MNISTInputFunction),
			'cifar': (cifar_model, CIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)
	
	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)

	tpupoint = TPUPoint( 
		estimator = est,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tpupoint.optimize_input_fn(train_eval_input_fn, blocking=True)

	bench_start = time.time()
	# tpupoint.Start()
	# est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	tpupoint.train(estimator=est, input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def dcgan_run_dynamic():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.
	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, MNISTInputFunction),
			'cifar': (cifar_model, CIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)
	
	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)

	tpupoint = TPUPoint( 
		estimator = est,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tpupoint.optimize_input_fn(train_eval_input_fn)

	bench_start = time.time()
	# tpupoint.Start()
	# est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	tpupoint.train_dynamic(model_fn=model ,estimator=est, input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()



def dcgan_run_naive_wo_tpupoint():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.

	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, NaiveMNISTInputFunction),
			'cifar': (cifar_model, NaiveCIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)

	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)
	tpupoint = TPUPoint( 
		estimator = est,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 
	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)
	bench_start = time.time()
	tpupoint.Start()
	est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def dcgan_run_naive_wo_tpupoint1():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.
	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, MNISTInputFunction),
			'cifar': (cifar_model, CIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)
	
	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)

	tpupoint = TPUPoint( 
		estimator = est,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tpupoint.optimize_input_fn(train_eval_input_fn, blocking=True, worst=True)

	bench_start = time.time()
	# tpupoint.Start()
	# est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	tpupoint.train_naive(estimator=est, input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	# tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()


def dcgan_run_naive_w_tpupoint():
	bench_total_start = time.time()
	_NUM_VIZ_IMAGES = dcgan_main._NUM_VIZ_IMAGES
	tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
			FLAGS.tpu,
			zone=FLAGS.tpu_zone,
			project=FLAGS.gcp_project)

	config = tf.compat.v1.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			model_dir=FLAGS.model_dir,
			tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
					num_shards=FLAGS.num_shards,
					iterations_per_loop=FLAGS.iterations_per_loop))

	# Get the generator and discriminator functions depending on which dataset
	# we want to train on.

	# model, dataset = {
	# 		'mnist': (mnist_model, mnist_input),
	# 		'cifar': (cifar_model, cifar_input),
	# }[FLAGS.dataset]
	model, dataset = {
			'mnist': (mnist_model, NaiveMNISTInputFunction),
			'cifar': (cifar_model, NaiveCIFARInputFunction),
	}[FLAGS.dataset]
	train_eval_input_fn = dataset(True, FLAGS.noise_dim)

	def unconditional_generator(noise, mode):
		"""Generator with extra argument for tf.Estimator's `mode`."""
		is_training = (mode == tf.estimator.ModeKeys.TRAIN)
		return model.generator(noise, is_training=is_training)

	def unconditional_discriminator(images, unused_conditioning):
		"""Discriminator that conforms to TF-GAN API."""
		return model.discriminator(images, is_training=True)

	# TPU-based estimator used for TRAIN and EVAL
	# TODO(joelshor): Add get_eval_metric_ops_fn.
	est = tfgan.estimator.TPUGANEstimator(
			generator_fn=unconditional_generator,
			discriminator_fn=unconditional_discriminator,
			generator_loss_fn=tfgan.losses.minimax_generator_loss,
			discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss,
			generator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			discriminator_optimizer=tf.train.AdamOptimizer(FLAGS.learning_rate, 0.5),
			joint_train=True,	# train G and D jointly instead of sequentially.
			eval_on_tpu=True,
			train_batch_size=FLAGS.batch_size,
			eval_batch_size=FLAGS.batch_size,
			predict_batch_size=_NUM_VIZ_IMAGES,
			use_tpu=FLAGS.use_tpu,
			config=config)

	tpupoint = TPUPoint( 
		estimator = est,
		gcp_project=FLAGS.gcp_project,
		tpu_zone=FLAGS.tpu_zone,
		tpu=FLAGS.tpu,
		logdir=FLAGS.model_dir,
		workers_list = None,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 ) 

	tpupoint.optimize_input_fn(train_eval_input_fn, blocking=False, worst=False)

	# Get the tf.Estimator `input_fn` for training and evaluation.
	# train_eval_input_fn = functools.partial(input_fn, dataset=dataset)
	tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))

	current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)	 # pylint: disable=protected-access,line-too-long
	tf.logging.info('Starting training for %d steps, current step: %d' %
									(FLAGS.train_steps, current_step))
	# while current_step < FLAGS.train_steps:
	#	 next_checkpoint = min(current_step + FLAGS.train_steps_per_eval,
	#												 FLAGS.train_steps)
	bench_start = time.time()
	# est.train(input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	# tpupoint.train(estimator=est, input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	tpupoint.train_dynamic(model_fn=model ,estimator=est, input_fn=train_eval_input_fn, max_steps=FLAGS.train_steps)
	# 
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()













