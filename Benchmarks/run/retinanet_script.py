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

sys.path.append( os.path.join( os.getcwd() , '..', 'tpu', 'models', 'official', 'retinanet' ) )
sys.path.append( os.path.join( os.getcwd() , '..', 'tpu', 'models' ) )

from retinanet_main import *
import retinanet_main

from dataloader import *
import anchors
from object_detection import preprocessor
from object_detection import tf_example_decoder

MAX_NUM_INSTANCES = 100
# Represents the number of bytes in the read buffer.
BUFFER_SIZE = None

class Naive_InputReader(object):
  """Input reader for dataset."""

  def __init__(self, file_pattern, is_training, num_examples=0):
    self._file_pattern = file_pattern
    self._is_training = is_training
    self._num_examples = num_examples
    self._max_num_instances = MAX_NUM_INSTANCES

  def __call__(self, params):
    input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                    params['num_scales'],
                                    params['aspect_ratios'],
                                    params['anchor_scale'],
                                    params['image_size'])
    anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
    example_decoder = tf_example_decoder.TfExampleDecoder()

    def _dataset_parser(value):
      """Parse data to a fixed dimension input image and learning targets.

      Args:
        value: A dictionary contains an image and groundtruth annotations.

      Returns:
        image: Image tensor that is preproessed to have normalized value and
          fixed dimension [image_size, image_size, 3]
        cls_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors]. The height_l and width_l
          represent the dimension of class logits at l-th level.
        box_targets_dict: ordered dictionary with keys
          [min_level, min_level+1, ..., max_level]. The values are tensor with
          shape [height_l, width_l, num_anchors * 4]. The height_l and
          width_l represent the dimension of bounding box regression output at
          l-th level.
        num_positives: Number of positive anchors in the image.
        source_id: Source image id. Default value -1 if the source id is empty
          in the groundtruth annotation.
        image_scale: Scale of the proccessed image to the original image.
        image_info: image information that includes the original height and
            width, the scale of the proccessed image to the original image, and
            the scaled height and width.
        boxes: Groundtruth bounding box annotations. The box is represented in
          [y1, x1, y2, x2] format. The tennsor is padded with -1 to the fixed
          dimension [self._max_num_instances, 4].
        is_crowds: Groundtruth annotations to indicate if an annotation
          represents a group of instances by value {0, 1}. The tennsor is
          padded with 0 to the fixed dimension [self._max_num_instances].
        areas: Groundtruth areas annotations. The tennsor is padded with -1
          to the fixed dimension [self._max_num_instances].
        classes: Groundtruth classes annotations. The tennsor is padded with -1
          to the fixed dimension [self._max_num_instances].
      """
      with tf.name_scope('parser'):
        data = example_decoder.decode(value)
        data['groundtruth_is_crowd'] = tf.cond(
            tf.greater(tf.size(data['groundtruth_is_crowd']), 0),
            lambda: data['groundtruth_is_crowd'],
            lambda: tf.zeros_like(data['groundtruth_classes'], dtype=tf.bool))
        source_id = data['source_id']
        image = data['image']
        boxes = data['groundtruth_boxes']
        classes = data['groundtruth_classes']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        areas = data['groundtruth_area']
        is_crowds = data['groundtruth_is_crowd']
        classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        input_height = tf.shape(image)[0]
        input_width = tf.shape(image)[1]

        if params['skip_crowd_during_training'] and self._is_training:
          indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
          classes = tf.gather_nd(classes, indices)
          boxes = tf.gather_nd(boxes, indices)

        input_processor = DetectionInputProcessor(
            image, params['image_size'], boxes, classes)
        input_processor.normalize_image()
        if self._is_training and params['input_rand_hflip']:
          input_processor.random_horizontal_flip()
        if self._is_training:
          input_processor.set_training_random_scale_factors(
              params['train_scale_min'], params['train_scale_max'])
        else:
          input_processor.set_scale_factors_to_output_size()
        image = input_processor.resize_and_crop_image()
        boxes, classes = input_processor.resize_and_crop_boxes()

        # Assign anchors.
        (cls_targets, box_targets,
         num_positives) = anchor_labeler.label_anchors(boxes, classes)

        source_id = tf.where(tf.equal(source_id, tf.constant('')), '-1',
                             source_id)
        source_id = tf.string_to_number(source_id)

        # Pad groundtruth data for evaluation.
        image_scale = input_processor.image_scale_to_original
        scaled_height = tf.to_float(input_height) * input_processor.image_scale
        scaled_width = tf.to_float(input_width) * input_processor.image_scale
        image_info = tf.stack([
            tf.cast(scaled_height, dtype=tf.float32),
            tf.cast(scaled_width, dtype=tf.float32),
            image_scale,
            tf.cast(input_height, dtype=tf.float32),
            tf.cast(input_width, dtype=tf.float32),
        ])
        boxes *= image_scale
        is_crowds = tf.cast(is_crowds, dtype=tf.float32)
        boxes = pad_to_fixed_size(boxes, -1, [self._max_num_instances, 4])
        is_crowds = pad_to_fixed_size(is_crowds, 0,
                                      [self._max_num_instances, 1])
        areas = pad_to_fixed_size(areas, -1, [self._max_num_instances, 1])
        classes = pad_to_fixed_size(classes, -1, [self._max_num_instances, 1])
        if params['use_bfloat16']:
          image = tf.cast(image, dtype=tf.bfloat16)
        return (image, cls_targets, box_targets, num_positives, source_id,
                image_scale, image_info, boxes, is_crowds, areas, classes)

    batch_size = params['batch_size']
    dataset = tf.data.Dataset.list_files( self._file_pattern, shuffle=self._is_training, seed=tf.random.set_random_seed(int(time.time() * 1e9)))
    if self._is_training:
      dataset = dataset.repeat()

    # Prefetch data from files.
    def _prefetch_dataset(filename):
      dataset = tf.data.TFRecordDataset(
          filename, buffer_size=BUFFER_SIZE).prefetch(1)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            _prefetch_dataset, cycle_length=32, sloppy=self._is_training))

    if params.get('dataset_private_threadpool_size', None):
      # options = tf.data.Options()
      # options.experimental_threading.private_threadpool_size = params[ 'dataset_private_threadpool_size']
      # dataset = dataset.with_options(options)
      pass

    if params.get('dataset_max_intra_op_parallelism', None):
      # options = tf.data.Options()
      # options.experimental_threading.max_intra_op_parallelism = params['dataset_max_intra_op_parallelism']
      # dataset = dataset.with_options(options)
      pass

    if self._is_training:
      dataset = dataset.shuffle(64)

    # Parse the fetched records to input tensors for model function.
    # dataset = dataset.map(_dataset_parser, num_parallel_calls=64)
    dataset = dataset.map(_dataset_parser, num_parallel_calls=1)
    # dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.prefetch(1)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def _process_example(images, cls_targets, box_targets, num_positives,
                         source_ids, image_scales, image_info, boxes, is_crowds,
                         areas, classes):
      """Processes one batch of data."""
      labels = {}
      # Count num_positives in a batch.
      num_positives_batch = tf.reduce_mean(num_positives)
      labels['mean_num_positives'] = tf.reshape(
          tf.tile(tf.expand_dims(num_positives_batch, 0), [
              batch_size,
          ]), [batch_size, 1])

      for level in range(params['min_level'], params['max_level'] + 1):
        labels['cls_targets_%d' % level] = cls_targets[level]
        labels['box_targets_%d' % level] = box_targets[level]
      # Concatenate groundtruth annotations to a tensor.
      groundtruth_data = tf.concat([boxes, is_crowds, areas, classes], axis=2)
      labels['source_ids'] = source_ids
      labels['groundtruth_data'] = groundtruth_data
      labels['image_scales'] = image_scales
      labels['image_info'] = image_info
      if not self._is_training:
        return {'inputs': images, 'image_info': image_info, 'labels': labels}
      return images, labels

    # dataset = dataset.map(_process_example)
    dataset = dataset.map(_process_example, num_parallel_calls=1)
    # dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    dataset = dataset.prefetch(1)
    if self._num_examples > 0:
      dataset = dataset.take(self._num_examples)
    return dataset






def retinanet_run_baseline():
	bench_total_start = time.time()
	if FLAGS.use_tpu:
		if FLAGS.distribution_strategy is None:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
					FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
			tpu_grpc_url = tpu_cluster_resolver.get_master()
			tf.Session.reset(tpu_grpc_url)
		else:
			raise RuntimeError(
					'Distribution strategy must be None when --use_tpu is True.')
	else:
		tpu_cluster_resolver = None

	# if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
	#	 raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

	# Check data path
	if(FLAGS.training_file_pattern is None):
		raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('train',
	#									 'train_and_eval') and FLAGS.training_file_pattern is None:
	#	 raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('eval', 'train_and_eval'):
	if FLAGS.validation_file_pattern is None:
		raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')

	# Parse hparams
	hparams = retinanet_model.default_hparams()
	config_file = FLAGS.config_file
	hparams.num_epochs = FLAGS.num_epochs
	if config_file and tf.gfile.Exists(config_file):
		# load params from file.
		with tf.gfile.Open(config_file, 'r') as f:
			values_map = json.load(f)
			hparams.override_from_dict(values_map)
	hparams.parse(FLAGS.hparams)

	# The following is for spatial partitioning. `features` has one tensor while
	# `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
	# partition is performed on `features` and all partitionable tensors of
	# `labels`, see the partition logic below.
	# In the TPUEstimator context, the meaning of `shard` and `replica` is the
	# same; follwing the API, here has mixed use of both.
	if FLAGS.use_spatial_partition:
		# Checks input_partition_dims agrees with num_cores_per_replica.
		if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
			raise RuntimeError('--num_cores_per_replica must be a product of array'
												 'elements in --input_partition_dims.')

		labels_partition_dims = {
				'mean_num_positives': None,
				'source_ids': None,
				'groundtruth_data': None,
				'image_scales': None,
		}
		# The Input Partition Logic: We partition only the partition-able tensors.
		# Spatial partition requires that the to-be-partitioned tensors must have a
		# dimension that is a multiple of `partition_dims`. Depending on the
		# `partition_dims` and the `image_size` and the `max_level` in hparams, some
		# high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
		# be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
		# size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
		# [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
		# case, the level-8 and level-9 target tensors are not partition-able, and
		# the highest partition-able level is 7.
		image_size = hparams.get('image_size')
		for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

			def _can_partition(spatial_dim):
				partitionable_index = np.where(
						spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
				return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

			spatial_dim = image_size // (2**level)
			if _can_partition(spatial_dim):
				labels_partition_dims['box_targets_%d' %
															level] = FLAGS.input_partition_dims
				labels_partition_dims['cls_targets_%d' %
															level] = FLAGS.input_partition_dims
			else:
				labels_partition_dims['box_targets_%d' % level] = None
				labels_partition_dims['cls_targets_%d' % level] = None

		num_cores_per_replica = FLAGS.num_cores_per_replica
		input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
		num_shards = FLAGS.num_cores // num_cores_per_replica
	else:
		num_cores_per_replica = None
		input_partition_dims = None
		num_shards = FLAGS.num_cores

	config_proto = tf.ConfigProto(
			allow_soft_placement=True, log_device_placement=False)
	if FLAGS.use_xla and not FLAGS.use_tpu:
		config_proto.graph_options.optimizer_options.global_jit_level = (
				tf.OptimizerOptions.ON_1)
	if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
		config_proto.graph_options.rewrite_options.auto_mixed_precision = (
				rewriter_config_pb2.RewriterConfig.ON)

	# if FLAGS.distribution_strategy is None:
	# Uses TPUEstimator.
	params = dict(
			hparams.values(),
			num_shards=num_shards,
			num_examples_per_epoch=FLAGS.num_examples_per_epoch,
			use_tpu=FLAGS.use_tpu,
			resnet_checkpoint=FLAGS.resnet_checkpoint,
			val_json_file=FLAGS.val_json_file,
			mode=FLAGS.mode,
	)
	tpu_config = tf.contrib.tpu.TPUConfig(
			FLAGS.iterations_per_loop,
			num_shards=num_shards,
			num_cores_per_replica=num_cores_per_replica,
			input_partition_dims=input_partition_dims,
			per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
			.PER_HOST_V2)

	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			evaluation_master=FLAGS.eval_master,
			model_dir=FLAGS.model_dir,
			log_step_count_steps=FLAGS.iterations_per_loop,
			session_config=config_proto,
			tpu_config=tpu_config,
	)

	if FLAGS.model_dir is not None:
		if not tf.gfile.Exists(FLAGS.model_dir):
			tf.gfile.MakeDirs(FLAGS.model_dir)
		with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
			json.dump(hparams.values(), f, sort_keys=True, indent=2)
	tf.logging.info(params)
	# if FLAGS.distribution_strategy is None:
	total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
	train_estimator = tf.contrib.tpu.TPUEstimator(
			model_fn=retinanet_model.tpu_retinanet_model_fn,
			use_tpu=FLAGS.use_tpu,
			train_batch_size=FLAGS.train_batch_size,
			config=run_config,
			params=params)

	bench_start = time.time()
	
	train_estimator.train(
			input_fn=dataloader.InputReader(
					FLAGS.training_file_pattern, is_training=True),
			max_steps=total_steps)

	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tf.logging.info('Finished training')
	return 



def retinanet_run_eval():
	if FLAGS.use_tpu:
		if FLAGS.distribution_strategy is None:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
					FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
			tpu_grpc_url = tpu_cluster_resolver.get_master()
			tf.Session.reset(tpu_grpc_url)
		else:
			raise RuntimeError(
					'Distribution strategy must be None when --use_tpu is True.')
	else:
		tpu_cluster_resolver = None

	# if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
	#	 raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

	# Check data path
	if(FLAGS.training_file_pattern is None):
		raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('train',
	#									 'train_and_eval') and FLAGS.training_file_pattern is None:
	#	 raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('eval', 'train_and_eval'):
	if FLAGS.validation_file_pattern is None:
		raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')

	# Parse hparams
	hparams = retinanet_model.default_hparams()
	config_file = FLAGS.config_file
	hparams.num_epochs = FLAGS.num_epochs
	if config_file and tf.gfile.Exists(config_file):
		# load params from file.
		with tf.gfile.Open(config_file, 'r') as f:
			values_map = json.load(f)
			hparams.override_from_dict(values_map)
	hparams.parse(FLAGS.hparams)

	# The following is for spatial partitioning. `features` has one tensor while
	# `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
	# partition is performed on `features` and all partitionable tensors of
	# `labels`, see the partition logic below.
	# In the TPUEstimator context, the meaning of `shard` and `replica` is the
	# same; follwing the API, here has mixed use of both.
	if FLAGS.use_spatial_partition:
		# Checks input_partition_dims agrees with num_cores_per_replica.
		if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
			raise RuntimeError('--num_cores_per_replica must be a product of array'
												 'elements in --input_partition_dims.')

		labels_partition_dims = {
				'mean_num_positives': None,
				'source_ids': None,
				'groundtruth_data': None,
				'image_scales': None,
		}
		# The Input Partition Logic: We partition only the partition-able tensors.
		# Spatial partition requires that the to-be-partitioned tensors must have a
		# dimension that is a multiple of `partition_dims`. Depending on the
		# `partition_dims` and the `image_size` and the `max_level` in hparams, some
		# high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
		# be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
		# size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
		# [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
		# case, the level-8 and level-9 target tensors are not partition-able, and
		# the highest partition-able level is 7.
		image_size = hparams.get('image_size')
		for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

			def _can_partition(spatial_dim):
				partitionable_index = np.where(
						spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
				return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

			spatial_dim = image_size // (2**level)
			if _can_partition(spatial_dim):
				labels_partition_dims['box_targets_%d' %
															level] = FLAGS.input_partition_dims
				labels_partition_dims['cls_targets_%d' %
															level] = FLAGS.input_partition_dims
			else:
				labels_partition_dims['box_targets_%d' % level] = None
				labels_partition_dims['cls_targets_%d' % level] = None

		num_cores_per_replica = FLAGS.num_cores_per_replica
		input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
		num_shards = FLAGS.num_cores // num_cores_per_replica
	else:
		num_cores_per_replica = None
		input_partition_dims = None
		num_shards = FLAGS.num_cores

	config_proto = tf.ConfigProto(
			allow_soft_placement=True, log_device_placement=False)
	if FLAGS.use_xla and not FLAGS.use_tpu:
		config_proto.graph_options.optimizer_options.global_jit_level = (
				tf.OptimizerOptions.ON_1)
	if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
		config_proto.graph_options.rewrite_options.auto_mixed_precision = (
				rewriter_config_pb2.RewriterConfig.ON)

	# if FLAGS.distribution_strategy is None:
	# Uses TPUEstimator.
	params = dict(
			hparams.values(),
			num_shards=num_shards,
			num_examples_per_epoch=FLAGS.num_examples_per_epoch,
			use_tpu=FLAGS.use_tpu,
			resnet_checkpoint=FLAGS.resnet_checkpoint,
			val_json_file=FLAGS.val_json_file,
			mode=FLAGS.mode,
	)
	tpu_config = tf.contrib.tpu.TPUConfig(
			FLAGS.iterations_per_loop,
			num_shards=num_shards,
			num_cores_per_replica=num_cores_per_replica,
			input_partition_dims=input_partition_dims,
			per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
			.PER_HOST_V2)

	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			evaluation_master=FLAGS.eval_master,
			model_dir=FLAGS.model_dir,
			log_step_count_steps=FLAGS.iterations_per_loop,
			session_config=config_proto,
			tpu_config=tpu_config,
	)

	######################

	# Eval only runs on CPU or GPU host with batch_size = 1.
	# Override the default options: disable randomization in the input pipeline
	# and don't run on the TPU.
	# Also, disable use_bfloat16 for eval on CPU/GPU.
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')
	eval_params = dict(
			params,
			input_rand_hflip=False,
			resnet_checkpoint=None,
			is_training_bn=False,
	)
	if FLAGS.distribution_strategy is None:
		# Uses TPUEstimator.
		eval_estimator = tf.contrib.tpu.TPUEstimator(
				model_fn=retinanet_model.tpu_retinanet_model_fn,
				use_tpu=FLAGS.use_tpu,
				train_batch_size=FLAGS.train_batch_size,
				eval_batch_size=FLAGS.eval_batch_size,
				predict_batch_size=FLAGS.eval_batch_size,
				config=run_config,
				params=eval_params)
	else:
		# Uses Estimator.
		if FLAGS.distribution_strategy == 'multi_worker_mirrored':
			raise ValueError(
					'--distribution_strategy=multi_worker_mirrored is not supported '
					'for eval.')
		elif FLAGS.distribution_strategy == 'mirrored':
			eval_estimator = tf.estimator.Estimator(
					model_fn=retinanet_model.est_retinanet_model_fn,
					model_dir=FLAGS.model_dir,
					config=run_config,
					params=params)
		else:
			raise ValueError('Unrecognized distribution strategy.')

	def terminate_eval():
		tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
										FLAGS.eval_timeout)
		return True

	output_dir = os.path.join(FLAGS.model_dir, 'eval')
	tf.gfile.MakeDirs(output_dir)
	summary_writer = tf.summary.FileWriter(output_dir)
	# Run evaluation when there's a new checkpoint
	# for ckpt in tf.contrib.training.checkpoints_iterator( FLAGS.model_dir, min_interval_secs=FLAGS.min_eval_interval, timeout=FLAGS.eval_timeout, timeout_fn=terminate_eval):
	for ckpt in tf.contrib.training.checkpoints_iterator( FLAGS.model_dir ):

		tf.logging.info('Starting to evaluate.')
		try:
			eval_results = evaluation.evaluate(
					eval_estimator,
					input_fn=dataloader.InputReader(
							FLAGS.validation_file_pattern, is_training=False),
					num_eval_samples=FLAGS.eval_samples,
					eval_batch_size=FLAGS.eval_batch_size,
					validation_json_file=FLAGS.val_json_file)
			tf.logging.info('Eval results: %s' % eval_results)

			# Terminate eval job when final checkpoint is reached
			current_step = int(os.path.basename(ckpt).split('-')[1])
			total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
											 FLAGS.train_batch_size)
			evaluation.write_summary(eval_results, summary_writer, current_step)
			if current_step >= total_step:
				tf.logging.info(
						'Evaluation finished after training step %d' % current_step)
				break

		except tf.errors.NotFoundError:
			# Since the coordinator is on a different job than the TPU worker,
			# sometimes the TPU worker does not finish initializing until long after
			# the CPU job tells it to start evaluating. In this case, the checkpoint
			# file could have been deleted already.
			tf.logging.info( 'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)
	tf.logging.info("Evaluation finished")
	return
	##########################

	if FLAGS.mode == 'train':
		if FLAGS.model_dir is not None:
			if not tf.gfile.Exists(FLAGS.model_dir):
				tf.gfile.MakeDirs(FLAGS.model_dir)
			with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'),
												 'w') as f:
				json.dump(hparams.values(), f, sort_keys=True, indent=2)
		tf.logging.info(params)
		if FLAGS.distribution_strategy is None:
			total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
												FLAGS.train_batch_size)
			train_estimator = tf.contrib.tpu.TPUEstimator(
					model_fn=retinanet_model.tpu_retinanet_model_fn,
					use_tpu=FLAGS.use_tpu,
					train_batch_size=FLAGS.train_batch_size,
					config=run_config,
					params=params)
			train_estimator.train(
					input_fn=dataloader.InputReader(
							FLAGS.training_file_pattern, is_training=True),
					max_steps=total_steps)

			# Run evaluation after training finishes.
			eval_params = dict(
					params,
					input_rand_hflip=False,
					resnet_checkpoint=None,
					is_training_bn=False,
			)
			eval_estimator = tf.contrib.tpu.TPUEstimator(
					model_fn=retinanet_model.tpu_retinanet_model_fn,
					use_tpu=FLAGS.use_tpu,
					train_batch_size=FLAGS.train_batch_size,
					eval_batch_size=FLAGS.eval_batch_size,
					predict_batch_size=FLAGS.eval_batch_size,
					config=run_config,
					params=eval_params)
			if FLAGS.eval_after_training:

				if FLAGS.val_json_file is None:
					raise RuntimeError('You must specify --val_json_file for evaluation.')

				eval_results = evaluation.evaluate(
						eval_estimator,
						input_fn=dataloader.InputReader(
								FLAGS.validation_file_pattern, is_training=False),
						num_eval_samples=FLAGS.eval_samples,
						eval_batch_size=FLAGS.eval_batch_size,
						validation_json_file=FLAGS.val_json_file)
				tf.logging.info('Eval results: %s' % eval_results)
				output_dir = os.path.join(FLAGS.model_dir, 'train_eval')
				tf.gfile.MakeDirs(output_dir)
				summary_writer = tf.summary.FileWriter(output_dir)

				evaluation.write_summary(eval_results, summary_writer, total_steps)
		else:
			train_estimator = tf.estimator.Estimator(
					model_fn=retinanet_model.est_retinanet_model_fn,
					model_dir=FLAGS.model_dir,
					config=run_config,
					params=params)
			if FLAGS.distribution_strategy == 'mirrored':
				total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
													FLAGS.train_batch_size)
				tf.logging.info('Starting `MirroredStrategy` training...')
				train_estimator.train(
						input_fn=dataloader.InputReader(
								FLAGS.training_file_pattern, is_training=True),
						max_steps=total_steps)
			elif FLAGS.distribution_strategy == 'multi_worker_mirrored':
				total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
													(len(worker_hosts) * FLAGS.train_batch_size))
				train_spec = tf.estimator.TrainSpec(
						input_fn=dataloader.InputReader(
								FLAGS.training_file_pattern, is_training=True),
						max_steps=total_steps)
				eval_spec = tf.estimator.EvalSpec(input_fn=tf.data.Dataset)
				tf.logging.info('Starting `MultiWorkerMirroredStrategy` training...')
				tf.estimator.train_and_evaluate(train_estimator, train_spec, eval_spec)
			else:
				raise ValueError('Unrecognized distribution strategy.')

	elif FLAGS.mode == 'eval':
		# Eval only runs on CPU or GPU host with batch_size = 1.
		# Override the default options: disable randomization in the input pipeline
		# and don't run on the TPU.
		# Also, disable use_bfloat16 for eval on CPU/GPU.
		if FLAGS.val_json_file is None:
			raise RuntimeError('You must specify --val_json_file for evaluation.')
		eval_params = dict(
				params,
				input_rand_hflip=False,
				resnet_checkpoint=None,
				is_training_bn=False,
		)
		if FLAGS.distribution_strategy is None:
			# Uses TPUEstimator.
			eval_estimator = tf.contrib.tpu.TPUEstimator(
					model_fn=retinanet_model.tpu_retinanet_model_fn,
					use_tpu=FLAGS.use_tpu,
					train_batch_size=FLAGS.train_batch_size,
					eval_batch_size=FLAGS.eval_batch_size,
					predict_batch_size=FLAGS.eval_batch_size,
					config=run_config,
					params=eval_params)
		else:
			# Uses Estimator.
			if FLAGS.distribution_strategy == 'multi_worker_mirrored':
				raise ValueError(
						'--distribution_strategy=multi_worker_mirrored is not supported '
						'for eval.')
			elif FLAGS.distribution_strategy == 'mirrored':
				eval_estimator = tf.estimator.Estimator(
						model_fn=retinanet_model.est_retinanet_model_fn,
						model_dir=FLAGS.model_dir,
						config=run_config,
						params=params)
			else:
				raise ValueError('Unrecognized distribution strategy.')

		def terminate_eval():
			tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
											FLAGS.eval_timeout)
			return True

		output_dir = os.path.join(FLAGS.model_dir, 'eval')
		tf.gfile.MakeDirs(output_dir)
		summary_writer = tf.summary.FileWriter(output_dir)
		# Run evaluation when there's a new checkpoint
		for ckpt in tf.contrib.training.checkpoints_iterator( FLAGS.model_dir, min_interval_secs=FLAGS.min_eval_interval, timeout=FLAGS.eval_timeout, timeout_fn=terminate_eval):

			tf.logging.info('Starting to evaluate.')
			try:
				eval_results = evaluation.evaluate(
						eval_estimator,
						input_fn=dataloader.InputReader(
								FLAGS.validation_file_pattern, is_training=False),
						num_eval_samples=FLAGS.eval_samples,
						eval_batch_size=FLAGS.eval_batch_size,
						validation_json_file=FLAGS.val_json_file)
				tf.logging.info('Eval results: %s' % eval_results)

				# Terminate eval job when final checkpoint is reached
				current_step = int(os.path.basename(ckpt).split('-')[1])
				total_step = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) /
												 FLAGS.train_batch_size)
				evaluation.write_summary(eval_results, summary_writer, current_step)
				if current_step >= total_step:
					tf.logging.info(
							'Evaluation finished after training step %d' % current_step)
					break

			except tf.errors.NotFoundError:
				# Since the coordinator is on a different job than the TPU worker,
				# sometimes the TPU worker does not finish initializing until long after
				# the CPU job tells it to start evaluating. In this case, the checkpoint
				# file could have been deleted already.
				tf.logging.info(
						'Checkpoint %s no longer exists, skipping checkpoint' % ckpt)

	elif FLAGS.mode == 'train_and_eval':
		if FLAGS.distribution_strategy is not None:
			raise ValueError(
					'Distribution strategy is not implemented for --mode=train_and_eval.')
		if FLAGS.val_json_file is None:
			raise RuntimeError('You must specify --val_json_file for evaluation.')

		output_dir = os.path.join(FLAGS.model_dir, 'train_and_eval')
		tf.gfile.MakeDirs(output_dir)
		summary_writer = tf.summary.FileWriter(output_dir)
		num_cycles = int(FLAGS.num_epochs * FLAGS.num_examples_per_epoch /
										 FLAGS.num_steps_per_eval)
		for cycle in range(num_cycles):
			tf.logging.info('Starting training cycle, epoch: %d.' % cycle)
			train_estimator = tf.contrib.tpu.TPUEstimator(
					model_fn=retinanet_model.tpu_retinanet_model_fn,
					use_tpu=FLAGS.use_tpu,
					train_batch_size=FLAGS.train_batch_size,
					config=run_config,
					params=params)
			train_estimator.train(
					input_fn=dataloader.InputReader(
							FLAGS.training_file_pattern, is_training=True),
					steps=FLAGS.num_steps_per_eval)

			tf.logging.info('Starting evaluation cycle, epoch: %d.' % cycle)
			# Run evaluation after every epoch.
			eval_params = dict(
					params,
					input_rand_hflip=False,
					resnet_checkpoint=None,
					is_training_bn=False,
			)

			eval_estimator = tf.contrib.tpu.TPUEstimator(
					model_fn=retinanet_model.tpu_retinanet_model_fn,
					use_tpu=FLAGS.use_tpu,
					train_batch_size=FLAGS.train_batch_size,
					eval_batch_size=FLAGS.eval_batch_size,
					config=run_config,
					params=eval_params)
			eval_results = evaluation.evaluate(
					eval_estimator,
					input_fn=dataloader.InputReader(
							FLAGS.validation_file_pattern, is_training=False),
					num_eval_samples=FLAGS.eval_samples,
					eval_batch_size=FLAGS.eval_batch_size,
					validation_json_file=FLAGS.val_json_file)
			tf.logging.info('Evaluation results: %s' % eval_results)
			current_step = int(cycle * FLAGS.num_steps_per_eval)
			evaluation.write_summary(eval_results, summary_writer, current_step)

	else:
		tf.logging.info('Mode not found.')

	# if FLAGS.model_dir:
	#	 tf.logging.info('Exporting saved model.')
	#	 eval_params = dict(
	#			 params,
	#			 use_tpu=True,
	#			 input_rand_hflip=False,
	#			 resnet_checkpoint=None,
	#			 is_training_bn=False,
	#			 use_bfloat16=False,
	#	 )
	#	 eval_estimator = tf.contrib.tpu.TPUEstimator(
	#			 model_fn=retinanet_model.tpu_retinanet_model_fn,
	#			 use_tpu=True,
	#			 train_batch_size=FLAGS.train_batch_size,
	#			 predict_batch_size=FLAGS.inference_batch_size,
	#			 config=run_config,
	#			 params=eval_params)
	#	 export_path = eval_estimator.export_saved_model(
	#			 export_dir_base=FLAGS.model_dir,
	#			 serving_input_receiver_fn=build_serving_input_fn(
	#					 hparams.image_size,
	#					 FLAGS.inference_batch_size))
	#	 if FLAGS.add_warmup_requests:
	#		 inference_warmup.write_warmup_requests(
	#				 export_path,
	#				 FLAGS.model_name,
	#				 hparams.image_size,
	#				 batch_sizes=[FLAGS.inference_batch_size])



def retinanet_run_profile():
	bench_total_start = time.time()
	if FLAGS.use_tpu:
		if FLAGS.distribution_strategy is None:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
					FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
			tpu_grpc_url = tpu_cluster_resolver.get_master()
			tf.Session.reset(tpu_grpc_url)
		else:
			raise RuntimeError(
					'Distribution strategy must be None when --use_tpu is True.')
	else:
		tpu_cluster_resolver = None

	# if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
	#	 raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

	# Check data path
	if(FLAGS.training_file_pattern is None):
		raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('train',
	#									 'train_and_eval') and FLAGS.training_file_pattern is None:
	#	 raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('eval', 'train_and_eval'):
	if FLAGS.validation_file_pattern is None:
		raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')

	# Parse hparams
	hparams = retinanet_model.default_hparams()
	config_file = FLAGS.config_file
	hparams.num_epochs = FLAGS.num_epochs
	if config_file and tf.gfile.Exists(config_file):
		# load params from file.
		with tf.gfile.Open(config_file, 'r') as f:
			values_map = json.load(f)
			hparams.override_from_dict(values_map)
	hparams.parse(FLAGS.hparams)

	# The following is for spatial partitioning. `features` has one tensor while
	# `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
	# partition is performed on `features` and all partitionable tensors of
	# `labels`, see the partition logic below.
	# In the TPUEstimator context, the meaning of `shard` and `replica` is the
	# same; follwing the API, here has mixed use of both.
	if FLAGS.use_spatial_partition:
		# Checks input_partition_dims agrees with num_cores_per_replica.
		if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
			raise RuntimeError('--num_cores_per_replica must be a product of array'
												 'elements in --input_partition_dims.')

		labels_partition_dims = {
				'mean_num_positives': None,
				'source_ids': None,
				'groundtruth_data': None,
				'image_scales': None,
		}
		# The Input Partition Logic: We partition only the partition-able tensors.
		# Spatial partition requires that the to-be-partitioned tensors must have a
		# dimension that is a multiple of `partition_dims`. Depending on the
		# `partition_dims` and the `image_size` and the `max_level` in hparams, some
		# high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
		# be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
		# size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
		# [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
		# case, the level-8 and level-9 target tensors are not partition-able, and
		# the highest partition-able level is 7.
		image_size = hparams.get('image_size')
		for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

			def _can_partition(spatial_dim):
				partitionable_index = np.where(
						spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
				return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

			spatial_dim = image_size // (2**level)
			if _can_partition(spatial_dim):
				labels_partition_dims['box_targets_%d' %
															level] = FLAGS.input_partition_dims
				labels_partition_dims['cls_targets_%d' %
															level] = FLAGS.input_partition_dims
			else:
				labels_partition_dims['box_targets_%d' % level] = None
				labels_partition_dims['cls_targets_%d' % level] = None

		num_cores_per_replica = FLAGS.num_cores_per_replica
		input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
		num_shards = FLAGS.num_cores // num_cores_per_replica
	else:
		num_cores_per_replica = None
		input_partition_dims = None
		num_shards = FLAGS.num_cores

	config_proto = tf.ConfigProto(
			allow_soft_placement=True, log_device_placement=False)
	if FLAGS.use_xla and not FLAGS.use_tpu:
		config_proto.graph_options.optimizer_options.global_jit_level = (
				tf.OptimizerOptions.ON_1)
	if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
		config_proto.graph_options.rewrite_options.auto_mixed_precision = (
				rewriter_config_pb2.RewriterConfig.ON)

	# if FLAGS.distribution_strategy is None:
	# Uses TPUEstimator.
	params = dict(
			hparams.values(),
			num_shards=num_shards,
			num_examples_per_epoch=FLAGS.num_examples_per_epoch,
			use_tpu=FLAGS.use_tpu,
			resnet_checkpoint=FLAGS.resnet_checkpoint,
			val_json_file=FLAGS.val_json_file,
			mode=FLAGS.mode,
	)
	tpu_config = tf.contrib.tpu.TPUConfig(
			FLAGS.iterations_per_loop,
			num_shards=num_shards,
			num_cores_per_replica=num_cores_per_replica,
			input_partition_dims=input_partition_dims,
			per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
			.PER_HOST_V2)

	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			evaluation_master=FLAGS.eval_master,
			model_dir=FLAGS.model_dir,
			log_step_count_steps=FLAGS.iterations_per_loop,
			session_config=config_proto,
			tpu_config=tpu_config,
	)

	if FLAGS.model_dir is not None:
		if not tf.gfile.Exists(FLAGS.model_dir):
			tf.gfile.MakeDirs(FLAGS.model_dir)
		with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
			json.dump(hparams.values(), f, sort_keys=True, indent=2)
	tf.logging.info(params)
	# if FLAGS.distribution_strategy is None:
	total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
	train_estimator = tf.contrib.tpu.TPUEstimator(
			model_fn=retinanet_model.tpu_retinanet_model_fn,
			use_tpu=FLAGS.use_tpu,
			train_batch_size=FLAGS.train_batch_size,
			config=run_config,
			params=params)


	tpupoint = TPUPoint(
		estimator=train_estimator, 
		gcp_project = FLAGS.gcp_project,
		tpu_zone = FLAGS.tpu_zone,
		tpu = FLAGS.tpu,
		logdir = FLAGS.model_dir,
		# workers_list = FLAGS.workers_list ,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 
	) 

	bench_start = time.time()
	tpupoint.Start()

	train_estimator.train(
			input_fn=dataloader.InputReader(
					FLAGS.training_file_pattern, is_training=True),
			max_steps=total_steps)


	tpupoint.Stop()
	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()
	tf.logging.info('Finished training')
	return



def retinanet_run_optimize():
	bench_total_start = time.time()
	if FLAGS.use_tpu:
		if FLAGS.distribution_strategy is None:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
					FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
			tpu_grpc_url = tpu_cluster_resolver.get_master()
			tf.Session.reset(tpu_grpc_url)
		else:
			raise RuntimeError(
					'Distribution strategy must be None when --use_tpu is True.')
	else:
		tpu_cluster_resolver = None

	# if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
	#	 raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

	# Check data path
	if(FLAGS.training_file_pattern is None):
		raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('train',
	#									 'train_and_eval') and FLAGS.training_file_pattern is None:
	#	 raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('eval', 'train_and_eval'):
	if FLAGS.validation_file_pattern is None:
		raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')

	# Parse hparams
	hparams = retinanet_model.default_hparams()
	config_file = FLAGS.config_file
	hparams.num_epochs = FLAGS.num_epochs
	if config_file and tf.gfile.Exists(config_file):
		# load params from file.
		with tf.gfile.Open(config_file, 'r') as f:
			values_map = json.load(f)
			hparams.override_from_dict(values_map)
	hparams.parse(FLAGS.hparams)

	# The following is for spatial partitioning. `features` has one tensor while
	# `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
	# partition is performed on `features` and all partitionable tensors of
	# `labels`, see the partition logic below.
	# In the TPUEstimator context, the meaning of `shard` and `replica` is the
	# same; follwing the API, here has mixed use of both.
	if FLAGS.use_spatial_partition:
		# Checks input_partition_dims agrees with num_cores_per_replica.
		if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
			raise RuntimeError('--num_cores_per_replica must be a product of array'
												 'elements in --input_partition_dims.')

		labels_partition_dims = {
				'mean_num_positives': None,
				'source_ids': None,
				'groundtruth_data': None,
				'image_scales': None,
		}
		# The Input Partition Logic: We partition only the partition-able tensors.
		# Spatial partition requires that the to-be-partitioned tensors must have a
		# dimension that is a multiple of `partition_dims`. Depending on the
		# `partition_dims` and the `image_size` and the `max_level` in hparams, some
		# high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
		# be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
		# size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
		# [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
		# case, the level-8 and level-9 target tensors are not partition-able, and
		# the highest partition-able level is 7.
		image_size = hparams.get('image_size')
		for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

			def _can_partition(spatial_dim):
				partitionable_index = np.where(
						spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
				return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

			spatial_dim = image_size // (2**level)
			if _can_partition(spatial_dim):
				labels_partition_dims['box_targets_%d' %
															level] = FLAGS.input_partition_dims
				labels_partition_dims['cls_targets_%d' %
															level] = FLAGS.input_partition_dims
			else:
				labels_partition_dims['box_targets_%d' % level] = None
				labels_partition_dims['cls_targets_%d' % level] = None

		num_cores_per_replica = FLAGS.num_cores_per_replica
		input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
		num_shards = FLAGS.num_cores // num_cores_per_replica
	else:
		num_cores_per_replica = None
		input_partition_dims = None
		num_shards = FLAGS.num_cores

	config_proto = tf.ConfigProto(
			allow_soft_placement=True, log_device_placement=False)
	if FLAGS.use_xla and not FLAGS.use_tpu:
		config_proto.graph_options.optimizer_options.global_jit_level = (
				tf.OptimizerOptions.ON_1)
	if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
		config_proto.graph_options.rewrite_options.auto_mixed_precision = (
				rewriter_config_pb2.RewriterConfig.ON)

	# if FLAGS.distribution_strategy is None:
	# Uses TPUEstimator.
	params = dict(
			hparams.values(),
			num_shards=num_shards,
			num_examples_per_epoch=FLAGS.num_examples_per_epoch,
			use_tpu=FLAGS.use_tpu,
			resnet_checkpoint=FLAGS.resnet_checkpoint,
			val_json_file=FLAGS.val_json_file,
			mode=FLAGS.mode,
	)
	tpu_config = tf.contrib.tpu.TPUConfig(
			FLAGS.iterations_per_loop,
			num_shards=num_shards,
			num_cores_per_replica=num_cores_per_replica,
			input_partition_dims=input_partition_dims,
			per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
			.PER_HOST_V2)

	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			evaluation_master=FLAGS.eval_master,
			model_dir=FLAGS.model_dir,
			log_step_count_steps=FLAGS.iterations_per_loop,
			session_config=config_proto,
			tpu_config=tpu_config,
	)

	if FLAGS.model_dir is not None:
		if not tf.gfile.Exists(FLAGS.model_dir):
			tf.gfile.MakeDirs(FLAGS.model_dir)
		with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
			json.dump(hparams.values(), f, sort_keys=True, indent=2)
	tf.logging.info(params)
	# if FLAGS.distribution_strategy is None:
	total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
	train_estimator = tf.contrib.tpu.TPUEstimator(
			model_fn=retinanet_model.tpu_retinanet_model_fn,
			use_tpu=FLAGS.use_tpu,
			train_batch_size=FLAGS.train_batch_size,
			config=run_config,
			params=params)

	tpupoint = TPUPoint(
		estimator=train_estimator, 
		gcp_project = FLAGS.gcp_project,
		tpu_zone = FLAGS.tpu_zone,
		tpu = FLAGS.tpu,
		logdir = FLAGS.model_dir,
		# workers_list = FLAGS.workers_list ,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 
	) 

	tpupoint.optimize_input_fn( dataloader.InputReader(FLAGS.training_file_pattern, is_training=True) , blocking=True)

	bench_start = time.time()

	# train_estimator.train( input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True), max_steps=total_steps)
	tpupoint.train(estimator=train_estimator, input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True), max_steps=total_steps)


	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()
	tf.logging.info('Finished training')
	return



def retinanet_run_dynamic():
	bench_total_start = time.time()
	if FLAGS.use_tpu:
		if FLAGS.distribution_strategy is None:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
					FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
			tpu_grpc_url = tpu_cluster_resolver.get_master()
			tf.Session.reset(tpu_grpc_url)
		else:
			raise RuntimeError(
					'Distribution strategy must be None when --use_tpu is True.')
	else:
		tpu_cluster_resolver = None

	# if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
	#	 raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

	# Check data path
	if(FLAGS.training_file_pattern is None):
		raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('train',
	#									 'train_and_eval') and FLAGS.training_file_pattern is None:
	#	 raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('eval', 'train_and_eval'):
	if FLAGS.validation_file_pattern is None:
		raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')

	# Parse hparams
	hparams = retinanet_model.default_hparams()
	config_file = FLAGS.config_file
	hparams.num_epochs = FLAGS.num_epochs
	if config_file and tf.gfile.Exists(config_file):
		# load params from file.
		with tf.gfile.Open(config_file, 'r') as f:
			values_map = json.load(f)
			hparams.override_from_dict(values_map)
	hparams.parse(FLAGS.hparams)

	# The following is for spatial partitioning. `features` has one tensor while
	# `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
	# partition is performed on `features` and all partitionable tensors of
	# `labels`, see the partition logic below.
	# In the TPUEstimator context, the meaning of `shard` and `replica` is the
	# same; follwing the API, here has mixed use of both.
	if FLAGS.use_spatial_partition:
		# Checks input_partition_dims agrees with num_cores_per_replica.
		if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
			raise RuntimeError('--num_cores_per_replica must be a product of array'
												 'elements in --input_partition_dims.')

		labels_partition_dims = {
				'mean_num_positives': None,
				'source_ids': None,
				'groundtruth_data': None,
				'image_scales': None,
		}
		# The Input Partition Logic: We partition only the partition-able tensors.
		# Spatial partition requires that the to-be-partitioned tensors must have a
		# dimension that is a multiple of `partition_dims`. Depending on the
		# `partition_dims` and the `image_size` and the `max_level` in hparams, some
		# high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
		# be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
		# size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
		# [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
		# case, the level-8 and level-9 target tensors are not partition-able, and
		# the highest partition-able level is 7.
		image_size = hparams.get('image_size')
		for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

			def _can_partition(spatial_dim):
				partitionable_index = np.where(
						spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
				return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

			spatial_dim = image_size // (2**level)
			if _can_partition(spatial_dim):
				labels_partition_dims['box_targets_%d' %
															level] = FLAGS.input_partition_dims
				labels_partition_dims['cls_targets_%d' %
															level] = FLAGS.input_partition_dims
			else:
				labels_partition_dims['box_targets_%d' % level] = None
				labels_partition_dims['cls_targets_%d' % level] = None

		num_cores_per_replica = FLAGS.num_cores_per_replica
		input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
		num_shards = FLAGS.num_cores // num_cores_per_replica
	else:
		num_cores_per_replica = None
		input_partition_dims = None
		num_shards = FLAGS.num_cores

	config_proto = tf.ConfigProto(
			allow_soft_placement=True, log_device_placement=False)
	if FLAGS.use_xla and not FLAGS.use_tpu:
		config_proto.graph_options.optimizer_options.global_jit_level = (
				tf.OptimizerOptions.ON_1)
	if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
		config_proto.graph_options.rewrite_options.auto_mixed_precision = (
				rewriter_config_pb2.RewriterConfig.ON)

	# if FLAGS.distribution_strategy is None:
	# Uses TPUEstimator.
	params = dict(
			hparams.values(),
			num_shards=num_shards,
			num_examples_per_epoch=FLAGS.num_examples_per_epoch,
			use_tpu=FLAGS.use_tpu,
			resnet_checkpoint=FLAGS.resnet_checkpoint,
			val_json_file=FLAGS.val_json_file,
			mode=FLAGS.mode,
	)
	tpu_config = tf.contrib.tpu.TPUConfig(
			FLAGS.iterations_per_loop,
			num_shards=num_shards,
			num_cores_per_replica=num_cores_per_replica,
			input_partition_dims=input_partition_dims,
			per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
			.PER_HOST_V2)

	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			evaluation_master=FLAGS.eval_master,
			model_dir=FLAGS.model_dir,
			log_step_count_steps=FLAGS.iterations_per_loop,
			session_config=config_proto,
			tpu_config=tpu_config,
	)

	if FLAGS.model_dir is not None:
		if not tf.gfile.Exists(FLAGS.model_dir):
			tf.gfile.MakeDirs(FLAGS.model_dir)
		with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
			json.dump(hparams.values(), f, sort_keys=True, indent=2)
	tf.logging.info(params)
	# if FLAGS.distribution_strategy is None:
	total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
	train_estimator = tf.contrib.tpu.TPUEstimator(
			model_fn=retinanet_model.tpu_retinanet_model_fn,
			use_tpu=FLAGS.use_tpu,
			train_batch_size=FLAGS.train_batch_size,
			config=run_config,
			params=params)

	tpupoint = TPUPoint(
		estimator=train_estimator, 
		gcp_project = FLAGS.gcp_project,
		tpu_zone = FLAGS.tpu_zone,
		tpu = FLAGS.tpu,
		logdir = FLAGS.model_dir,
		# workers_list = FLAGS.workers_list ,
		num_tracing_attempts = 3,
		include_dataset_ops = False, # False for longer traces
		monitoring_level = 1, # 1 or 2 logging level
		num_queries = 4 
	) 

	tpupoint.optimize_input_fn( dataloader.InputReader(FLAGS.training_file_pattern, is_training=True) )

	bench_start = time.time()

	# train_estimator.train( input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True), max_steps=total_steps)
	tpupoint.train_dynamic(model_fn=retinanet_model.tpu_retinanet_model_fn ,estimator=train_estimator, input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True), max_steps=total_steps)


	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tpupoint.CleanUp()
	tf.logging.info('Finished training')
	return



def retinanet_run_squad_naive_wo_tpupoint():
	bench_total_start = time.time()
	if FLAGS.use_tpu:
		if FLAGS.distribution_strategy is None:
			tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
					FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
			tpu_grpc_url = tpu_cluster_resolver.get_master()
			tf.Session.reset(tpu_grpc_url)
		else:
			raise RuntimeError(
					'Distribution strategy must be None when --use_tpu is True.')
	else:
		tpu_cluster_resolver = None

	# if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
	#	 raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

	# Check data path
	if(FLAGS.training_file_pattern is None):
		raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('train',
	#									 'train_and_eval') and FLAGS.training_file_pattern is None:
	#	 raise RuntimeError('You must specify --training_file_pattern for training.')
	# if FLAGS.mode in ('eval', 'train_and_eval'):
	if FLAGS.validation_file_pattern is None:
		raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
	if FLAGS.val_json_file is None:
		raise RuntimeError('You must specify --val_json_file for evaluation.')

	# Parse hparams
	hparams = retinanet_model.default_hparams()
	config_file = FLAGS.config_file
	hparams.num_epochs = FLAGS.num_epochs
	if config_file and tf.gfile.Exists(config_file):
		# load params from file.
		with tf.gfile.Open(config_file, 'r') as f:
			values_map = json.load(f)
			hparams.override_from_dict(values_map)
	hparams.parse(FLAGS.hparams)

	# The following is for spatial partitioning. `features` has one tensor while
	# `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
	# partition is performed on `features` and all partitionable tensors of
	# `labels`, see the partition logic below.
	# In the TPUEstimator context, the meaning of `shard` and `replica` is the
	# same; follwing the API, here has mixed use of both.
	if FLAGS.use_spatial_partition:
		# Checks input_partition_dims agrees with num_cores_per_replica.
		if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
			raise RuntimeError('--num_cores_per_replica must be a product of array'
												 'elements in --input_partition_dims.')

		labels_partition_dims = {
				'mean_num_positives': None,
				'source_ids': None,
				'groundtruth_data': None,
				'image_scales': None,
		}
		# The Input Partition Logic: We partition only the partition-able tensors.
		# Spatial partition requires that the to-be-partitioned tensors must have a
		# dimension that is a multiple of `partition_dims`. Depending on the
		# `partition_dims` and the `image_size` and the `max_level` in hparams, some
		# high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
		# be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
		# size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
		# [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
		# case, the level-8 and level-9 target tensors are not partition-able, and
		# the highest partition-able level is 7.
		image_size = hparams.get('image_size')
		for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

			def _can_partition(spatial_dim):
				partitionable_index = np.where(
						spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
				return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

			spatial_dim = image_size // (2**level)
			if _can_partition(spatial_dim):
				labels_partition_dims['box_targets_%d' %
															level] = FLAGS.input_partition_dims
				labels_partition_dims['cls_targets_%d' %
															level] = FLAGS.input_partition_dims
			else:
				labels_partition_dims['box_targets_%d' % level] = None
				labels_partition_dims['cls_targets_%d' % level] = None

		num_cores_per_replica = FLAGS.num_cores_per_replica
		input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
		num_shards = FLAGS.num_cores // num_cores_per_replica
	else:
		num_cores_per_replica = None
		input_partition_dims = None
		num_shards = FLAGS.num_cores

	config_proto = tf.ConfigProto(
			allow_soft_placement=True, log_device_placement=False)
	if FLAGS.use_xla and not FLAGS.use_tpu:
		config_proto.graph_options.optimizer_options.global_jit_level = (
				tf.OptimizerOptions.ON_1)
	if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
		config_proto.graph_options.rewrite_options.auto_mixed_precision = (
				rewriter_config_pb2.RewriterConfig.ON)

	# if FLAGS.distribution_strategy is None:
	# Uses TPUEstimator.
	params = dict(
			hparams.values(),
			num_shards=num_shards,
			num_examples_per_epoch=FLAGS.num_examples_per_epoch,
			use_tpu=FLAGS.use_tpu,
			resnet_checkpoint=FLAGS.resnet_checkpoint,
			val_json_file=FLAGS.val_json_file,
			mode=FLAGS.mode,
	)
	tpu_config = tf.contrib.tpu.TPUConfig(
			FLAGS.iterations_per_loop,
			num_shards=num_shards,
			num_cores_per_replica=num_cores_per_replica,
			input_partition_dims=input_partition_dims,
			per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
			.PER_HOST_V2)

	run_config = tf.contrib.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			evaluation_master=FLAGS.eval_master,
			model_dir=FLAGS.model_dir,
			log_step_count_steps=FLAGS.iterations_per_loop,
			session_config=config_proto,
			tpu_config=tpu_config,
	)

	if FLAGS.model_dir is not None:
		if not tf.gfile.Exists(FLAGS.model_dir):
			tf.gfile.MakeDirs(FLAGS.model_dir)
		with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
			json.dump(hparams.values(), f, sort_keys=True, indent=2)
	tf.logging.info(params)
	# if FLAGS.distribution_strategy is None:
	total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
	train_estimator = tf.contrib.tpu.TPUEstimator(
			model_fn=retinanet_model.tpu_retinanet_model_fn,
			use_tpu=FLAGS.use_tpu,
			train_batch_size=FLAGS.train_batch_size,
			config=run_config,
			params=params)
	tpupoint = TPUPoint(
    		estimator=train_estimator,
    		gcp_project = FLAGS.gcp_project,
    		tpu_zone = FLAGS.tpu_zone,
    		tpu = FLAGS.tpu,
    		logdir = FLAGS.model_dir,
    		# workers_list = FLAGS.workers_list ,
    		num_tracing_attempts = 3,
    		include_dataset_ops = False, # False for longer traces
    		monitoring_level = 1, # 1 or 2 logging level
    		num_queries = 4 ) 
	bench_start = time.time()
  	tpupoint.Start()
	train_estimator.train(
			input_fn=Naive_InputReader(FLAGS.training_file_pattern, is_training=True),
			max_steps=total_steps)
	tpupoint.Stop()

	bench_elapsed = time.time() - bench_start
	bench_total_dur = time.time() - bench_total_start
	tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
	tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
	tf.logging.info('Finished training')
	return 



def retinanet_run_squad_naive_wo_tpupoint1():
  bench_total_start = time.time()
  if FLAGS.use_tpu:
    if FLAGS.distribution_strategy is None:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      raise RuntimeError(
          'Distribution strategy must be None when --use_tpu is True.')
  else:
    tpu_cluster_resolver = None

  # if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
  #  raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

  # Check data path
  if(FLAGS.training_file_pattern is None):
    raise RuntimeError('You must specify --training_file_pattern for training.')
  # if FLAGS.mode in ('train',
  #                  'train_and_eval') and FLAGS.training_file_pattern is None:
  #  raise RuntimeError('You must specify --training_file_pattern for training.')
  # if FLAGS.mode in ('eval', 'train_and_eval'):
  if FLAGS.validation_file_pattern is None:
    raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
  if FLAGS.val_json_file is None:
    raise RuntimeError('You must specify --val_json_file for evaluation.')

  # Parse hparams
  hparams = retinanet_model.default_hparams()
  config_file = FLAGS.config_file
  hparams.num_epochs = FLAGS.num_epochs
  if config_file and tf.gfile.Exists(config_file):
    # load params from file.
    with tf.gfile.Open(config_file, 'r') as f:
      values_map = json.load(f)
      hparams.override_from_dict(values_map)
  hparams.parse(FLAGS.hparams)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  if FLAGS.use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids': None,
        'groundtruth_data': None,
        'image_scales': None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    # Spatial partition requires that the to-be-partitioned tensors must have a
    # dimension that is a multiple of `partition_dims`. Depending on the
    # `partition_dims` and the `image_size` and the `max_level` in hparams, some
    # high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
    # be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
    # size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
    # [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
    # case, the level-8 and level-9 target tensors are not partition-able, and
    # the highest partition-able level is 7.
    image_size = hparams.get('image_size')
    for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = image_size // (2**level)
      if _can_partition(spatial_dim):
        labels_partition_dims['box_targets_%d' %
                              level] = FLAGS.input_partition_dims
        labels_partition_dims['cls_targets_%d' %
                              level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None

    num_cores_per_replica = FLAGS.num_cores_per_replica
    input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores

  config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  if FLAGS.use_xla and not FLAGS.use_tpu:
    config_proto.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
    config_proto.graph_options.rewrite_options.auto_mixed_precision = (
        rewriter_config_pb2.RewriterConfig.ON)

  # if FLAGS.distribution_strategy is None:
  # Uses TPUEstimator.
  params = dict(
      hparams.values(),
      num_shards=num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      use_tpu=FLAGS.use_tpu,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      mode=FLAGS.mode,
  )
  tpu_config = tf.contrib.tpu.TPUConfig(
      FLAGS.iterations_per_loop,
      num_shards=num_shards,
      num_cores_per_replica=num_cores_per_replica,
      input_partition_dims=input_partition_dims,
      per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
      .PER_HOST_V2)

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      evaluation_master=FLAGS.eval_master,
      model_dir=FLAGS.model_dir,
      log_step_count_steps=FLAGS.iterations_per_loop,
      session_config=config_proto,
      tpu_config=tpu_config,
  )

  if FLAGS.model_dir is not None:
    if not tf.gfile.Exists(FLAGS.model_dir):
      tf.gfile.MakeDirs(FLAGS.model_dir)
    with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
      json.dump(hparams.values(), f, sort_keys=True, indent=2)
  tf.logging.info(params)
  # if FLAGS.distribution_strategy is None:
  total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
  train_estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=retinanet_model.tpu_retinanet_model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.train_batch_size,
      config=run_config,
      params=params)

  tpupoint = TPUPoint(
    estimator=train_estimator, 
    gcp_project = FLAGS.gcp_project,
    tpu_zone = FLAGS.tpu_zone,
    tpu = FLAGS.tpu,
    logdir = FLAGS.model_dir,
    # workers_list = FLAGS.workers_list ,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 
  ) 

  tpupoint.optimize_input_fn( dataloader.InputReader(FLAGS.training_file_pattern, is_training=True) , blocking=True, worst=True)

  bench_start = time.time()

  # train_estimator.train( input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True), max_steps=total_steps)
  tpupoint.train_naive(estimator=train_estimator, input_fn=dataloader.InputReader(FLAGS.training_file_pattern, is_training=True), max_steps=total_steps)


  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()
  tf.logging.info('Finished training')
  tpupoint.CleanUp()
  return



def retinanet_run_squad_naive_w_tpupoint():
  bench_total_start = time.time()
  if FLAGS.use_tpu:
    if FLAGS.distribution_strategy is None:
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
      tpu_grpc_url = tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)
    else:
      raise RuntimeError(
          'Distribution strategy must be None when --use_tpu is True.')
  else:
    tpu_cluster_resolver = None

  # if FLAGS.mode not in ['train', 'eval', 'train_and_eval']:
  #  raise ValueError('Unrecognize --mode: %s' % FLAGS.mode)

  # Check data path
  if(FLAGS.training_file_pattern is None):
    raise RuntimeError('You must specify --training_file_pattern for training.')
  # if FLAGS.mode in ('train',
  #                  'train_and_eval') and FLAGS.training_file_pattern is None:
  #  raise RuntimeError('You must specify --training_file_pattern for training.')
  # if FLAGS.mode in ('eval', 'train_and_eval'):
  if FLAGS.validation_file_pattern is None:
    raise RuntimeError('You must specify --validation_file_pattern for evaluation.')
  if FLAGS.val_json_file is None:
    raise RuntimeError('You must specify --val_json_file for evaluation.')

  # Parse hparams
  hparams = retinanet_model.default_hparams()
  config_file = FLAGS.config_file
  hparams.num_epochs = FLAGS.num_epochs
  if config_file and tf.gfile.Exists(config_file):
    # load params from file.
    with tf.gfile.Open(config_file, 'r') as f:
      values_map = json.load(f)
      hparams.override_from_dict(values_map)
  hparams.parse(FLAGS.hparams)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` had 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # In the TPUEstimator context, the meaning of `shard` and `replica` is the
  # same; follwing the API, here has mixed use of both.
  if FLAGS.use_spatial_partition:
    # Checks input_partition_dims agrees with num_cores_per_replica.
    if FLAGS.num_cores_per_replica != np.prod(FLAGS.input_partition_dims):
      raise RuntimeError('--num_cores_per_replica must be a product of array'
                         'elements in --input_partition_dims.')

    labels_partition_dims = {
        'mean_num_positives': None,
        'source_ids': None,
        'groundtruth_data': None,
        'image_scales': None,
    }
    # The Input Partition Logic: We partition only the partition-able tensors.
    # Spatial partition requires that the to-be-partitioned tensors must have a
    # dimension that is a multiple of `partition_dims`. Depending on the
    # `partition_dims` and the `image_size` and the `max_level` in hparams, some
    # high-level anchor labels (i.e., `cls_targets` and `box_targets`) cannot
    # be partitioned. For example, when `partition_dims` is [1, 4, 2, 1], image
    # size is 1536, `max_level` is 9, `cls_targets_8` has a shape of
    # [batch_size, 6, 6, 9], which cannot be partitioned (6 % 4 != 0). In this
    # case, the level-8 and level-9 target tensors are not partition-able, and
    # the highest partition-able level is 7.
    image_size = hparams.get('image_size')
    for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):

      def _can_partition(spatial_dim):
        partitionable_index = np.where(
            spatial_dim % np.array(FLAGS.input_partition_dims) == 0)
        return len(partitionable_index[0]) == len(FLAGS.input_partition_dims)

      spatial_dim = image_size // (2**level)
      if _can_partition(spatial_dim):
        labels_partition_dims['box_targets_%d' %
                              level] = FLAGS.input_partition_dims
        labels_partition_dims['cls_targets_%d' %
                              level] = FLAGS.input_partition_dims
      else:
        labels_partition_dims['box_targets_%d' % level] = None
        labels_partition_dims['cls_targets_%d' % level] = None

    num_cores_per_replica = FLAGS.num_cores_per_replica
    input_partition_dims = [FLAGS.input_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores

  config_proto = tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False)
  if FLAGS.use_xla and not FLAGS.use_tpu:
    config_proto.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  if FLAGS.auto_mixed_precision and FLAGS.distribution_strategy:
    config_proto.graph_options.rewrite_options.auto_mixed_precision = (
        rewriter_config_pb2.RewriterConfig.ON)

  # if FLAGS.distribution_strategy is None:
  # Uses TPUEstimator.
  params = dict(
      hparams.values(),
      num_shards=num_shards,
      num_examples_per_epoch=FLAGS.num_examples_per_epoch,
      use_tpu=FLAGS.use_tpu,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      mode=FLAGS.mode,
  )
  tpu_config = tf.contrib.tpu.TPUConfig(
      FLAGS.iterations_per_loop,
      num_shards=num_shards,
      num_cores_per_replica=num_cores_per_replica,
      input_partition_dims=input_partition_dims,
      per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
      .PER_HOST_V2)

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      evaluation_master=FLAGS.eval_master,
      model_dir=FLAGS.model_dir,
      log_step_count_steps=FLAGS.iterations_per_loop,
      session_config=config_proto,
      tpu_config=tpu_config,
  )

  if FLAGS.model_dir is not None:
    if not tf.gfile.Exists(FLAGS.model_dir):
      tf.gfile.MakeDirs(FLAGS.model_dir)
    with tf.gfile.Open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
      json.dump(hparams.values(), f, sort_keys=True, indent=2)
  tf.logging.info(params)
  # if FLAGS.distribution_strategy is None:
  total_steps = int((FLAGS.num_epochs * FLAGS.num_examples_per_epoch) / FLAGS.train_batch_size)
  train_estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=retinanet_model.tpu_retinanet_model_fn,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.train_batch_size,
      config=run_config,
      params=params)

  tpupoint = TPUPoint(
    estimator=train_estimator, 
    gcp_project = FLAGS.gcp_project,
    tpu_zone = FLAGS.tpu_zone,
    tpu = FLAGS.tpu,
    logdir = FLAGS.model_dir,
    # workers_list = FLAGS.workers_list ,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 
  ) 

  tpupoint.optimize_input_fn( Naive_InputReader(FLAGS.training_file_pattern, is_training=True) , blocking=False, worst=False)
  naive_params = {
    'auto_vectorize_map': ['False'],
    'auto_prefetch_buffer_size': ['Autotune', 'Autotune'],
    'auto_map_parallel': ['Autotune', 2],
    'auto_map_and_batch':['FALSE'],
    'auto_model_fn': tpupoint.autoadjustclass.train_test_model_fn,
  }

  # tpupoint.autoadjustclass.train_params_results[float('inf')] = tpupoint.autoadjustclass.GetAutoParams(**naive_params)
  # train_input = tpupoint.autoadjustclass.GetWorstModifiedDataset
  train_input = Naive_InputReader(FLAGS.training_file_pattern, is_training=True)

  bench_start = time.time()
  
  # train_metrics = train_estimator.train( input_fn=train_input, max_steps=total_steps )

  # train_estimator.train(
  #     input_fn=Naive_InputReader(FLAGS.training_file_pattern, is_training=True),
  #     max_steps=total_steps)

  tpupoint.train_dynamic(
    model_fn=retinanet_model.tpu_retinanet_model_fn ,
    estimator=train_estimator, 
    input_fn=train_input,
    max_steps=total_steps)

  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tf.logging.info('Finished training')
  tpupoint.CleanUp()
  return 
