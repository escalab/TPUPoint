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

sys.path.append( os.path.join( os.getcwd() , '..', 'bert' ) )

from run_classifier import *


def Naive_file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    # d = d.apply(
    #     tf.contrib.data.map_and_batch(
    #         lambda record: _decode_record(record, name_to_features),
    #         batch_size=batch_size,
    #         drop_remainder=drop_remainder))
    d = d.map( lambda record: _decode_record(record, name_to_features) , num_parallel_calls=1 )
    d = d.batch( batch_size, drop_remainder=drop_remainder )

    return d

  return input_fn


def bert_run_baseline():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  bench_start = time.time()
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")


def bert_run_eval():
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  #   train_examples = processor.get_train_examples(FLAGS.data_dir)
  #   num_train_steps = int(
  #       len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  #   num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  #   train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  #   file_based_convert_examples_to_features(
  #       train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  #   tf.logging.info("***** Running training *****")
  #   tf.logging.info("  Num examples = %d", len(train_examples))
  #   tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  #   tf.logging.info("  Num steps = %d", num_train_steps)
  #   train_input_fn = file_based_input_fn_builder(
  #       input_file=train_file,
  #       seq_length=FLAGS.max_seq_length,
  #       is_training=True,
  #       drop_remainder=True)
  #   estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  # if FLAGS.do_eval:
  eval_examples = processor.get_dev_examples(FLAGS.data_dir)
  num_actual_eval_examples = len(eval_examples)
  if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on. These do NOT count towards the metric (all tf.metrics
    # support a per-instance weight, and these get a weight of 0.0).
    while len(eval_examples) % FLAGS.eval_batch_size != 0:
      eval_examples.append(PaddingInputExample())

  eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
  file_based_convert_examples_to_features(
      eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

  tf.logging.info("***** Running evaluation *****")
  tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                  len(eval_examples), num_actual_eval_examples,
                  len(eval_examples) - num_actual_eval_examples)
  tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

  # This tells the estimator to run through the entire set.
  eval_steps = None
  # However, if running eval on the TPU, you will need to specify the
  # number of steps.
  if FLAGS.use_tpu:
    assert len(eval_examples) % FLAGS.eval_batch_size == 0
    eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

  eval_drop_remainder = True if FLAGS.use_tpu else False
  eval_input_fn = file_based_input_fn_builder(
      input_file=eval_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=eval_drop_remainder)

  result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

  output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
  with tf.gfile.GFile(output_eval_file, "w") as writer:
    tf.logging.info("***** Eval results *****")
    for key in sorted(result.keys()):
      tf.logging.info("  %s = %s", key, str(result[key]))
      writer.write("%s = %s\n" % (key, str(result[key])))


def bert_run_profile():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  
  bench_start = time.time()
  tpupoint.Start()
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()


def bert_run_optimize():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  
  tpupoint.optimize_input_fn(train_input_fn, blocking=True)
  bench_start = time.time()
  # tpupoint.Start()
  # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()


def bert_run_dynamic():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  
  tpupoint.optimize_input_fn(train_input_fn)
  bench_start = time.time()
  # tpupoint.Start()
  # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train_dynamic(model_fn=model_fn, estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()


def bert_run_naive_wo_tpupoint():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  # train_input_fn = file_based_input_fn_builder(
  train_input_fn = Naive_file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  bench_start = time.time()
  tpupoint.Start()
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")


def bert_run_naive_wo_tpupoint1():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  
  tpupoint.optimize_input_fn(train_input_fn, blocking=True, worst=True)
  bench_start = time.time()
  # tpupoint.Start()
  # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train_naive(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()



def bert_run_naive_w_tpupoint():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "mnli": MnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
  #   raise ValueError(
  #       "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # if FLAGS.do_train:
  train_examples = processor.get_train_examples(FLAGS.data_dir)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
  file_based_convert_examples_to_features(
      train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  # train_input_fn = file_based_input_fn_builder(
  #     input_file=train_file,
  #     seq_length=FLAGS.max_seq_length,
  #     is_training=True,
  #     drop_remainder=True)
  train_input_fn = Naive_file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.gcp_project,
    tpu_zone=FLAGS.tpu_zone,
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  
  # tpupoint.optimize_input_fn(train_input_fn, blocking=True, worst=False)
  tpupoint.optimize_input_fn(train_input_fn, blocking=False, worst=False)
  bench_start = time.time()
  # tpupoint.Start()
  # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.train(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.train_naive(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train_dynamic(model_fn=model_fn ,estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()







