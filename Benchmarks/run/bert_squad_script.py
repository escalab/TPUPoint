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

from run_squad import *


def Naive_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

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


  def slow_mapped_function(*args):
    # Do some hard pre-processing
    tf.py_func(lambda: time.sleep(0.05), [], ())
    return args

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
    # d = d.map( lambda x: slow_mapped_function(x) , num_parallel_calls=1 )
    d = d.batch( batch_size, drop_remainder=drop_remainder )

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    d = d.with_options(options)


    return d

  return input_fn



def bert_run_squad_baseline():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples

  train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  bench_start = time.time()
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def bert_run_squad_eval():
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  
  eval_examples = read_squad_examples(
      input_file=FLAGS.predict_file, is_training=False)

  eval_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
      is_training=False)
  eval_features = []

  def append_feature(feature):
    eval_features.append(feature)
    eval_writer.process_feature(feature)

  convert_examples_to_features(
      examples=eval_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=False,
      output_fn=append_feature)
  eval_writer.close()

  tf.logging.info("***** Running predictions *****")
  tf.logging.info(" Num orig examples = %d", len(eval_examples))
  tf.logging.info(" Num split examples = %d", len(eval_features))
  tf.logging.info(" Batch size = %d", FLAGS.predict_batch_size)

  all_results = []

  predict_input_fn = input_fn_builder(
      input_file=eval_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

  # If running eval on the TPU, you will need to specify the number of
  # steps.
  all_results = []
  for result in estimator.predict(predict_input_fn, yield_single_examples=True):
    if len(all_results) % 1000 == 0:
      tf.logging.info("Processing example: %d" % (len(all_results)))
    unique_id = int(result["unique_ids"])
    start_logits = [float(x) for x in result["start_logits"].flat]
    end_logits = [float(x) for x in result["end_logits"].flat]
    all_results.append(
        RawResult(
            unique_id=unique_id,
            start_logits=start_logits,
            end_logits=end_logits))

  output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
  output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
  output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")


  write_predictions(eval_examples, eval_features, all_results,
                    FLAGS.n_best_size, FLAGS.max_answer_length,
                    FLAGS.do_lower_case, output_prediction_file,
                    output_nbest_file, output_null_log_odds_file)



def bert_run_squad_profile():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.bench_gcp_project,
    tpu_zone=FLAGS.bench_tpu_zone,    
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  bench_start = time.time()
  tpupoint.Start()
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()



def bert_run_squad_optimize():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.bench_gcp_project,
    tpu_zone=FLAGS.bench_tpu_zone,    
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)


  tpupoint.optimize_input_fn(train_input_fn, blocking=True)

  bench_start = time.time()
  
  # # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()



def bert_run_squad_dynamic():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.bench_gcp_project,
    tpu_zone=FLAGS.bench_tpu_zone,    
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)


  tpupoint.optimize_input_fn(train_input_fn)

  bench_start = time.time()
  
  # # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train_dynamic(model_fn=model_fn,estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()



def bert_run_squad_naive_wo_tpupoint():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples
  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.bench_gcp_project,
    tpu_zone=FLAGS.bench_tpu_zone,    
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 
  # train_input_fn = input_fn_builder(
  train_input_fn = Naive_input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  bench_start = time.time()
  tpupoint.Start()
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.Stop()
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")



def bert_run_squad_naive_wo_tpupoint1():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.bench_gcp_project,
    tpu_zone=FLAGS.bench_tpu_zone,    
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)


  tpupoint.optimize_input_fn(train_input_fn, blocking=True, worst=True)

  bench_start = time.time()
  
  # # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train_naive(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()



def bert_run_squad_naive_w_tpupoint():
  bench_total_start = time.time()
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

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
  if FLAGS.do_train:
    train_examples = read_squad_examples(
        input_file=FLAGS.train_file, is_training=True)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    # Pre-shuffle the input to avoid having to make a very large shuffle
    # buffer in in the `input_fn`.
    rng = random.Random(12345)
    rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
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
      predict_batch_size=FLAGS.predict_batch_size)

  # if FLAGS.do_train:
  # We write to a temporary file to avoid storing very large constant tensors
  # in memory.
  train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
  convert_examples_to_features(
      examples=train_examples,
      tokenizer=tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      doc_stride=FLAGS.doc_stride,
      max_query_length=FLAGS.max_query_length,
      is_training=True,
      output_fn=train_writer.process_feature)
  train_writer.close()

  tf.logging.info("***** Running training *****")
  tf.logging.info(" Num orig examples = %d", len(train_examples))
  tf.logging.info(" Num split examples = %d", train_writer.num_features)
  tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
  tf.logging.info(" Num steps = %d", num_train_steps)
  del train_examples

  tpupoint = TPUPoint( 
    estimator = estimator,
    gcp_project=FLAGS.bench_gcp_project,
    tpu_zone=FLAGS.bench_tpu_zone,    
    tpu=FLAGS.tpu_name,
    logdir=FLAGS.output_dir,
    workers_list = None,
    num_tracing_attempts = 3,
    include_dataset_ops = False, # False for longer traces
    monitoring_level = 1, # 1 or 2 logging level
    num_queries = 4 ) 

  # train_input_fn = input_fn_builder(
  #     input_file=train_writer.filename,
  #     seq_length=FLAGS.max_seq_length,
  #     is_training=True,
  #     drop_remainder=True)
  train_input_fn = Naive_input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

  tpupoint.optimize_input_fn(train_input_fn, blocking=False, worst=False)

  bench_start = time.time()
  
  # # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  # tpupoint.train_naive(estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  tpupoint.train_dynamic(model_fn=model_fn,estimator=estimator, input_fn=train_input_fn, max_steps=num_train_steps)
  
  bench_elapsed = time.time() - bench_start
  bench_total_dur = time.time() - bench_total_start
  tf.logging.info("Train End-to-End: " + str(bench_elapsed) + " seconds")
  tf.logging.info("Total End-to-End: " + str(bench_total_dur) + " seconds")
  tpupoint.CleanUp()


