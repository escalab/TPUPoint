"""
SingleProfile.py
crated by ESCAL Lab at the University of California Riverside
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from absl import app
import os
import subprocess
import sys
from distutils.version import LooseVersion
import tensorflow as tf

from tensorflow.core.profiler import profiler_analysis_pb2
from tensorflow.python.eager import profiler_client
# from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors


FLAGS = flags.FLAGS


# Tool specific parameters
flags.DEFINE_string(
    'service_addr', None, 'Address of TPU profiler service e.g. '
    'localhost:8466, you must specify either this flag or --tpu.')
flags.DEFINE_string(
    'logdir', None, 'Path of TensorBoard log directory e.g. /tmp/tb_log, '
    'gs://tb_bucket')
flags.DEFINE_integer('duration_ms', 0,
                     'Duration of tracing or monitoring in ms.')
flags.DEFINE_string(
    'workers_list', None, 'The list of worker TPUs that we are about to profile'
    ' e.g. 10.0.1.2, 10.0.1.3. You can specify this flag with --tpu or '
    '--service_addr to profile a subset of tpu nodes. You can also use only'
    '--tpu and leave this flag unspecified to profile all the tpus.')
flags.DEFINE_boolean('include_dataset_ops', True,
                     'Set to false to profile longer TPU '
                     'device traces.')
flags.DEFINE_integer(
    'num_tracing_attempts', 3, 'Automatically retry N times when no trace '
    'event is collected.')


def ProfileSubProcess(service_addr, path, duration_ms, workers_list, include_dataset_ops, num_tracing_attempts):
  try:
    profiler_client.start_tracing(
      service_addr=service_addr, 
      logdir=path, 
      duration_ms=duration_ms, 
      worker_list=workers_list,
      include_dataset_ops=include_dataset_ops, 
      num_tracing_attempts=num_tracing_attempts)
  except errors.UnavailableError:
    exit(1)
  except:
    exit(-1)

  # # pywrap_tensorflow.TFE_ProfilerClientStartTracing(service_addr, path, workers_list, include_dataset_ops, duration_ms, num_tracing_attempts)
  # except:
  #   print("Failed")
  #   exit(1)
  return


def main(unused_argv=None):
	service_addr = FLAGS.service_addr
	path = FLAGS.logdir
	duration_ms = FLAGS.duration_ms
	workers_list = FLAGS.workers_list
	include_dataset_ops = FLAGS.include_dataset_ops
	num_tracing_attempts = FLAGS.num_tracing_attempts
	ProfileSubProcess(service_addr, path, duration_ms, workers_list, include_dataset_ops, num_tracing_attempts)


def run_main():
  app.run(main)


if __name__ == '__main__':
	run_main()