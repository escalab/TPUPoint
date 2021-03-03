# TPUPoint

You will need to know the following about your project before beginning: 
- GCP project name (`gcp_project_name`)
- GCP bucket name (`gs://bucket_name` a.k.a. `model_dir`)
- TPU version (`TPUv2` or `TPUv3`)
- TPU name (`tpu_name`)
- TPU zone (`tpu_zone`)


## Requirements

To use TPUPoint, you will require [Bazel](https://bazel.build/) and the following python packages

```bash
pip install pandas --user;
pip install matplotlib
pip install -U --user pip numpy wheel
pip install -U --user keras_preprocessing --no-deps
pip install pyyaml
```

## BUILD


```bash
# Configure the build
./configure

## Build for CPU/TPU
bazel build -c opt --define=grpc_no_ares=true  //tensorflow/tools/pip_package:build_pip_package

## Run the Bazel build
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

## Uninstall the original tensorflow already Installed
pip uninstall tensorflow

pip install /tmp/tensorflow_pkg/tensorflow-package-name.whl
```

## Usage

TPUPoint can be used to either profile an entire model training, or used to optimize a model dynamically.

```python
import tensorflow as tf
from tensorflow.contrib.tpu import TPUPoint as TP
#...

def main(argv):
	#...
	estimator = tf.contrib.tpu.TPUEstimator(...)
	#...
	tpupoint = TP( 
		estimator = estimator,
		gcp_project=gcp_project_name,
		tpu_zone=tpu_zone,
		tpu=tpu_name,
		logdir=model_dir )

	# Profiling Only
	tpprofiler.Start(analyzer = true)
	estimator.train(...)
	tpprofiler.Stop()


	# Dynamic Optimization
	tpprofiler.Start(analyzer = false)
	tpupoint.train_dynamic(...)
	tpprofiler.Stop()

	# Upload any TPUPoint files to GCP model bucket directory
	tpupoint.CleanUp()

```
