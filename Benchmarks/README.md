# Benchmarks


All benchmarks are intended to run on the Google Cloud Platform (GCP) with a single TPU.
You can start by creating a new project using the [GCP Console](https://console.cloud.google.com).

You will need to know the following about your project before beginning: 
- GCP project name (`gcp_project_name`)
- GCP bucket name (`gs://bucket_name`)
- TPU version (`TPUv2` or `TPUv3`)
- TPU name (`tpu_name`)
- TPU zone (`tpu_zone`)


**NOTE:** Make sure to clone using `git clone --recurse-submodules` to also include all submodules required.

## Download Datasets

To download the required datasets, follow the commands below.

**NOTE:** You will have to download the [`COCO`](https://cocodataset.org/) and [`ImageNet`](http://www.image-net.org/)
dataset on your own as this script does not supply those.
If no `ImageNet` dataset is provided, you may use `gs://cloud-tpu-test-datasets/fake_imagenet` instead for input.  


```bash
### MOVE INTO THE dataset DIRECTORY
cd datasets

### DOWNLOAD REQUIRED DATASETS DATASET
python main.py --data_dir='gs://bucket_name' \
--mnist=True \
--SQUAD=True \
--BERT=True \
--CIFAR10=True \
--ML20M=True \
--ML1M=True \
--ResNetCheckpoint=True \
--InstallPycocotools=True ;
```

For more information, run `python main.py --help`


## Running Benchmarks

Below, you can run each benchmark individually for either TPUv2 or TPUv3.


```bash
### MOVE INTO THE run DIRECTORY
cd run

### RUN TPUv2

python main.py \
--bench_data_location='gs://bucket_name' \
--bench_model_location='gs://bucket_name' \
--bench_tpu_version="TPUv2" \
--bench_tpu_name="tpu_name" \
--bench_tpu_zone="tpu_zone" \
--bench_gcp_project="gcp_project_name" \
--bench_coco_dir="gs://bucket_name/COCO/" \
--bench_imagenet_dir="gs://bucket_name/ImageNet/" \
--bench_use_tpu=True \
--bench_tpupoint_baseline=True \
--bench_tpupoint_profile=True \
--bench_tpupoint_optimize=True \
--bench_run_eval=False \
--BERT=True \
--DCGAN=True \
--QaNET=True \
--ResNet=True \
--RetinaNet=True \
--QaNET_Small=True \
--RetinaNet_Small=True;

### RUN TPUv3

python main.py \
--bench_data_location='gs://bucket_name' \
--bench_model_location='gs://bucket_name' \
--bench_tpu_version="TPUv3" \
--bench_tpu_name="tpu_name" \
--bench_tpu_zone="tpu_zone" \
--bench_gcp_project="gcp_project_name" \
--bench_coco_dir="gs://bucket_name/COCO/" \
--bench_imagenet_dir="gs://bucket_name/ImageNet/" \
--bench_use_tpu=True \
--bench_tpupoint_baseline=True \
--bench_tpupoint_profile=True \
--bench_tpupoint_optimize=True \
--bench_run_eval=False \
--BERT=True \
--DCGAN=True \
--QaNET=True \
--ResNet=True \
--RetinaNet=True \
--QaNET_Small=True \
--RetinaNet_Small=True;

```

For more information, run `python main.py --help`
