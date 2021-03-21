```
________________ _____  __________        _____         _____ 
___  __/___  __ \__  / / /___  __ \______ ___(_)_______ __  /_
__  /   __  /_/ /_  / / / __  /_/ /_  __ \__  / __  __ \_  __/
_  /    _  ____/ / /_/ /  _  ____/ / /_/ /_  /  _  / / // /_  
/_/     /_/      \____/   /_/      \____/ /_/   /_/ /_/ \__/  
```


TPUPoint is a tool facilitate the development of efficient applications on TPU-based cloud platforms. 
TPUPoint automatically classifies repetitive patterns into phases and identifies the most timing-critical operations in each phase.
TPUPoint is built into TensorFlow 1.15.3.

**NOTE:** Make sure to clone using `git clone --recurse-submodules` to also include all submodules required.

## Install

To install TensorFlow with TPUPoint, please go to the `TensorflowTPUPoint` directory and follow the `README.md`.

**Note:** TPUPoint is built to run on [Google Cloud Platform (GCP)](https://console.cloud.google.com).
If you have not already done so, please create a project with TPUv2-8 or TPUv3-8 access. 

## Benchmarks

To run TPUPoint with pre-built benchmarks, please go to the `Benchmarks` directory and follow the `README.md`.

**Note:** These benchmarks assume you have already installed TensorFlow with TPUPoint.
