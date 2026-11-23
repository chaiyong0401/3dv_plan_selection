# rr_atnorm_20261123 amendment (written before any result)
First start failed at model construction: the AT class moves itself to CUDA in its constructor ("No CUDA GPUs are available" with the GPU hidden).
Change: the local RTX 4070 (idle) is left visible so the constructor can run; the script is unchanged and still moves the AT to CPU and computes on CPU.
No number had been produced. Script sha256 unchanged (frozen.sha256).
