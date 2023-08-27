# PARSE
The source code of PARSE

Considering issues such as authorization, we do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. You should first build benchmark dataset according to https://github.com/YerevaNN/mimic3-benchmarks/. Please save the files in in-hospital-mortality directory to data/ directory, then shuffle and devide by yourself.

Env:

python=3.6.9=h265db76_0

pytorch=1.4.0=py3.6_cuda10.0.130_cudnn7.6.3_0

numpy=1.17.2=py36haad9e8e_0
