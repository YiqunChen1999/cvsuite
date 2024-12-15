conda create -n cvsuite python=3.11 -y \
&& conda activate cvsuite \
&& conda install -c conda-forge opencv -y \
&& conda install -c conda-forge matplotlib -y \
&& conda install -c conda-forge scikit-image -y \
&& conda install -c conda-forge scikit-learn -y \
&& pip install -e .