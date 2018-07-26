Setup GPU environment to execute CUDA codes with pyCUDA
=======================================================

Enable a module environment or install on your local computer python version 3.6 (and install virtualenv with ``pip install --user virtualenv``), then::

  $ module load python/3.6.4 gnu/6.1.0
  $ virtualenv ~/.venvs/boppyenv/
  $ . ~/.venvs/boppyenv/bin/activate

Download from pypi pycuda version 2017.1.1 (https://pypi.org/project/pycuda/#files) or from github (https://github.com/inducer/pycuda/releases)::

  $ wget https://files.pythonhosted.org/packages/b3/30/9e1c0a4c10e90b4c59ca7aa3c518e96f37aabcac73ffe6b5d9658f6ef843/pycuda-2017.1.1.tar.gz

Install the requirements (don't worry if it fails on pycuda, we're gonna install it right after)::

  $ pip install -r boppy/requirements.txt
  $ pip install -r boppy/requirements_gpu.txt

And finally::

  $ pip install pycuda-2017.1.1.tar.gz
