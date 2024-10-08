#!/usr/bin/env bash
set -euo pipefail

NPROC=${NPROC:?'env var must be set to number of available CPUs.'}
PIP_VER="23.2.1"

echo 1. Prepare the CloudViewer-ML repo and install dependencies
echo
export PATH_TO_CLOUDVIEWER_ML="$PWD"
echo "$PATH_TO_CLOUDVIEWER_ML"
# the build system of the main repo expects a main branch. make sure main exists
git checkout -b main || true
python -m pip install -U pip==$PIP_VER
python -m pip install -r requirements.txt \
    -r requirements-torch.txt
# -r requirements-tensorflow.txt # TF disabled on Linux (ACloudViewer PR#6288)
# -r requirements-openvino.txt # Numpy version conflict with TF 2.8.2
cd ..
python -m pip install -U Cython

echo 2. clone ACloudViewer and install dependencies
echo
git clone --branch main https://github.com/Asher-1/ACloudViewer.git

./ACloudViewer/util/install_deps_ubuntu.sh assume-yes
python -m pip install -r ACloudViewer/python/requirements.txt \
    -r ACloudViewer/python/requirements_style.txt \
    -r ACloudViewer/python/requirements_test.txt

echo 3. Configure for bundling the CloudViewer-ML part
echo
mkdir ACloudViewer/build
pushd ACloudViewer/build
# TF disabled on Linux (ACloudViewer PR#6288)
cmake -DBUNDLE_CLOUDVIEWER_ML=ON \
    -DCLOUDVIEWER_ML_ROOT="${PATH_TO_CLOUDVIEWER_ML}" \
    -DGLIBCXX_USE_CXX11_ABI=OFF \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_GUI=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXAMPLES=OFF \
    ..

echo 4. Build and install wheel
echo
make -j"$NPROC" install-pip-package

echo 5. run examples/tests in the CloudViewer-ML repo outside of the repo directory to
echo make sure that the installed package works.
echo
popd
mkdir test_workdir
pushd test_workdir
mv "$PATH_TO_CLOUDVIEWER_ML/tests" .
echo Add --randomly-seed=SEED to the test command to reproduce test order.
python -m pytest tests

echo "... now do the same but in dev mode by setting CLOUDVIEWER_ML_ROOT"
echo
export CLOUDVIEWER_ML_ROOT="$PATH_TO_CLOUDVIEWER_ML"
echo Add --randomly-seed=SEED to the test command to reproduce test order.
python -m pytest tests
unset CLOUDVIEWER_ML_ROOT

popd
