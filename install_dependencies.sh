#!/usr/bin/env bash
set -euo pipefail

# This was tested on Ubuntu 18.04 and 22.04
sudo apt update
sudo apt install build-essential --yes wget curl gfortran git ncurses-dev libncursesw5-dev unzip tar libxcb-xinerama0

################
# This part is ONLY needed if you'll work with the original Human3.6M files.
# For this, we need to install the [CDF library](https://cdf.gsfc.nasa.gov/) since the annotations are in CDF files.
# We will use [SpacePy](https://spacepy.github.io/) as a wrapper, which in turn depends on this CDF library.
CDF_VERSION=39_0
wget "https://spdf.gsfc.nasa.gov/pub/software/cdf/dist/cdf${CDF_VERSION}/linux/cdf${CDF_VERSION}-dist-cdf.tar.gz"
tar xf "cdf${CDF_VERSION}-dist-cdf.tar.gz"
rm "cdf${CDF_VERSION}-dist-cdf.tar.gz"
pushd "cdf${CDF_VERSION}-dist"
make OS=linux ENV=gnu CURSES=no FORTRAN=no UCOPTIONS=-O2 SHARED=yes -j"$(nproc)" all
# If you have sudo rights, simply run `sudo make install`. If you have no `sudo` rights, set the env var
# `CDF_LIB` to `cdf${CDF_VERSION}-dist/src/lib`
# add the export line to ~/.bashrc for permanent effect, or use GNU Stow.
# mv src ~/.local/stow/cdf-${CDF_VERSION}
# stow cdf-${CDF_VERSION}
# The following will work temporarily:
export CDF_LIB=$PWD/src/lib
popd
####################

# Micromamba is the simplest way to install the dependencies
# If you don't have it yet, install it as follows:
export MAMBA_ROOT_PREFIX=$HOME/micromamba
mkdir -p $MAMBA_ROOT_PREFIX
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C $MAMBA_ROOT_PREFIX bin/micromamba
# Execute and add to ~/.bashrc the following two lines
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s posix)"
#

# Create a new environment and install the dependencies
envsubst < environment_comfortable_py10.yml > env_subst.yml
micromamba env create --name=nlf --file=env_subst.yml -y
micromamba activate nlf
pip install --no-build-isolation git+https://github.com/spacepy/spacepy


micromamba install -y -c conda-forge \
    cachetools \
    cython \
    ezc3d \
    ffmpeg \
    imageio \
    matplotlib \
    mkl \
    trimesh \
    numba \
    "numpy<2.0" \
    libiconv \
    pandas \
    pillow \
    scikit-image \
    scikit-learn \
    scikit-sparse \
    tqdm \
    conda-forge::mayavi>=4.8 \
    conda-forge::PySide6==6.7.1 \
    conda-forge::ffmpeg-python

pip install \
    setuptools \
    addict \
    tensorflow==2.15 \
    tensorflow-hub \
    torch \
    torchvision \
    torchdata \
    chumpy \
    embreex \
    einops \
    imageio-ffmpeg \
    importlib_resources \
    jpeg4py \
    more_itertools \
    opencv-python \
    pyrender \
    tetgen \
    pymeshfix


# Optional:
# Install libjpeg-turbo for faster JPEG decoding.
# wget https://sourceforge.net/projects/libjpeg-turbo/files/2.0.5/libjpeg-turbo-2.0.5.tar.gz
# Then compile it.
# Or use the repo:
# git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
# cd libjpeg-turbo
# PACKAGE_NAME=libjpeg-turbo
# TARGET=$HOME/.local
# sudo apt install nasm
# cmake -DCMAKE_INSTALL_PREFIX="$TARGET"  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -G"Unix Makefiles" .
# TEMP_DESTDIR=$(mktemp --directory --tmpdir="$STOW_DIR")
# make -j "$(nproc)" install DESTDIR="$TEMP_DESTDIR"
# mv -T "$TEMP_DESTDIR/$TARGET" "$STOW_DIR/$PACKAGE_NAME"
# rm -rf "$TEMP_DESTDIR"
# stow "$PACKAGE_NAME" --target="$TARGET"

 - posepile image barecat
 - anno barecats (4)
 - code projects
 - micromamba, env install
 - cuda, cudnn copy
 - set up project initializer bashrc command that sets envvars, activates env, cd to project
 - wacv23_models
 - stuff from $DATA_ROOT/cache
 - projects/localizerfields


 #sudo apt-get install libavformat-dev libavdevice-dev
#pip install av --no-binary av
# https://stackoverflow.com/questions/72604912/cant-show-image-with-opencv-when-importing-av