FROM jairhul/centos7-geant4v10.7.2.3-jai-environment:latest

# Downloading BDSIM
RUN git clone https://bitbucket.org/jairhul/bdsim /bdsim

# Use devtoolset-7 for all subsequent commands
SHELL ["/usr/bin/scl", "enable", "devtoolset-7", "--", "/bin/bash", "-c"]

# Installing BDSIM
WORKDIR /bdsim-build
RUN cmake ../bdsim && make -j$(nproc) && make install

# Installing necessary development libraries for Python 3.12
RUN yum install -y bzip2-devel libffi-devel zlib-devel readline-devel sqlite-devel \
    gdbm-devel db4-devel wget
WORKDIR /usr/src
RUN wget https://www.openssl.org/source/openssl-1.1.1l.tar.gz && \
    tar xzf openssl-1.1.1l.tar.gz && \
    cd openssl-1.1.1l && \
    ./config --prefix=/usr/local/openssl --openssldir=/usr/local/openssl && \
    make -j$(nproc) && \
    make install

# Downloading Python 3.12
WORKDIR /usr/src/python-build
RUN wget https://www.python.org/ftp/python/3.12.4/Python-3.12.4.tgz && \
    tar xzf Python-3.12.4.tgz 

# Building and installing Python 3.12
WORKDIR /usr/src/python-build/Python-3.12.4
ENV LD_LIBRARY_PATH=/usr/local/openssl/lib:$LD_LIBRARY_PATH
RUN ./configure --enable-optimizations --with-openssl=/usr/local/openssl && \
    make -j$(nproc) && \
    make install

# Installing python packages
WORKDIR /
COPY ./requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

# Cleaning up source files used for building
WORKDIR /usr/src
RUN rm -rf *

# Set the working directory to the root
WORKDIR /