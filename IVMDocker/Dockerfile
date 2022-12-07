# syntax=docker/dockerfile:1
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
# install app dependencies
RUN apt-get update 
RUN apt-get install -y python3 python3-pip git cmake wget curl zlib1g pkg-config libhdf5-openmpi-dev libc6-dev libnuma-dev doxygen libyaml-cpp-dev gfortran libtbb-dev
RUN cd / && git clone https://gitlab.lrz.de/exahype/ExaHyPE-Engine.git
# WORKDIR /ExaHyPE-Engine

RUN cd /ExaHyPE-Engine && ./Submodules/updateSubmodules.sh

RUN cd /home && mkdir tbb-2020 && cd tbb-2020 && wget https://github.com/oneapi-src/oneTBB/releases/download/v2020.0/tbb-2020.0-lin.tgz && tar zxvf tbb-2020.0-lin.tgz
ENV TBB_INC=-I/home/tbb-2020/tbb/include
ENV TBB_SHLIB='-L/home/tbb-2020/tbb/lib/intel64/gcc4.8 -ltbb' 
ENV COMPILER=GNU

COPY ./IVM_Feng /ExaHyPE-Engine/ApplicationExamples/IVM_Feng/

RUN cd / && git clone https://github.com/UM-Bridge/umbridge.git

ENV SHARED_DIR=/shared
RUN mkdir /output /shared
RUN cp ./ExaHyPE-Engine/ApplicationExamples/IVM_Feng/CopyCode/in.txt /shared/

RUN ./ExaHyPE-Engine/Toolkit/toolkit.sh ./ExaHyPE-Engine/ApplicationExamples/IVM_Feng/IVM_ADERDG_IO_User2.exahype
RUN cp ./ExaHyPE-Engine/ApplicationExamples/IVM_Feng/CopyCode/IVM_ADERDG.cpp ../ExaHyPE-Engine/ApplicationExamples/IVM_Feng/IO_User2/
RUN cp ./ExaHyPE-Engine/ApplicationExamples/IVM_Feng/CopyCode/IVM_ADERDG.h ../ExaHyPE-Engine/ApplicationExamples/IVM_Feng/IO_User2/
RUN cp ./ExaHyPE-Engine/ApplicationExamples/IVM_Feng/CopyCode/PointsSol.cpp ../ExaHyPE-Engine/ApplicationExamples/IVM_Feng/IO_User2/Writers/
RUN cp ./ExaHyPE-Engine/ApplicationExamples/IVM_Feng/CopyCode/PointsSol.h ../ExaHyPE-Engine/ApplicationExamples/IVM_Feng/IO_User2/Writers/
WORKDIR /ExaHyPE-Engine/ApplicationExamples/IVM_Feng/
RUN ls
RUN bash run.sh

# Build model server
WORKDIR / 
COPY ./server.cpp server/
RUN cd /server && \
    g++ -o server server.cpp -I /umbridge/lib -lpthread

RUN apt-get -y install vim

ENV PORT=4244
ENV RANKS=1
CMD /server/server
