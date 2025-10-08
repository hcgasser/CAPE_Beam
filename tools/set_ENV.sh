#!/usr/bin/env bash

export PYTHONPATH="${PF}/libs"
export PATH="${PF}/${REPO//_/-}:${PF}/tools:${PATH}"

export OUTPUT_DIR_PATH=/${REPO}/artefacts/CAPE-Beam/designs                   # set output directory path
mkdir -p $OUTPUT_DIR_PATH
mkdir -p ${PF}/data/input/proteomes
