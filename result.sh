#!/bin/bash
com=./common.h
function one_time(){
        gcc --std=c99 host.c -lm -fopenmp -o host `dpu-pkg-config --cflags --libs dpu`
        dpu-upmem-dpurte-clang  -DNR_TASKLETS=16 -o decoder decoder.c
        ./host > result

}
one_time


