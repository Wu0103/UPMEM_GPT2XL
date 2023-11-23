#ifndef __HOST_H_
#define __HOST_H_
 
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include "common.h"
#include "define.h"

void Softmax(){
    int length;
	if(which_res==0) length = sequence_length;
	else length = 1;
    long double sum = 0;
    float max = finalout[(length-1)*target_predefined];
    float soft[target_predefined];
    long double expsave[target_predefined];
    for(int i=0;i<target_predefined;i++){
		if(finalout[(length-1)*target_predefined + i]>max) max = finalout[(length-1)*target_predefined + i];
	}
    for(int i=0;i<target_predefined;i++){
        expsave[i] = exp(finalout[(length-1)*target_predefined + i]-max);
        sum += expsave[i];
    }
    for(int i=0;i<target_predefined;i++){
        soft[i] = expsave[i]/sum;
    }
    memcpy(&finalout[(length-1)*target_predefined],soft,sizeof(T)*target_predefined);
    return;
}

void softmax_decoder() {
    int length;
    if(which_res==0) length = sequence_length;
	else length = 1;
	memset(sum,0,sizeof(long double)*sequence_length*multihead);
	//memset(max,-1e9,sizeof(long double)*sequence_length*multihead);
    long double expsave[multihead][sequence_length*Round(total_length,NR_DPUS)];
	# pragma omp parallel for
	for(int head=0;head<multihead;head++){
		for(int r=0;r<length;r++){
            max[r+head*length] = CQK_decoder[head*length*(sequence_length+which_res)+r*(sequence_length+which_res)]/sqrt(d_weight);
			for(int i=0;i<sequence_length+which_res;i++){
				CQK_decoder[head*length*(sequence_length+which_res)+r*(sequence_length+which_res)+i] /= sqrt(d_weight);
				if(CQK_decoder[head*length*(sequence_length+which_res)+r*(sequence_length+which_res)+i]>max[r+head*length]) max[r+head*length] = CQK_decoder[head*length*(sequence_length+which_res)+r*(sequence_length+which_res)+i];
			}
		}

		for(int r=0;r<length;r++){
			for(int i=0;i<sequence_length+which_res;i++){
				expsave[head][r*(sequence_length+which_res)+i] = exp(CQK_decoder[head*length*(sequence_length+which_res)+r*(sequence_length+which_res)+i]-max[r+head*length])/sqrt(d_weight);
				sum[r+head*length] += expsave[head][r*(sequence_length+which_res)+i];
            }
		}

		long double temp[length*(sequence_length+which_res)];
        for(int r=0;r<length;r++){
			for(int i=0;i<sequence_length+which_res;i++){
                CQK_dpu_decoder[head][r*(sequence_length+which_res)+i] = (float)(expsave[head][r*(sequence_length+which_res)+i]/sum[r+head*length]);
			}
		}

	}
}

void transpose_decoder(){
	# pragma omp parallel for
	for(int head = 0;head <multihead;head++){
		for(int j=0;j<sequence_length+which_res;j++){
			for(int n=0;n<d_weight;n++){
                CTV_decoder[head*total_length*d_weight+n*(sequence_length+which_res)+j] = V_cache[head][layers_index][j*d_weight+n];
			}
		}
	}
}

void layernormal_decoder(T* normal_before,T* normal_after,bool final,bool first){
	long double* mean = (long double*)malloc(total_length * sizeof(long double));
	memset(mean,0,total_length*sizeof(long double));
	long double* std = (long double*)malloc(total_length * sizeof(long double));
	memset(std,0,total_length*sizeof(long double));
    long double* Temp = (long double*)malloc(sequence_length * sizeof(long double) * d_model);
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;
	double A = 1.0;
	double B = 0.0;
	double eps = 1e-5;
	# pragma omp parallel for
	for(int j=0;j<(length);j++){
		for(int x=0;x<d_model;x++){
			mean[j] += normal_before[j*d_model+x];
		}
	}
	# pragma omp parallel for
	for(int j=0;j<(length);j++){
		mean[j] /= d_model;		
	}
	# pragma omp parallel for
	for(int j=0;j<(length);j++){
		for(int x=0;x<d_model;x++){
			std[j] += (normal_before[j*d_model+x]-mean[j])*(normal_before[j*d_model+x]-mean[j]);
		}
	}
	# pragma omp parallel for
	for(int j=0;j<(length);j++){
		std[j] = sqrt((std[j]/d_model)+eps);
	}
	# pragma omp parallel for
	for(int j=0;j<(length);j++){
		for(int x=0;x<d_model;x++){
            //Temp[j*d_model+x] = A * (normal_before[j*d_model+x] - mean[j]) / (std[j]) + B;
			normal_after[j*d_model+x] = A * (normal_before[j*d_model+x] - mean[j]) / (std[j]) + B;
		}
	}
	free(mean);
	free(std);
    free(Temp);
}

void gelu(T* middle){
    int length;
	if(which_res==0) length = sequence_length;
	else length = 1;
    double prefix = sqrt(2/3.14159265358979323846);
    # pragma omp parallel for
    for(int i=0;i<length;i++){
        for(int j=0;j<h_predefined;j++){
            double inner = middle[i*h_predefined+j] + 0.044715 * pow(middle[i*h_predefined+j],3);
            double beforetanh = prefix * inner;
            double aftertanh = tanh(beforetanh);
            double outer = middle[i*h_predefined+j] * (1 + aftertanh);
            //float value = 0.5 * middle[i*h_predefined+j] * (1+tanh(prefix * (middle[i*h_predefined+j] + pow(middle[i*h_predefined+j],3))));
            middle[i*h_predefined+j] = 0.5 * outer;
        }
    }
    return;
}


void stage1_decoder(){
	input_args[0].kernel = 0;
	input_args[0].layerindex = layers_index;
	input_args[0].which = which_res;
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;

    if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
    layernormal_decoder(input_before_norm,IM_decoder,false,true);
    if(which_res==0) {
		gettimeofday(&sumcome, NULL);
    	sumattencom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
    	genattencom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "mramI", 0, IM_decoder, d_model * length * sizeof(T), DPU_XFER_DEFAULT));
    }
	for(int head_index=0;head_index<multihead;head_index++){
        DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "DPU_INPUT_ARGUMENTS", 0 , input_args, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
    }
	if(which_res==0){
		gettimeofday(&sumcpudpue, NULL);
    	sumattencpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
    	genattencpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
    if(which_res==0){
		gettimeofday(&sumkernele, NULL);
    	sumattenkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
    }else{
		gettimeofday(&genkernele, NULL);
    	genattenkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}

#if PRINT
	int each=0;
	for(int head_index=0;head_index<1;head_index++){
	    DPU_FOREACH(dpu_set_decoder[head_index], dpu,each) if(each==0)DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

#if info
	uint32_t count[multihead];
	uint32_t clocks_per_sec[multihead];
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
		DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count[head_index], sizeof(uint32_t)));}
		DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec[head_index],sizeof(uint32_t)));}
  		if(inst_cycle)printf("DPU insts: %u\n", count[head_index]);
		else{printf("DPU cycles: %u\n", count[head_index]);printf("DPU time: %.2e secs.\n", (double)count[head_index] / clocks_per_sec[head_index]);}
    }
#endif

	if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, CK_dpu_decoder[head_index] + i * Round(length * w_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramK", 0, Round(length * w_dpu,2) * sizeof(T), DPU_XFER_DEFAULT));
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, CQ_dpu_decoder[head_index] + i * Round(length * w_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramQ", 0, Round(length * w_dpu,2) * sizeof(T), DPU_XFER_DEFAULT));
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, CV_dpu_decoder[head_index] + i * Round(length * w_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramV", 0, Round(length * w_dpu,2) * sizeof(T), DPU_XFER_DEFAULT));
	}
    if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
    	sumattendpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
		genattendpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}

    int offset=0;
    if(which_res>0) offset = sequence_length + which_res - 1;
    else offset = 0; 

	if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems,NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUS;i++){
		for(int head_index=0;head_index<multihead;head_index++){
			for(int s_index=0;s_index<length;s_index++){
				memcpy(&CK_decoder[head_index*total_length*d_weight+(s_index+offset)*d_weight+i*w_dpu],CK_dpu_decoder[head_index]+s_index*w_dpu+i*Round(length * w_dpu,2),sizeof(T)*w_dpu);
				memcpy(&CQ_decoder[head_index*total_length*d_weight+(s_index)*d_weight+i*w_dpu],CQ_dpu_decoder[head_index]+s_index*w_dpu+i*Round(length * w_dpu,2),sizeof(T)*w_dpu);
				memcpy(&CV_decoder[head_index*total_length*d_weight+(s_index+offset)*d_weight+i*w_dpu],CV_dpu_decoder[head_index]+s_index*w_dpu+i*Round(length * w_dpu,2),sizeof(T)*w_dpu);
			}
		}
	}
    # pragma omp parallel for
    for(int i=0;i<NR_DPUS;i++){
		for(int head_index=0;head_index<multihead;head_index++){
            memcpy(&K_cache[head_index][layers_index][d_weight*offset],&CK_decoder[head_index*total_length*d_weight+offset*d_weight],sizeof(T)*d_weight*length);
            memmove(&V_cache[head_index][layers_index][d_weight*offset],&CV_decoder[head_index*total_length*d_weight+offset*d_weight],sizeof(T)*d_weight*length);
        }
    }
    if(which_res==0){
		gettimeofday(&summeme, NULL);
    	sumattenmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
    	genattenmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}
}

void stage2_decoder(){
	int left = ((sequence_length+which_res)%NR_DPUS);
	int each = ((sequence_length+which_res)/NR_DPUS);
    int length;
	if(which_res==0) length = sequence_length;
	else length = 1;
	for(int i=0;i<NR_DPUS;i++) {
		input_args[i].kernel = 1;
		input_args[i].layerindex = layers_index;
		input_args[i].which = which_res;
		if(left && i<left)		input_args[i].rows = each+1;
		else	input_args[i].rows = each;
	}
	int offset;
	if(left!=0) offset = each;
	else offset = each - 1;

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		if(each==0){
			DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, K_cache[head_index][layers_index]+(i%(sequence_length+which_res))*d_weight));
        	DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "SaveK", 0, d_weight * sizeof(T), DPU_XFER_DEFAULT));
		}else{
			int iter = 0;
			if(left!=0) each++;
			while(each!=0){
                each--;
				DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, K_cache[head_index][layers_index]+((i+iter*NR_DPUS)%(sequence_length+which_res))*d_weight));
	        	DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "SaveK", iter*d_weight*sizeof(T), d_weight * sizeof(T), DPU_XFER_DEFAULT));
				iter++;
			}
		}
		DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "FullQ", 0, CQ_decoder + head_index*total_length*d_weight, d_weight * length * sizeof(T), DPU_XFER_DEFAULT));
        DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
        DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
    }
	if(which_res==0){
		gettimeofday(&sumcpudpue, NULL);
	    sumattencpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
	    genattencpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
	if(which_res==0){
		gettimeofday(&sumkernele, NULL);
	    sumattenkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
	}else{
		gettimeofday(&genkernele, NULL);
	    genattenkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}

#if PRINT
if(which_res==0 && layers_index==0){
	for(int head_index=0;head_index<1;head_index++){
	    //DPU_FOREACH(dpu_set_decoder[head_index], dpu,each) if(each==0)DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
}
#endif

#if info
	uint32_t count[multihead];
	uint32_t clocks_per_sec[multihead];
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
		DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count[head_index], sizeof(uint32_t)));}
		DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec[head_index],sizeof(uint32_t)));}
  		if(inst_cycle)printf("DPU insts: %u\n", count[head_index]);
		else{printf("DPU cycles: %u\n", count[head_index]);printf("DPU time: %.2e secs.\n", (double)count[head_index] / clocks_per_sec[head_index]);}
    }
#endif

	left = ((sequence_length+which_res)%NR_DPUS);
	each = ((sequence_length+which_res)/NR_DPUS);
	int s_dpu = (Round((sequence_length+which_res),NR_DPUS)/NR_DPUS);

	if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, CQK_dpu_decoder[head_index] + i*Round(length*s_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramQK", 0, Round(length * s_dpu * sizeof(T),8), DPU_XFER_DEFAULT));
	}
	if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
	    sumattendpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
	    genattendpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}

    if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems, NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUS;i++){
		for(int head_index=0;head_index<multihead;head_index++){
			for(int s_index=0;s_index<length;s_index++){
				for(int j=0;j<s_dpu;j++){
					if(left>0 && (j==s_dpu-1 && i>=left)) break;
					else memcpy(&CQK_decoder[j*NR_DPUS+i+s_index*(sequence_length+which_res)+head_index*length*(sequence_length+which_res)],CQK_dpu_decoder[head_index]+i*Round(length*s_dpu,2)+s_index*s_dpu+j,sizeof(T));
                }
			}
		}
	}
	transpose_decoder();
	if(which_res==0){
		gettimeofday(&summeme, NULL);
	    sumattenmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
	    genattenmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
	softmax_decoder();
    if(which_res==0){
		gettimeofday(&sumcome, NULL);
	    sumattencom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
	    genattencom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}
}

void stage3_decoder(){
	input_args[0].kernel = 2;
	input_args[0].layerindex = layers_index;
	input_args[0].which = which_res;
	input_args[0].rows = w_dpu;
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) DPU_ASSERT(dpu_prepare_xfer(dpu, CQK_dpu_decoder[head_index]));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "mramQK", 0,Round(sizeof(T) * length * (sequence_length+which_res),8), DPU_XFER_DEFAULT));
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) DPU_ASSERT(dpu_prepare_xfer(dpu, CTV_decoder + head_index * d_weight * total_length + i * w_dpu * (sequence_length+which_res)));
        DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "SaveV", 0, Round(w_dpu * sizeof(T) * (sequence_length+which_res),8), DPU_XFER_DEFAULT));
        DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "DPU_INPUT_ARGUMENTS", 0 , input_args, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
	}
    if(which_res==0){
		gettimeofday(&sumcpudpue, NULL);
    	sumattencpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
    	genattencpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
	if(which_res==0){
		gettimeofday(&sumkernele, NULL);
    	sumattenkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
	}else{
		gettimeofday(&genkernele, NULL);
    	genattenkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}


#if PRINT
	int each = 0;
	for(int head_index=0;head_index<1;head_index++){
	    DPU_FOREACH(dpu_set_decoder[head_index], dpu,each) if(each==0) DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

#if info
	uint32_t count[multihead];
	uint32_t clocks_per_sec[multihead];
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
		DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count[head_index], sizeof(uint32_t)));}
		DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec[head_index],sizeof(uint32_t)));}
  		if(inst_cycle)printf("DPU insts: %u\n", count[head_index]);
		else{printf("DPU cycles: %u\n", count[head_index]);printf("DPU time: %.2e secs.\n", (double)count[head_index] / clocks_per_sec[head_index]);}
    }
#endif


	if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
        DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, CR_dpu_decoder[head_index] + i * Round(length * w_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramV", 0, Round(length * w_dpu * sizeof(T),8), DPU_XFER_DEFAULT));
	}
	if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
    	sumattendpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
    	genattendpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}


	if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems, NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUS;i++){
		for(int head_index=0;head_index<multihead;head_index++){
			for(int s_index=0;s_index<length;s_index++){
				memcpy(&CR_decoder[i*w_dpu+head_index*d_weight+s_index*d_model],CR_dpu_decoder[head_index]+s_index*w_dpu+i*Round(length*w_dpu,2),w_dpu*sizeof(T));
			}
		}
	}
	if(which_res==0){
		gettimeofday(&summeme, NULL);
    	sumattenmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
    	genattenmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}
}

void stage4_decoder(){
	input_args[0].kernel = 3;
	input_args[0].layerindex = layers_index;
	input_args[0].which = which_res;
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
		DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "FullQ", 0 , CR_decoder, length*d_model*sizeof(T), DPU_XFER_DEFAULT));
	    DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "DPU_INPUT_ARGUMENTS", 0 , input_args, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
	}
	if(which_res==0){
		gettimeofday(&sumcpudpue, NULL);
    	sumattencpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
    	genattencpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
	if(which_res==0){
		gettimeofday(&sumkernele, NULL);
    	sumattenkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
	}else{
		gettimeofday(&genkernele, NULL);
    	genattenkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}


#if PRINT
int each = 0;
	for(int head_index=0;head_index<1;head_index++){
	//    DPU_FOREACH(dpu_set_decoder[head_index], dpu,each) if(each==0)DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
    }
#endif

#if info
	uint32_t count[multihead];
	uint32_t clocks_per_sec[multihead];
	for(int head_index=0;head_index<multihead;head_index++){
	//    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
	//	DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count[head_index], sizeof(uint32_t)));}
	//	DPU_FOREACH(dpu_set_decoder[head_index], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec[head_index],sizeof(uint32_t)));}
  		if(inst_cycle)printf("DPU insts: %u\n", count[head_index]);
		else{printf("DPU cycles: %u\n", count[head_index]);printf("DPU time: %.2e secs.\n", (double)count[head_index] / clocks_per_sec[head_index]);}
    }
#endif

	if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
        DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) 	DPU_ASSERT(dpu_prepare_xfer(dpu, CO_dpu_decoder[head_index] + i * Round(length * m_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramO", 0, Round(length * m_dpu * sizeof(T),8), DPU_XFER_DEFAULT));
	}
    if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
    	sumattendpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
    	genattendpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}

	if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems, NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUS;i++){
		for(int head_index=0;head_index<multihead;head_index++){
			for(int s_index=0;s_index<(length);s_index++){
				memcpy(&CO_decoder[i*m_dpu+s_index*d_model+head_index*(d_model/multihead)],CO_dpu_decoder[head_index]+i*Round(length*m_dpu,2)+s_index*m_dpu,m_dpu*sizeof(T));
			}
		}
	}
	if(which_res==0){
		gettimeofday(&summeme, NULL);
    	sumattenmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
    	genattenmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
	# pragma omp parallel for
	for(int s_index=0;s_index<(length);s_index++){
		for(int i=0;i<d_model;i++){
			CO_decoder[s_index*d_model+i] += input_before_norm[s_index*d_model+i];///////////改改改
		}	
	}

	layernormal_decoder(CO_decoder,COnorm_decoder,false,false);

    if(which_res==0){
		gettimeofday(&sumcome, NULL);
    	sumattencom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
    	genattencom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}
}

void stage5_decoder(){
	input_args[0].kernel = 4;
	input_args[0].layerindex = layers_index;
	input_args[0].which = which_res;
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "mramI", 0 , COnorm_decoder, length*d_model*sizeof(T), DPU_XFER_DEFAULT));
	    DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "DPU_INPUT_ARGUMENTS", 0 , input_args, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
	}
	if(which_res==0) {
		gettimeofday(&sumcpudpue, NULL);
	    sumffncpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
	    genffncpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
	if(which_res==0){
		gettimeofday(&sumkernele, NULL);
	    sumffnkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
	}else{
		gettimeofday(&genkernele, NULL);
	    genffnkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}

#if PRINT
	DPU_FOREACH(dpu_set_decoder[0], dpu) DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
#endif

#if info
	uint32_t count;
	uint32_t clocks_per_sec;
    //DPU_ASSERT(dpu_sync(dpu_set_fcdecode[0]));
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count, sizeof(uint32_t)));}
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec,sizeof(uint32_t)));}
  	if(inst_cycle)printf("DPU insts: %u\n", count);
	else{printf("DPU cycles: %u\n", count);printf("DPU time: %.2e secs.\n", (double)count / clocks_per_sec);}
#endif


    if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) DPU_ASSERT(dpu_prepare_xfer(dpu, FCm_dpu_decoder + i*(length)*h_dpu + (length)*h_dpu*NR_DPUS*head_index));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramO1", 0,Round((length)*sizeof(T)*h_dpu,8), DPU_XFER_DEFAULT));
	}
    if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
	    sumffndpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
	    genffndpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}

    if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems, NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUSFC;i++){
		for(int s_index=0;s_index<(length);s_index++){
			memcpy(&FCm_decoder[s_index*h_predefined+i*h_dpu],&FCm_dpu_decoder[s_index*h_dpu+i*(length)*h_dpu],h_dpu*sizeof(T));
		}
	}
	if(which_res==0){
		gettimeofday(&summeme, NULL);
	    sumffnmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
	    genffnmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}

    if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
	gelu(FCm_decoder);

    if(which_res==0){
		gettimeofday(&sumcome, NULL);
	    sumffncom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
	    genffncom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}
}

void stage6_decoder(){
	input_args[0].kernel = 5;
	input_args[0].layerindex = layers_index;
	input_args[0].which = which_res;
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
		DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "mramO1", 0 , FCm_decoder, length*h_predefined*sizeof(T), DPU_XFER_DEFAULT));
	    DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "DPU_INPUT_ARGUMENTS", 0 , input_args, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
	}
    if(which_res==0) {
		gettimeofday(&sumcpudpue, NULL);
	    sumffncpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
	    genffncpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

    if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
	for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
    if(which_res==0){
		gettimeofday(&sumkernele, NULL);
	    sumffnkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
	}else{
		gettimeofday(&genkernele, NULL);
	    genffnkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}

#if PRINT
//	DPU_FOREACH(dpu_set_fcdecode[0], dpu) DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
#endif

#if info
	uint32_t count;
	uint32_t clocks_per_sec;
    //DPU_ASSERT(dpu_sync(dpu_set_fcdecode[0]));
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count, sizeof(uint32_t)));}
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec,sizeof(uint32_t)));}
  	if(inst_cycle)printf("DPU insts: %u\n", count);
	else{printf("DPU cycles: %u\n", count);printf("DPU time: %.2e secs.\n", (double)count / clocks_per_sec);}
#endif


    if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) DPU_ASSERT(dpu_prepare_xfer(dpu, FC_dpu_decoder + i*(length)*mfc_dpu + NR_DPUS*(length)*mfc_dpu*head_index));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramO2", 0,Round((length) * sizeof(T) * mfc_dpu,8), DPU_XFER_DEFAULT));
	}
	if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
	    sumffndpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
	    genffndpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}

    if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems, NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUSFC;i++){
		for(int s_index=0;s_index<(length);s_index++){
			memcpy(&FC_decoder[s_index*d_model+i*mfc_dpu],&FC_dpu_decoder[s_index*mfc_dpu+i*(length)*mfc_dpu],mfc_dpu*sizeof(T));
		}
	}
    #pragma omp barrier
	if(which_res==0){
		gettimeofday(&summeme, NULL);
	    sumffnmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
	    genffnmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
	# pragma omp parallel for
	for(int s_index=0;s_index<(length);s_index++){
		for(int i=0;i<d_model;i++){
			FC_decoder[s_index*d_model+i] += CO_decoder[s_index*d_model+i];
		}	
	}
	if(which_res==0){
		gettimeofday(&sumcome, NULL);
	    sumffncom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
	    genffncom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}

    memcpy(input_before_norm,FC_decoder,sizeof(T)*length*d_model);

}

void stagefinal_decoder(){
	input_args[0].kernel = 6;
	input_args[0].layerindex = layers_index;
	input_args[0].which = which_res;
	int length;
	if(which_res==0) length = sequence_length;
	else length = 1;

	if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
    layernormal_decoder(FC_decoder,IM_decoder,true,false);
	if(which_res==0){
		gettimeofday(&sumcome, NULL);
	    sumfinalcom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
	    genfinalcom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}

    if(which_res==0) gettimeofday(&sumcpudpus, NULL);
	else gettimeofday(&gencpudpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "mramO2", 0 , IM_decoder, length*d_model*sizeof(T), DPU_XFER_DEFAULT));
	    DPU_ASSERT(dpu_broadcast_to(dpu_set_decoder[head_index], "DPU_INPUT_ARGUMENTS", 0 , input_args, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
	}
    if(which_res==0) {
		gettimeofday(&sumcpudpue, NULL);
	    sumfinalcpudpu += get_runtime(sumcpudpus.tv_sec, sumcpudpus.tv_usec, sumcpudpue.tv_sec, sumcpudpue.tv_usec);
	}else{
		gettimeofday(&gencpudpue, NULL);
	    genfinalcpudpu += get_runtime(gencpudpus.tv_sec, gencpudpus.tv_usec, gencpudpue.tv_sec, gencpudpue.tv_usec);
	}

    if(which_res==0) gettimeofday(&sumkernels, NULL);
	else gettimeofday(&genkernels, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_launch(dpu_set_decoder[head_index], DPU_ASYNCHRONOUS));
    }
    for(int head_index=0;head_index<multihead;head_index++){
	    DPU_ASSERT(dpu_sync(dpu_set_decoder[head_index]));
    }
	if(which_res==0){
		gettimeofday(&sumkernele, NULL);
	    sumfinalkernel += get_runtime(sumkernels.tv_sec, sumkernels.tv_usec, sumkernele.tv_sec, sumkernele.tv_usec);
	}else{
		gettimeofday(&genkernele, NULL);
	    genfinalkernel += get_runtime(genkernels.tv_sec, genkernels.tv_usec, genkernele.tv_sec, genkernele.tv_usec);
	}

#if PRINT
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
#endif

#if info
	uint32_t count;
	uint32_t clocks_per_sec;
    //DPU_ASSERT(dpu_sync(dpu_set_fcdecode[0]));
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "count", 0, &count, sizeof(uint32_t)));}
	//DPU_FOREACH(dpu_set_fcdecode[0], dpu) {DPU_ASSERT(dpu_copy_from(dpu, "CLOCKS_PER_SEC", 0, &clocks_per_sec,sizeof(uint32_t)));}
  	if(inst_cycle)printf("DPU insts: %u\n", count);
	else{printf("DPU cycles: %u\n", count);printf("DPU time: %.2e secs.\n", (double)count / clocks_per_sec);}
#endif

    if(which_res==0) gettimeofday(&sumdpucpus, NULL);
	else gettimeofday(&gendpucpus, NULL);
    for(int head_index=0;head_index<multihead;head_index++){
		DPU_FOREACH(dpu_set_decoder[head_index], dpu, i) DPU_ASSERT(dpu_prepare_xfer(dpu, finalout_dpu + i*Round(length*t_dpu,2) + head_index*NR_DPUS*Round(length*t_dpu,2)));
		DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_FROM_DPU, "mramO3", 0,Round(length*t_dpu,2)*sizeof(T), DPU_XFER_DEFAULT));
	}
	if(which_res==0){
		gettimeofday(&sumdpucpue, NULL);
	    sumfinaldpucpu += get_runtime(sumdpucpus.tv_sec, sumdpucpus.tv_usec, sumdpucpue.tv_sec, sumdpucpue.tv_usec);
	}else{
		gettimeofday(&gendpucpue, NULL);
	    genfinaldpucpu += get_runtime(gendpucpus.tv_sec, gendpucpus.tv_usec, gendpucpue.tv_sec, gendpucpue.tv_usec);
	}

    if(which_res==0) gettimeofday(&summems, NULL);
	else gettimeofday(&genmems, NULL);
	# pragma omp parallel for
	for(int i=0;i<NR_DPUSFC;i++){
		for(int s_index=length-1;s_index<(length);s_index++){
			memcpy(&finalout[s_index*target_predefined+i*t_dpu],&finalout_dpu[s_index*t_dpu+i*Round(length*t_dpu,2)],t_dpu*sizeof(T));//////有问题
		}
	}
	if(which_res==0){
		gettimeofday(&summeme, NULL);
	    sumfinalmem += get_runtime(summems.tv_sec, summems.tv_usec, summeme.tv_sec, summeme.tv_usec);
	}else{
		gettimeofday(&genmeme, NULL);
	    genfinalmem += get_runtime(genmems.tv_sec, genmems.tv_usec, genmeme.tv_sec, genmeme.tv_usec);
	}

	if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
	Softmax();
	if(which_res==0){
		gettimeofday(&sumcome, NULL);
	    sumfinalcom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
	    genfinalcom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}

	T max = -1e9;
	T temp = -1e9;
	int sec = 0;
	pos = 0;

	if(which_res==0) gettimeofday(&sumcoms, NULL);
	else gettimeofday(&gencoms, NULL);
	for(int i=0;i<target_predefined;i++){
		if(max<finalout[(length-1)*target_predefined + i]){
            temp = max;
            sec = pos;
			max = finalout[(length-1)*target_predefined + i];
    		pos = i;
		}
	}
	if(which_res==0){
		gettimeofday(&sumcome, NULL);
	    sumfinalcom += get_runtime(sumcoms.tv_sec, sumcoms.tv_usec, sumcome.tv_sec, sumcome.tv_usec);
	}else{
		gettimeofday(&gencome, NULL);
	    genfinalcom += get_runtime(gencoms.tv_sec, gencoms.tv_usec, gencome.tv_sec, gencome.tv_usec);
	}

}

void alloc_decoder(){
	input_args = (dpu_arguments_t *) malloc(NR_DPUS * sizeof(dpu_arguments_t));
	IM_decoder = malloc(sequence_length * d_model * sizeof(T));
	CK_decoder = malloc(multihead * total_length * sizeof(T) * d_weight);
	CQ_decoder = malloc(multihead * total_length * sizeof(T) * d_weight);
	CV_decoder = malloc(multihead * total_length * sizeof(T) * d_weight);
	CQK_decoder = malloc(multihead * sequence_length * sizeof(T) * Round(total_length,NR_DPUS));
	CTV_decoder = malloc(multihead * total_length * sizeof(T) * d_weight);
	CR_decoder = malloc(sequence_length * sizeof(T) * d_weight * multihead);
	CO_decoder = malloc(total_length * sizeof(T) * d_model);
	COnorm_decoder = malloc(total_length * sizeof(T) * d_model);
	for(int head_index=0;head_index<multihead;head_index++){
		CK_dpu_decoder[head_index] = malloc(sequence_length * sizeof(T) * Round(w_dpu,2) * NR_DPUS);
		CQ_dpu_decoder[head_index] = malloc(sequence_length * sizeof(T) * Round(w_dpu,2) * NR_DPUS);
		CV_dpu_decoder[head_index] = malloc(total_length * sizeof(T) * Round(w_dpu,2) * NR_DPUS);
		CQK_dpu_decoder[head_index] = malloc(sequence_length * sizeof(T) * Round(total_length,NR_DPUS));
		CR_dpu_decoder[head_index] = malloc(Round(sequence_length,2) * sizeof(T) * d_weight);
		CO_dpu_decoder[head_index] = malloc(total_length * d_model * sizeof(T));
        for(int i=0;i<layer;i++){
            K_cache[head_index][i] = malloc(sizeof(T) * d_weight * total_length);
            V_cache[head_index][i] = malloc(sizeof(T) * d_weight * total_length);
        }
	}
	sum = (long double*) malloc(sizeof(long double)*sequence_length*multihead);
    max = (long double*) malloc(sizeof(long double)*sequence_length*multihead);

	FCm_decoder = malloc(total_length * sizeof(T) * h_predefined);
	FC_decoder = malloc(total_length * sizeof(T) * d_model);
	FCm_dpu_decoder = malloc(total_length * sizeof(T) * h_predefined);
	FC_dpu_decoder = malloc(total_length * sizeof(T) * d_model);
	finalout_dpu = malloc(sequence_length * Round(target_predefined,NR_DPUSFC) * sizeof(T));
	finalout = malloc(sequence_length * target_predefined * sizeof(T));
}

void dealloc_decoder(){
	free(input_args);
	free(IM_decoder);
	free(CK_decoder);
	free(CQ_decoder);
	free(CV_decoder);
	free(CQK_decoder);
	free(CTV_decoder);
	free(CR_decoder);
	free(CO_decoder);
	free(COnorm_decoder);
	for(int head_index=0;head_index<multihead;head_index++){
		free(CK_dpu_decoder[head_index]);
		free(CQ_dpu_decoder[head_index]);
		free(CV_dpu_decoder[head_index]);
		free(CQK_dpu_decoder[head_index]);
		free(CR_dpu_decoder[head_index]);
		free(CO_dpu_decoder[head_index]);
        for(int i=0;i<layer;i++){
            free(K_cache[head_index][i]);
            free(V_cache[head_index][i]);
        }
	}
	free(sum);
    free(max);
	free(FCm_decoder);
	free(FCm_dpu_decoder);
	free(FC_decoder);
	free(FC_dpu_decoder);
	free(finalout);
	free(finalout_dpu);
}

#endif