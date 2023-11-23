#include <stdint.h>
#include <stdio.h>
//#include <mram_unaligned.h>
#include <defs.h>
#include "mutex.h"
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>
#include <string.h>
#include <perfcounter.h>
#include "common.h"

#define debug false
MUTEX_INIT(my_mutex);
bool pass;
T recoder[3][NR_TASKLETS];
T buffer[3][2];
__mram_noinit T mramI[sequence_length][d_model];
__mram_noinit T WQ[layer][w_dpu][d_model];
__mram_noinit T WK[layer][w_dpu][d_model];
__mram_noinit T WV[layer][w_dpu][d_model];
__mram_noinit T WO[layer][m_dpu][d_model];
__mram_noinit T mramQ[sequence_length*w_dpu];
__mram_noinit T mramK[sequence_length*w_dpu];
__mram_noinit T mramV[sequence_length*w_dpu];
__mram_noinit T FullQ[total_length*d_model];
__mram_noinit T SaveK[((Round((total_length),NR_DPUS))/NR_DPUS)*d_weight];
__mram_noinit T mramQK[sequence_length*total_length];
__mram_noinit T SaveV[((Round((total_length),NR_DPUS))/NR_DPUS)*d_weight];
__mram_noinit T mramO[total_length*m_dpu];

__mram_noinit T WO1[layer][h_dpu][d_model];
__mram_noinit T WO2[layer][mfc_dpu][h_predefined];
__mram_noinit T WOF[t_dpu][d_model];
__mram_noinit T mramO1[sequence_length*h_predefined];
__mram_noinit T mramO2[sequence_length*d_model];
__mram_noinit T mramO3[sequence_length*t_dpu];

BARRIER_INIT(my_barrier, NR_TASKLETS);
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host uint32_t count;

static void gemv3(int id,T *bufferI, T *buffer1, T *buffer2, T *buffer3,int n,int n_size) {
	for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {
		if(i+n>=n_size) break;
		recoder[0][id] += bufferI[i] * buffer1[i];
		recoder[1][id] += bufferI[i] * buffer2[i];
		recoder[2][id] += bufferI[i] * buffer3[i];
	}
	return;
}

static void gemv1(int id,T *buffer1, T *buffer2,int n,int n_size,int size,bool odd1,bool odd2) {
	for (unsigned int i = 0; i < size / sizeof(T); i++) {
		if(i+n>=n_size) break;
		//if(odd)recoder[0][id] += buffer1[i+1] * buffer2[i+1];
		//else recoder[0][id] += buffer1[i] * buffer2[i];
		if(odd1&&odd2) recoder[0][id] += buffer1[i+1] * buffer2[i+1];
		else if(odd1) recoder[0][id] += buffer1[i+1] * buffer2[i];
		else if(odd2) recoder[0][id] += buffer1[i] * buffer2[i+1];
		else recoder[0][id] += buffer1[i] * buffer2[i];
	}
	return;
}

int main() {
#if inst_cycle
perfcounter_config(COUNT_INSTRUCTIONS, true);
#else
perfcounter_config(COUNT_CYCLES, true);
#endif
	unsigned int tasklet_id = me();
	if (tasklet_id == 0){
		mem_reset();
	}
	barrier_wait(&my_barrier);
	uint32_t layerindex = DPU_INPUT_ARGUMENTS.layerindex;
    uint32_t which = DPU_INPUT_ARGUMENTS.which;
	uint32_t head = DPU_INPUT_ARGUMENTS.head;
	unsigned int rows_per_tasklet; 
	unsigned int start_row;
	if(DPU_INPUT_ARGUMENTS.kernel==0){
		unsigned int nrows = d_model;
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_I = (T *) mem_alloc(BLOCK_SIZE);
		T *cache_Q = (T *) mem_alloc(BLOCK_SIZE);
		T *cache_K = (T *) mem_alloc(BLOCK_SIZE);
		T *cache_V = (T *) mem_alloc(BLOCK_SIZE);
		int upper;
		if(d_model>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = d_model;
		int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<length;s_index++){
			if(w_dpu==1 && s_index%2==0) memset(buffer,0,sizeof(T)*3*2);
			else if(w_dpu!=1)	memset(buffer,0,sizeof(T)*3*2);
			for(int w_index=0;w_index<w_dpu;w_index++){
				memset(recoder,0,sizeof(T)*3*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int h_index=start_row;h_index<upper;h_index+=(BLOCK_SIZE/sizeof(T))){
					mram_read(&mramI[s_index][h_index],  cache_I, BLOCK_SIZE);
                    mram_read(&WQ[layerindex][w_index][h_index], cache_Q, BLOCK_SIZE);
                    mram_read(&WK[layerindex][w_index][h_index], cache_K, BLOCK_SIZE);
                    mram_read(&WV[layerindex][w_index][h_index], cache_V, BLOCK_SIZE);
					gemv3(tasklet_id,cache_I, cache_Q, cache_K, cache_V, h_index,upper);
				}
				barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(w_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++){
							buffer[0][w_index%2] += recoder[0][i];
							buffer[1][w_index%2] += recoder[1][i];
							buffer[2][w_index%2] += recoder[2][i];
						}
					//	mram_write(buffer[0],&mramQ[s_index*w_dpu+(w_index)], 8);
	                //  mram_write(buffer[1],&mramK[s_index*w_dpu+(w_index)], 8); 
    	            //	mram_write(buffer[2],&mramV[s_index*w_dpu+(w_index)], 8);
						mramQ[s_index*w_dpu+(w_index)] = buffer[0][w_index%2];
						mramK[s_index*w_dpu+(w_index)] = buffer[1][w_index%2];
						mramV[s_index*w_dpu+(w_index)] = buffer[2][w_index%2];
					}else{
						for(int i=0;i<NR_TASKLETS;i++){
							buffer[0][s_index%2] += recoder[0][i];
							buffer[1][s_index%2] += recoder[1][i];
							buffer[2][s_index%2] += recoder[2][i];
						}
						mramQ[s_index] = buffer[0][s_index%2];
						mramK[s_index] = buffer[1][s_index%2];
						mramV[s_index] = buffer[2][s_index%2];
						//mram_write(buffer[0],&mramQ[s_index], 8);
                    	//mram_write(buffer[1],&mramK[s_index], 8); 
                    	//mram_write(buffer[2],&mramV[s_index], 8);
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(w_index%2==1) memset(buffer,0,sizeof(T)*3*2);
				barrier_wait(&my_barrier);
			}
		}
	}else if(DPU_INPUT_ARGUMENTS.kernel==1){
        unsigned int nrows = d_weight;
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_Q = (T *) mem_alloc(BLOCK_SIZE2);
		T *cache_K = (T *) mem_alloc(BLOCK_SIZE2);
		int s_dpu = Round((sequence_length+which),NR_DPUS)/NR_DPUS;
		int upper;
		if(d_weight>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = d_weight;
		int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<length;s_index++){
			memset(buffer,0,sizeof(T)*2);
			for(int h_index=0;h_index<(s_dpu);h_index++){
				memset(recoder,0,sizeof(T)*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int w_index=start_row;w_index<upper;w_index+=(BLOCK_SIZE2/sizeof(T))){
					mram_read(&FullQ[s_index*d_weight+w_index],  cache_Q, BLOCK_SIZE2);
                    mram_read(&SaveK[h_index*d_weight+w_index],  cache_K, BLOCK_SIZE2);
					bool odd1 = ((s_index*d_weight+w_index)%2!=0);
					bool odd2 = ((h_index*d_weight+w_index)%2!=0);
					gemv1(tasklet_id,cache_Q, cache_K, w_index,upper,BLOCK_SIZE2,odd1,odd2);
				}
				barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(s_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][h_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramQK[s_index*s_dpu+(h_index)], 8);
						mramQK[s_index*s_dpu+(h_index)] = buffer[0][h_index%2];
					}else{
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][s_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramQK[s_index], 8);
						mramQK[s_index] = buffer[0][s_index%2];
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(h_index%2==1) memset(buffer,0,sizeof(T)*2);
				barrier_wait(&my_barrier);
			}
		}
    }
	else if(DPU_INPUT_ARGUMENTS.kernel==2){
		unsigned int nrows = (sequence_length+which);
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_QK = (T *) mem_alloc(BLOCK_SIZE3);
		T *cache_V = (T *) mem_alloc(BLOCK_SIZE3);
		int upper;
		if(sequence_length+which>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = sequence_length+which;
		int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<length;s_index++){
			memset(buffer,0,sizeof(T)*2);
			for(int w_index=0;w_index<w_dpu;w_index++){
				memset(recoder,0,sizeof(T)*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int h_index=start_row;h_index<upper;h_index+=BLOCK_SIZE3/sizeof(T)){
					mram_read(&mramQK[s_index*(sequence_length+which)+h_index],  cache_QK, BLOCK_SIZE3);
                    mram_read(&SaveV[w_index*(sequence_length+which)+h_index],  cache_V, BLOCK_SIZE3);
					bool odd1 = ((s_index*(sequence_length+which)+h_index)%2!=0);
					bool odd2 = ((w_index*(sequence_length+which)+h_index)%2!=0);
					gemv1(tasklet_id,cache_QK,cache_V,h_index,upper,BLOCK_SIZE3,odd1,odd2);
				}
				barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(w_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][w_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramV[s_index*w_dpu+(w_index)], 8);
						mramV[s_index*w_dpu+(w_index)] = buffer[0][w_index%2];
					}else{
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][s_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramV[s_index], 8);
						mramV[s_index] = buffer[0][s_index%2];
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(w_index%2==1) memset(buffer,0,sizeof(T)*2);
				barrier_wait(&my_barrier);
			}
		}
	}else if(DPU_INPUT_ARGUMENTS.kernel==3){
		unsigned int nrows = d_model;
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_V = (T *) mem_alloc(BLOCK_SIZE);
		T *cache_O = (T *) mem_alloc(BLOCK_SIZE);
		int upper;
		if(d_model>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = d_model;
		int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<length;s_index++){
			memset(buffer,0,sizeof(T)*2);
			for(int w_index=0;w_index<m_dpu;w_index++){
				memset(recoder,0,sizeof(T)*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int h_index=start_row;h_index<upper;h_index+=BLOCK_SIZE/sizeof(T)){
					mram_read(&FullQ[s_index*(d_model)+h_index],  cache_V, BLOCK_SIZE);
                    mram_read(&WO[layerindex][w_index][h_index],  cache_O, BLOCK_SIZE);
					bool odd1 = ((s_index*(d_model)+h_index)%2!=0);
					bool odd2 = ((h_index)%2!=0);
					gemv1(tasklet_id,cache_V, cache_O, h_index,upper,BLOCK_SIZE,odd1,odd2);
				}
				barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(m_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][w_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO[s_index*m_dpu+(w_index)], 8);
						mramO[s_index*m_dpu+(w_index)] = buffer[0][w_index%2];
					}else{
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][s_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO[s_index], 8);
						mramO[s_index] = buffer[0][s_index%2];
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(w_index%2==1) memset(buffer,0,sizeof(T)*2);
				barrier_wait(&my_barrier);
			}
		}
	}else if(DPU_INPUT_ARGUMENTS.kernel==4){
        unsigned int nrows = d_model;
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_I = (T *) mem_alloc(BLOCK_SIZE);
		T *cache_W = (T *) mem_alloc(BLOCK_SIZE);
        int upper;
		if(d_model>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = d_model;
        int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<(length);s_index++){
			memset(buffer,0,sizeof(T)*2);
			for(int h_index=0;h_index<(h_dpu);h_index++){
				memset(recoder,0,sizeof(T)*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int w_index=start_row;w_index<upper;w_index+=(BLOCK_SIZE/sizeof(T))){
					mram_read(&mramI[s_index][w_index],  cache_I, BLOCK_SIZE);
                    mram_read(&WO1[layerindex][h_index][w_index],  cache_W, BLOCK_SIZE);
					bool odd1 = ((w_index%2)!=0);
					gemv1(tasklet_id,cache_I, cache_W, w_index,upper,BLOCK_SIZE,odd1,odd1);
				}
                barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(h_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][h_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO1[s_index*h_dpu+(h_index)], 8);
						mramO1[s_index*h_dpu+(h_index)] = buffer[0][h_index%2];
						//printf("%f\n",mramO1[s_index*h_dpu+(h_index)]);
					}else{
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][s_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO1[s_index], 8);
						mramO1[s_index] = buffer[0][s_index%2];
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(h_index%2==1) memset(buffer,0,sizeof(T)*2);
				barrier_wait(&my_barrier);
			}
		}
    }else if(DPU_INPUT_ARGUMENTS.kernel==5){
		unsigned int nrows = (h_predefined);
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_I = (T *) mem_alloc(BLOCK_SIZE);
		T *cache_W = (T *) mem_alloc(BLOCK_SIZE);
		int upper;
		if(h_predefined>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = h_predefined;
        int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<(length);s_index++){
			memset(buffer,0,sizeof(T)*2);
			for(int w_index=0;w_index<mfc_dpu;w_index++){
				memset(recoder,0,sizeof(T)*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int h_index=start_row;h_index<upper;h_index+=BLOCK_SIZE/sizeof(T)){
					mram_read(&mramO1[s_index*h_predefined+h_index],  cache_I, BLOCK_SIZE);
                    mram_read(&WO2[layerindex][w_index][h_index],  cache_W, BLOCK_SIZE);
					bool odd1 = (((s_index*h_predefined+h_index)%2)!=0);
					bool odd2 = ((h_index%2)!=0);
					gemv1(tasklet_id,cache_I, cache_W, h_index,upper,BLOCK_SIZE,odd1,odd2);
				}
                barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(mfc_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][w_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO2[s_index*mfc_dpu+(w_index)], 8);
						mramO2[s_index*mfc_dpu+(w_index)] = buffer[0][w_index%2];
					}else{
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][s_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO2[s_index], 8);
						mramO2[s_index] = buffer[0][s_index%2];
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(w_index%2==1) memset(buffer,0,sizeof(T)*2);
				barrier_wait(&my_barrier);
			}
		}
	}else if(DPU_INPUT_ARGUMENTS.kernel==6){
		unsigned int nrows = (d_model);
		unsigned int chunks = nrows / (NR_TASKLETS);
		unsigned int rest_rows = nrows % (NR_TASKLETS);
        rows_per_tasklet = chunks; 
		if (tasklet_id < rest_rows)   rows_per_tasklet += 1;
		if (rest_rows > 0) {
			if (tasklet_id >= rest_rows) 	start_row = rest_rows + tasklet_id * chunks; 
			else    start_row = tasklet_id * rows_per_tasklet;
		} else  start_row = tasklet_id * chunks;
		T *cache_I = (T *) mem_alloc(BLOCK_SIZE2);
		T *cache_W = (T *) mem_alloc(BLOCK_SIZE2);
		int upper;
		if(d_model>start_row+rows_per_tasklet) upper = start_row+rows_per_tasklet;
		else upper = d_model;
        int length;
		if(which==0) length = sequence_length;
		else length = 1;
		for(int s_index=0;s_index<(length);s_index++){
			memset(buffer,0,sizeof(T)*2);
			for(int w_index=0;w_index<t_dpu;w_index++){
				memset(recoder,0,sizeof(T)*NR_TASKLETS);
				barrier_wait(&my_barrier);
				pass = false;
				for(int h_index=start_row;h_index<upper;h_index+=BLOCK_SIZE2/sizeof(T)){
					mram_read(&mramO2[s_index*d_model+h_index],  cache_I, BLOCK_SIZE2);
                    mram_read(&WOF[w_index][h_index],  cache_W, BLOCK_SIZE2);
					bool odd1 = (((s_index*d_model+h_index)%2)!=0);
					bool odd2 = ((h_index%2)!=0);
					gemv1(tasklet_id,cache_I, cache_W, h_index,upper,BLOCK_SIZE2,odd1,odd2);
				}
                barrier_wait(&my_barrier);
				mutex_lock(my_mutex);
				if(pass==false){
					if(t_dpu!=1){
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][w_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO3[s_index*t_dpu+(w_index)], 8);
						mramO3[s_index*t_dpu+(w_index)] = buffer[0][w_index%2];
					}else{
						for(int i=0;i<NR_TASKLETS;i++)	buffer[0][s_index%2] += recoder[0][i];
						//mram_write(buffer[0],&mramO3[s_index], 8);
						mramO3[s_index] = buffer[0][s_index%2];
					}
					pass = true;
				}
				mutex_unlock(my_mutex);
				if(w_index%2==1) memset(buffer,0,sizeof(T)*2);
				barrier_wait(&my_barrier);
			}
		}
	}
    
	count = perfcounter_get();
	return 0;
}
