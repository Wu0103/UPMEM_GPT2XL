//把encoder和decoder结合，并且读取使用真实的权重数据
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
#include "host.h"
#ifndef DECODER
#define DECODER "decoder"
#endif

void create_emb(){
    int column=0,row=0;
    char *line,*record;
	char* buffer = malloc(sizeof(char) * readersize);
    memset(buffer,0,sizeof(char) * readersize);
    FILE *fp_tok = fopen("/home/kaist_icn/wuxiangyu/gpt-2-Pytorch/data/token_embedding.csv", "r");
    FILE *fp_pos = fopen("/home/kaist_icn/wuxiangyu/gpt-2-Pytorch/data/positions_embedding.csv", "r");
    if (fp_tok == NULL || fp_pos == NULL) {
        perror("无法打开emb文件");
        exit(EXIT_FAILURE);
    }
    fseek(fp_tok, 0L, SEEK_SET);
    while ((line = fgets(buffer, sizeof(char) * readersize, fp_tok))!=NULL){
        record = strtok(line, ",");
        while(record!=NULL){
            weight_tok[row*d_model + column] = (atof(record));
            column++;
            record = strtok(NULL, ",");
        }
        row++;
        column = 0;
    }
    column=0;
    row=0;
    fseek(fp_pos, 0L, SEEK_SET);
    while ((line = fgets(buffer, sizeof(char) * readersize, fp_pos))!=NULL){
        record = strtok(line, ",");
        while(record!=NULL){
            weight_pos[row*d_model + column] = (atof(record));
            column++;
            record = strtok(NULL, ",");
        }
        row++;
        column = 0;
    }
    fclose(fp_tok);
    fclose(fp_pos);
    free(buffer);
    return;
}

int main(int argc, char **argv) {
    
    bool debug = false;

    weight_tok = (T*)malloc(d_model * sizeof(T) * target_predefined);
    weight_pos = (T*)malloc(d_model * sizeof(T) * MAX_LENGTH);

    for(int head_index=0;head_index<multihead;head_index++){
    	DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set_decoder[head_index]));
	    DPU_ASSERT(dpu_load(dpu_set_decoder[head_index], DECODER, NULL));
    }

	alloc_decoder();


    printf("[" ANSI_COLOR_GREEN "start load weight" ANSI_COLOR_RESET "] \n");

    create_emb();
    //load_data();

    printf("[" ANSI_COLOR_GREEN "finish load weight" ANSI_COLOR_RESET "] \n");

    source = (int*)malloc(sizeof(int) * sequence_length);
    srand(time(NULL));

    for (i = 0; i < sequence_length; i++) {
       source[i] = rand() % 50000;
    }

    T position[d_model];
    for(int i=0;i<sequence_length;i++){
        int index = source[i];
        memcpy(&input_before_norm[d_model*i],&weight_tok[index*d_model],sizeof(T)*d_model);
        memcpy(position,&weight_pos[i*d_model],sizeof(T)*d_model);
        for(int j=0;j<d_model;j++) input_before_norm[j+i*d_model] += position[j];
    }
//////////////////////////////////input weight checked///////////////////////////////////////
    which_res = 0;
    decoderkernel = 0.0;
    decoderdpucpu = 0.0;
    decodercpudpu = 0.0;
    decodermhaact = 0.0;
    decodermhapro = 0.0;
    decoderffn = 0.0;
    decoderother = 0.0;

    sumkernel = 0.0;
    sumdpucpu = 0.0;
    sumcpudpu = 0.0;
    sumattention = 0.0;
    sumffn = 0.0;
    sumother = 0.0;
    genkernel = 0.0;
    gendpucpu = 0.0;
    gencpudpu = 0.0;
    genattention = 0.0;
    genffn = 0.0;
    genother = 0.0;

    printf("[" ANSI_COLOR_GREEN "start compute" ANSI_COLOR_RESET "] \n");
	

        which_res = 0;
        
		layers = layer;

        pos = 0;

        while(which_res<generation_length){
            if(debug)   printf("token %d start\n",which_res);
            printf("[" ANSI_COLOR_GREEN "start compute res: %d"ANSI_COLOR_RESET" ] \n",which_res);
            for(layers_index=0;layers_index<layers;layers_index++){
                stage1_decoder();
                if(debug)   printf("stage 1 end\n");
                stage2_decoder();
                if(debug)   printf("stage 2 end\n");
                stage3_decoder();
                if(debug)   printf("stage 3 end\n");
                stage4_decoder();
                if(debug)   printf("stage 4 end\n");
                stage5_decoder();
                if(debug)   printf("stage 5 end\n");
                stage6_decoder();
                if(debug)   printf("stage 6 end\n");
                
                if(layers_index==layers-1) {
                    stagefinal_decoder();
                    which_res++; 
                    memcpy(input_before_norm,&weight_tok[(pos)*d_model],sizeof(T)*d_model);
                    memcpy(position,&weight_pos[(sequence_length+which_res-1)*d_model],sizeof(T)*d_model);
                    for(int j=0;j<d_model;j++) input_before_norm[j] += position[j];
                }

                
            }
        }
        printf("[" ANSI_COLOR_GREEN "finish compute res: %d "ANSI_COLOR_RESET"] \n",which_res-1);

    
    printf("[" ANSI_COLOR_GREEN "finish compute" ANSI_COLOR_RESET "] \n");


    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("sum attention CPU-DPU is (ms): %.3f\n",((sumattencpudpu) ));
    printf("sum attention kernel is (ms): %.3f\n",((sumattenkernel) ));
    printf("sum attention DPU-CPU is (ms): %.3f\n",((sumattendpucpu) ));
    printf("sum attention data reorganization is (ms): %.3f\n",((sumattenmem) ));
    printf("sum attention host computation is (ms): %.3f\n",((sumattencom) ));
    printf("sum attention host is (ms): %.3f\n",((sumattencom + sumattenmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("sum ffn CPU-DPU is (ms): %.3f\n",((sumffncpudpu) ));
    printf("sum ffn kernel is (ms): %.3f\n",((sumffnkernel) ));
    printf("sum ffn DPU-CPU is (ms): %.3f\n",((sumffndpucpu) ));
    printf("sum ffn data reorganization is (ms): %.3f\n",((sumffnmem) ));
    printf("sum ffn host computation is (ms): %.3f\n",((sumffncom) ));
    printf("sum ffn host is (ms): %.3f\n",((sumffncom + sumffnmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("sum final CPU-DPU is (ms): %.3f\n",((sumfinalcpudpu) ));
    printf("sum final kernel is (ms): %.3f\n",((sumfinalkernel) ));
    printf("sum final DPU-CPU is (ms): %.3f\n",((sumfinaldpucpu) ));
    printf("sum final data reorganization is (ms): %.3f\n",((sumfinalmem) ));
    printf("sum final host computation is (ms): %.3f\n",((sumfinalcom) ));
    printf("sum final host is (ms): %.3f\n",((sumfinalcom + sumfinalmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("sum total CPU-DPU is (ms): %.3f\n",((sumattencpudpu+sumffncpudpu+sumfinalcpudpu) ));
    printf("sum total kernel is (ms): %.3f\n",((sumattenkernel+sumffnkernel+sumfinalkernel) ));
    printf("sum total DPU-CPU is (ms): %.3f\n",((sumattendpucpu+sumffndpucpu+sumfinaldpucpu) ));
    printf("sum total data reorganization is (ms): %.3f\n",((sumattenmem+sumffnmem+sumfinalmem) ));
    printf("sum total host computation is (ms): %.3f\n",((sumattencom+sumffncom+sumfinalcom) ));
    printf("sum total host is (ms): %.3f\n",((sumattenmem+sumffnmem+sumfinalmem+sumattencom+sumffncom+sumfinalcom) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");

    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("gen attention CPU-DPU is (ms): %.3f\n",((genattencpudpu) ));
    printf("gen attention kernel is (ms): %.3f\n",((genattenkernel) ));
    printf("gen attention DPU-CPU is (ms): %.3f\n",((genattendpucpu) ));
    printf("gen attention data reorganization is (ms): %.3f\n",((genattenmem) ));
    printf("gen attention host computation is (ms): %.3f\n",((genattencom) ));
    printf("gen attention host is (ms): %.3f\n",((genattencom + genattenmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("gen ffn CPU-DPU is (ms): %.3f\n",((genffncpudpu) ));
    printf("gen ffn kernel is (ms): %.3f\n",((genffnkernel) ));
    printf("gen ffn DPU-CPU is (ms): %.3f\n",((genffndpucpu) ));
    printf("gen ffn data reorganization is (ms): %.3f\n",((genffnmem) ));
    printf("gen ffn host computation is (ms): %.3f\n",((genffncom) ));
    printf("gen ffn host is (ms): %.3f\n",((genffncom + genffnmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("gen final CPU-DPU is (ms): %.3f\n",((genfinalcpudpu) ));
    printf("gen final kernel is (ms): %.3f\n",((genfinalkernel) ));
    printf("gen final DPU-CPU is (ms): %.3f\n",((genfinaldpucpu) ));
    printf("gen final data reorganization is (ms): %.3f\n",((genfinalmem) ));
    printf("gen final host computation is (ms): %.3f\n",((genfinalcom) ));
    printf("gen final host is (ms): %.3f\n",((genfinalcom + genfinalmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("gen total CPU-DPU is (ms): %.3f\n",((genattencpudpu+genffncpudpu+genfinalcpudpu) ));
    printf("gen total kernel is (ms): %.3f\n",((genattenkernel+genffnkernel+genfinalkernel) ));
    printf("gen total DPU-CPU is (ms): %.3f\n",((genattendpucpu+genffndpucpu+genfinaldpucpu) ));
    printf("gen total data reorganization is (ms): %.3f\n",((genattenmem+genffnmem+genfinalmem) ));
    printf("gen total host computation is (ms): %.3f\n",((genattencom+genffncom+genfinalcom) ));
    printf("gen total host is (ms): %.3f\n",((genattenmem+genffnmem+genfinalmem+genattencom+genffncom+genfinalcom) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");


    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("Total attention CPU-DPU is (ms): %.3f\n",((sumattencpudpu+genattencpudpu) ));
    printf("Total attention kernel is (ms): %.3f\n",((sumattenkernel+genattenkernel) ));
    printf("Total attention DPU-CPU is (ms): %.3f\n",((sumattendpucpu+genattendpucpu) ));
    printf("Total attention data reorganization is (ms): %.3f\n",((sumattenmem+genattenmem) ));
    printf("Total attention host computation is (ms): %.3f\n",((sumattencom+genattencom) ));
    printf("Total attention host is (ms): %.3f\n",((sumattencom + sumattenmem+genattencom + genattenmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("Total ffn CPU-DPU is (ms): %.3f\n",((sumffncpudpu+genffncpudpu) ));
    printf("Total ffn kernel is (ms): %.3f\n",((sumffnkernel+genffnkernel) ));
    printf("Total ffn DPU-CPU is (ms): %.3f\n",((sumffndpucpu+genffndpucpu) ));
    printf("Total ffn data reorganization is (ms): %.3f\n",((sumffnmem+genffnmem) ));
    printf("Total ffn host computation is (ms): %.3f\n",((sumffncom+genffncom) ));
    printf("Total ffn host is (ms): %.3f\n",((sumffncom + sumffnmem+genffncom + genffnmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("Total final CPU-DPU is (ms): %.3f\n",((sumfinalcpudpu+genfinalcpudpu) ));
    printf("Total final kernel is (ms): %.3f\n",((sumfinalkernel+genfinalkernel) ));
    printf("Total final DPU-CPU is (ms): %.3f\n",((sumfinaldpucpu+genfinaldpucpu) ));
    printf("Total final data reorganization is (ms): %.3f\n",((sumfinalmem+genfinalmem) ));
    printf("Total final host computation is (ms): %.3f\n",((sumfinalcom+genfinalcom) ));
    printf("Total final host is (ms): %.3f\n",((sumfinalcom + sumfinalmem+genfinalcom + genfinalmem) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");
    printf("Total CPU-DPU is (ms): %.3f\n",((sumattencpudpu+sumffncpudpu+sumfinalcpudpu+genattencpudpu+genffncpudpu+genfinalcpudpu) ));
    printf("Total kernel is (ms): %.3f\n",((sumattenkernel+sumffnkernel+sumfinalkernel+genattenkernel+genffnkernel+genfinalkernel) ));
    printf("Total DPU-CPU is (ms): %.3f\n",((sumattendpucpu+sumffndpucpu+sumfinaldpucpu+genattendpucpu+genffndpucpu+genfinaldpucpu) ));
    printf("Total data reorganization is (ms): %.3f\n",((sumattenmem+sumffnmem+sumfinalmem+genattenmem+genffnmem+genfinalmem) ));
    printf("Total host computation is (ms): %.3f\n",((sumattencom+sumffncom+sumfinalcom+genattencom+genffncom+genfinalcom) ));
    printf("Total host is (ms): %.3f\n",((sumattenmem+sumffnmem+sumfinalmem+sumattencom+sumffncom+sumfinalcom+genattenmem+genffnmem+genfinalmem+genattencom+genffncom+genfinalcom) ));
    printf("/////////////////////////////////////////////////////////////////////////////\n");


    //free(source);
	//dealloc_decoder();
    for(int head_index=0;head_index<multihead;head_index++){
    //	DPU_ASSERT(dpu_free(dpu_set_decoder[head_index]));
    }
    free(weight_tok);
    free(weight_pos);
	return 0;
}
