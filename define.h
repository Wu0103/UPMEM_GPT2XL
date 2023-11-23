#ifndef __DEFINE_H_
 
#define __DEFINE_H_

unsigned int rep;
unsigned int i;
struct dpu_set_t dpu, dpu_set_decoder[multihead];
uint32_t nr_of_dpus;
int which_res = 0;
int layers = 48;
int layers_index = 0;
int pos;
T input_before_norm[sequence_length*d_model];
 int *source;
 dpu_arguments_t *input_args;
 long double* sum;
 long double* max;
 T* IM_decoder;
 T* CQ_decoder;
 T* CK_decoder;
 T* CV_decoder;
 T* CQK_decoder;
 T* CTV_decoder;
 T* CR_decoder;
 T* CO_decoder;
 T* COnorm_decoder;
 T* CK_dpu_decoder[multihead];
 T* CQ_dpu_decoder[multihead];
 T* CV_dpu_decoder[multihead];
 T* CQK_dpu_decoder[multihead];
 T* CR_dpu_decoder[multihead];
 T* CO_dpu_decoder[multihead];
 T* K_cache[multihead][layer];
 T* V_cache[multihead][layer];
 T* FCm_decoder;
 T* FC_decoder;
 T* layer_decoder;
 T* FCm_dpu_decoder;
 T* FC_dpu_decoder;
 T* finalout;
 T* finalout_dpu;
 T* weight_tok;
 T* weight_pos;
 T* normweight1;
 T* normweight2;
 T* normbias1;
 T* normbias2;
 T* normweightfinal;
 T* normbiasfinal;

 struct timeval decoderkernls,decoderkernle,decoderdpucpus,decoderdpucpue,decodercpudpus,decodercpudpue;

 struct timeval decodermhaacts,decodermhaacte,decodermhapros,decodermhaproe,decoderffns,decoderffne,decoderothers,decoderothere;

 double decoderkernel,decoderdpucpu,decodercpudpu;

 double decodermhaact,decodermhapro,decoderffn,decoderother;

 double sumkernel,sumdpucpu,sumcpudpu,sumother,sumattention,sumffn;
 double genkernel,gencpudpu,gendpucpu,genother,genattention,genffn;


 struct timeval sumkernels,sumkernele,sumdpucpus,sumdpucpue,sumcpudpus,sumcpudpue,summems,summeme,sumcoms,sumcome;
 struct timeval genkernels,genkernele,gendpucpus,gendpucpue,gencpudpus,gencpudpue,genmems,genmeme,gencoms,gencome;
 double sumattenkernel,sumattencpudpu,sumattendpucpu,sumattenmem,sumattencom,sumffnkernel,sumffncpudpu,sumffndpucpu,sumffnmem,sumffncom,sumfinalkernel,sumfinalcpudpu,sumfinaldpucpu,sumfinalmem,sumfinalcom;
 double genattenkernel,genattencpudpu,genattendpucpu,genattenmem,genattencom,genffnkernel,genffncpudpu,genffndpucpu,genffnmem,genffncom,genfinalkernel,genfinalcpudpu,genfinaldpucpu,genfinalmem,genfinalcom;


double
get_runtime(double start_sec, double start_us, double end_sec, double end_us){
    double duration = (end_sec * 1000 + end_us / 1000) - (start_sec * 1000 + start_us / 1000 );
    return duration;
}


void load_data(){
    const char *wq_path = "/home/kaist_icn/wuxiangyu/upload/GPT2/data/wq/";
    const char *wk_path = "/home/kaist_icn/wuxiangyu/upload/GPT2/data/wk/";
    const char *wv_path = "/home/kaist_icn/wuxiangyu/upload/GPT2/data/wv/";
    const char *wo_path = "/home/kaist_icn/wuxiangyu/upload/GPT2/data/wo/";
    const char *wfc1_path = "/home/kaist_icn/wuxiangyu/upload/GPT2/data/wfc1/";
    const char *wfc2_path = "/home/kaist_icn/wuxiangyu/upload/GPT2/data/wfc2/";
    
    int row=0,column=0;
    int each;
    char *line,*record;
	char* buffer = malloc(sizeof(char) * readersize);
    memset(buffer,0,sizeof(char) * readersize);

    FILE *file;
    T* weight_q = (T*)malloc(d_model * sizeof(T) * d_weight * multihead);
    for(int i=1;i<=layer;i++){
        column = 0;
        row = 0;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/wq%d.csv", wq_path, i);
        file = fopen(file_path, "r");
        if (file == NULL) {
            perror("无法打开文件");
            exit(EXIT_FAILURE);
        }
        fseek(file, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(char) * readersize, file))!=NULL){
            record = strtok(line, ",");
            while(record!=NULL){
                weight_q[row*d_model + column] = (atof(record));
                column++;
                record = strtok(NULL, ",");
            }
            row++;
            column = 0;
        }
        for(int head_index=0;head_index<multihead;head_index++){
            DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_q+ head_index*d_weight*d_model+each*w_dpu*d_model));
		    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WQ", (i-1)*w_dpu*d_model*sizeof(T),(sizeof(T) * w_dpu * d_model), DPU_XFER_DEFAULT));
		}
    }
    free(weight_q);
    T* weight_k = (T*)malloc(d_model * sizeof(T) * d_weight * multihead);
    for(int i=1;i<=layer;i++){
        column = 0;
        row = 0;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/wk%d.csv", wk_path, i);
        file = fopen(file_path, "r");
        if (file == NULL) {
            perror("无法打开文件");
            exit(EXIT_FAILURE);
        }
        fseek(file, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(char) * readersize, file))!=NULL){
            record = strtok(line, ",");
            while(record!=NULL){
                weight_k[row*d_model + column] = (atof(record));
                column++;
                record = strtok(NULL, ",");
            }
            row++;
            column = 0;
        }
        for(int head_index=0;head_index<multihead;head_index++){
            DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_k+head_index*d_weight*d_model+each*w_dpu*d_model));
		    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WK", (i-1)*w_dpu*d_model*sizeof(T),(sizeof(T) * w_dpu * d_model), DPU_XFER_DEFAULT));
		}
    }
    free(weight_k);
    T* weight_v = (T*)malloc(d_model * sizeof(T) * d_weight * multihead);
    for(int i=1;i<=layer;i++){
        column = 0;
        row = 0;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/wv%d.csv", wv_path, i);
        file = fopen(file_path, "r");
        if (file == NULL) {
            perror("无法打开文件");
            exit(EXIT_FAILURE);
        }
        fseek(file, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(char) * readersize, file))!=NULL){
            record = strtok(line, ",");
            while(record!=NULL){
                weight_v[row*d_model + column] = (atof(record));
                column++;
                record = strtok(NULL, ",");
            }
            row++;
            column = 0;
        }
        for(int head_index=0;head_index<multihead;head_index++){
            DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_v+head_index*d_weight*d_model+each*w_dpu*d_model));
		    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WV", (i-1)*w_dpu*d_model*sizeof(T),(sizeof(T) * w_dpu * d_model), DPU_XFER_DEFAULT));
		}
    }
    free(weight_v);
    T* weight_o = (T*)malloc(d_model * sizeof(T) * d_weight * multihead);
    for(int i=1;i<=layer;i++){
        column = 0;
        row = 0;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/wo%d.csv", wo_path, i);
        file = fopen(file_path, "r");
        if (file == NULL) {
            perror("无法打开文件");
            exit(EXIT_FAILURE);
        }
        fseek(file, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(char) * readersize, file))!=NULL){
            record = strtok(line, ",");
            while(record!=NULL){
                weight_o[row*d_model + column] = (atof(record));
                column++;
                record = strtok(NULL, ",");
            }
            row++;
            column = 0;
        }
        for(int head_index=0;head_index<multihead;head_index++){
            DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_o+head_index*d_model*m_dpu*NR_DPUS+each*m_dpu*d_model));
		    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WO", (i-1)*m_dpu*d_model*sizeof(T),(sizeof(T)*m_dpu*d_model), DPU_XFER_DEFAULT));
		}
    }
    free(weight_o);
    T* weight_fc1 = (T*)malloc(d_model * sizeof(T) * h_predefined);
    for(int i=1;i<=layer;i++){
        column = 0;
        row = 0;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/wfc1%d.csv", wfc1_path, i);
        file = fopen(file_path, "r");
        if (file == NULL) {
            perror("无法打开文件");
            exit(EXIT_FAILURE);
        }
        fseek(file, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(char) * readersize, file))!=NULL){
            record = strtok(line, ",");
            while(record!=NULL){
                weight_fc1[row*d_model + column] = (atof(record));
                column++;
                record = strtok(NULL, ",");
            }
            row++;
            column = 0;
        }
        for(int head_index=0;head_index<multihead;head_index++){
            DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_fc1+head_index*d_model*h_dpu*NR_DPUS+each*h_dpu*d_model));
		    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WO1", (i-1)*h_dpu*d_model*sizeof(T),(sizeof(T)*h_dpu*d_model), DPU_XFER_DEFAULT));
		}
        //DPU_FOREACH(dpu_set_fcdecode[0], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_fc1+each*h_dpu*d_model));
		//DPU_ASSERT(dpu_push_xfer(dpu_set_fcdecode[0], DPU_XFER_TO_DPU, "WO1", (i-1)*h_dpu*d_model*sizeof(T),(sizeof(T) * h_dpu * d_model), DPU_XFER_DEFAULT));
    }
    free(weight_fc1);
    T* weight_fc2 = (T*)malloc(d_model * sizeof(T) * h_predefined);
    for(int i=1;i<=layer;i++){
        column = 0;
        row = 0;
        char file_path[256];
        snprintf(file_path, sizeof(file_path), "%s/wfc2%d.csv", wfc2_path, i);
        file = fopen(file_path, "r");
        if (file == NULL) {
            perror("无法打开文件");
            exit(EXIT_FAILURE);
        }
        fseek(file, 0L, SEEK_SET);
        while ((line = fgets(buffer, sizeof(char) * readersize, file))!=NULL){
            record = strtok(line, ",");
            while(record!=NULL){
                weight_fc2[row*h_predefined + column] = (atof(record));
                column++;
                record = strtok(NULL, ",");
            }
            row++;
            column = 0;
        }
        for(int head_index=0;head_index<multihead;head_index++){
            DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_fc2+head_index*h_predefined*mfc_dpu*NR_DPUS+each*mfc_dpu*h_predefined));
		    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WO2", (i-1)*mfc_dpu*h_predefined*sizeof(T),(sizeof(T)*mfc_dpu*h_predefined), DPU_XFER_DEFAULT));
		}
        //DPU_FOREACH(dpu_set_fcdecode[0], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_fc2+each*mfc_dpu*h_predefined));
		//DPU_ASSERT(dpu_push_xfer(dpu_set_fcdecode[0], DPU_XFER_TO_DPU, "WO2", (i-1)*mfc_dpu*h_predefined*sizeof(T),(sizeof(T)*mfc_dpu*h_predefined), DPU_XFER_DEFAULT));
    }
    free(weight_fc2);
    fclose(file);

    FILE *fp_final = fopen("/home/kaist_icn/wuxiangyu/gpt-2-Pytorch/data/Final.csv", "r");
    T* weight_final = (T*)malloc(d_model * sizeof(T) * Round(target_predefined,NR_DPUSFC));
    column = 0;
    row = 0;
    if (fp_final == NULL) {
        perror("无法打开文件");
        exit(EXIT_FAILURE);
    }
    fseek(fp_final, 0L, SEEK_SET);
    while ((line = fgets(buffer, sizeof(char) * readersize, fp_final))!=NULL){
        record = strtok(line, ",");
        while(record!=NULL){
            weight_final[row*d_model + column] = (atof(record));
            column++;
            record = strtok(NULL, ",");
        }
        row++;
        column = 0;
    }
    for(int head_index=0;head_index<multihead;head_index++){
        DPU_FOREACH(dpu_set_decoder[head_index], dpu, each) DPU_ASSERT(dpu_prepare_xfer(dpu, weight_final+head_index*d_model*t_dpu*NR_DPUS+each*t_dpu*d_model));
	    DPU_ASSERT(dpu_push_xfer(dpu_set_decoder[head_index], DPU_XFER_TO_DPU, "WOF", 0,(sizeof(T)*t_dpu*d_model), DPU_XFER_DEFAULT));
	}
    fclose(fp_final);
    free(weight_final);
    free(buffer);
}

#endif