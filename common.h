#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct Params {
    unsigned int  n_warmup;
    unsigned int  n_reps;
}Params;

// Structures used by both the host and the dpu to communicate information 
typedef struct {
    uint32_t which;
    uint32_t kernel;
    uint32_t layerindex;
    uint32_t head;
    uint32_t rows;
    uint32_t pad;
} dpu_arguments_t;

#define Round(n, m) ((n / m) * m + m * (n % m != 0))
#define info false
#define readersize 16777216 
#define T float
#define layer 48
#define PRINT 0
#define multihead 25
#define NR_DPUS 64
#define NR_DPUSFC (NR_DPUS*multihead)
#define d_weight 64
#define w_dpu ((d_weight+NR_DPUS-1)/NR_DPUS)
#define sequence_length 8
#define generation_length 4
#define total_length (sequence_length+generation_length)
#define d_model 1600
#define m_dpu ((d_model+(NR_DPUS*multihead)-1)/(NR_DPUS*multihead))
#define h_predefined (d_model*4) 
#define h_dpu ((h_predefined+(NR_DPUSFC)-1)/(NR_DPUSFC))
#define mfc_dpu ((d_model+(NR_DPUSFC)-1)/(NR_DPUSFC))
#define target_predefined 50257
#define t_dpu ((target_predefined+(NR_DPUSFC)-1)/(NR_DPUSFC))
#define MAX_LENGTH 1024

#define BLOCK_SIZE 256
#define BLOCK_SIZE2 16
#define BLOCK_SIZE3 64
#define inst_cycle 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#endif
