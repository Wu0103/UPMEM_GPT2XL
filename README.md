# Intro

UPMEM_GPT is a implementation of GPT2-XL model in UPMEM system.
GPT2-XL is a model with 1.6B parameters. More infomation can be found in the offical github page here: https://github.com/openai/gpt-2

# Pre-requirment

**You need to change the file path in **"/UPMEM_GPT2/define.h"**" into your specific settings---please refer to the figure below.**(Default path is "/home/kaist_icn/wuxiangyu/upload/GPT2/data" )

![image](https://github.com/Wu0103/UPMEM_GPT2XL/assets/94586355/7a026ffa-e265-48bb-967a-9e0d0a25da23)


# Usage

To use it, please config parameters in file **"/UPMEM_GPT2/common.h"**.**"common.h"** is a file that contains the model information and also your input & output sequence length.

User can freely configure the input tokens and also the number of output tokens they want to generate, by changing the number of **sequence_length** and **generation_length** in the file.

![image](https://github.com/Wu0103/UPMEM_GPT2XL/assets/94586355/da30d16a-dc1d-4bd3-9302-65379741182a)


Also, user can decide how many DPUs used to process each head in the multihead attention block by configuring the number of **NR_DPUS**.

![image](https://github.com/Wu0103/UPMEM_GPT2XL/assets/94586355/9ec623c5-af97-4db7-8dec-c93ad9d76292)


After configing all the parameters,use **./result.sh** in the terminal to try with synthetic input;

For now, even through the implementation are provided with trained model weights, the input tokens are generated randomly. The part that mapping words into numerical token is not provided yet.

# Results

GPT2 latency will be automatically print to a file **result**, just like the figure below shows: CPU-DPU indicates the latency of transfer input data from CPU to DPUs. kernel indicates DPU program running latency and DPU-CPU indicates the latency of get results back to CPU. Detailed latency breakdown will also being in the file, including the time for summarization stage and generation, as well as the time for attention block and feed forward block.

![image](https://github.com/Wu0103/UPMEM_GPT2XL/assets/94586355/6985ef58-2559-4d5b-b524-a9ccd3d4e8d2)


# For your customization

If you want to try with another model, you should change the model configuration in **common.h** file accordingly. 

# Contact

Please email me at wuxiangyu@kaist.ac.kr if you have any problem using it.
