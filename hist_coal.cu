#include <stdio.h>
#include <stdlib.h>

__global__ void hist_kernel(char* d1, char* d2, char* dmatch1, char* dmatch2, int sz1, int sz2) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < sz1; i += blockDim.x * gridDim.x) {
	for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < sz2; j += blockDim.y * gridDim.y) {
	    if (d1[i] == d2[j]) {
	        printf("Match found\n");
		printf("Indices %d %d \n", i, j);
	    }
	}	
    }
}

int main(int argc, char* argv[]){
    FILE *seq1;
    FILE *seq2;
    int grid;
    int block;
    if (argc == 5) {
        seq1 = fopen(argv[1], "rb");
	seq2 = fopen(argv[2], "rb");
        grid = atoi(argv[3]);
	block = atoi(argv[4]);
    }
    else {
        printf("Incorrect number of arguments. Arguments should take form: sequence 1, sequence 2, grid dim, block dim.\n");
	exit(0);
    }
    // First what we need to do is get the size of the input.
    char name1[128];
    char name2[128];
    fgets(name1, 128, seq1);
    fgets(name2, 128, seq2);
    fseek(seq1, 0L, SEEK_END);
    int sz1 = ftell(seq1)-strlen(name1)-1;
    fseek(seq2, 0L, SEEK_END);
    int sz2 = ftell(seq2)-strlen(name2)-1;
    printf("%d %d\n", sz1, sz2);
    // sz1 and sz2 are the sizes of the sequences after the first line

    // get back to start of sequence
    fseek(seq1, 0L, SEEK_SET);
    fseek(seq2, 0L, SEEK_SET);
    fgets(name1, 128, seq1);
    fgets(name2, 128, seq2);
 
    char* text1 = (char*)malloc(sz1*sizeof(char));
    char* text2 = (char*)malloc(sz1*sizeof(char));
    fread(text1, sizeof(char), sz1, seq1);
    fread(text2, sizeof(char), sz2, seq2);
    
    // printf("%c", text1[6]);
    
    // data is all read in!
    // now time to get cuda working
    // create data in CUDA for both texts and for best matchstrs   
    char* d1;
    cudaMalloc((void**)&d1, sz1*sizeof(char));
    cudaMemcpy(d1, text1, sz1, cudaMemcpyHostToDevice);
    char* d2;
    cudaMalloc((void**)&d2, sz2*sizeof(char));
    cudaMemcpy(d2, text2, sz2, cudaMemcpyHostToDevice);
 
    char* match1 = (char*) malloc(sz1*sizeof(char));
    char* match2 = (char*) malloc(sz2*sizeof(char));
    char* dmatch1;
    char* dmatch2;
    cudaMalloc((void**)&dmatch1, sz1*sizeof(char));
    cudaMalloc((void**)&dmatch2, sz2*sizeof(char));

    // create start and end events
    cudaEvent_t tick, tock;
    cudaEventCreate(&tick);
    cudaEventCreate(&tock);
    // call funciton and report time
    cudaEventRecord(tick, 0);
    hist_kernel<<<grid, block>>>(d1, d2, dmatch1, dmatch2, sz1, sz2);
    cudaEventRecord(tock, 0);
    cudaEventSynchronize(tock);
    float time;
    cudaEventElapsedTime(&time, tick, tock);
    printf("time elapsed: %0.6f ms\n", time);
}
