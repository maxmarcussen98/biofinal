#include <stdio.h>
#include <stdlib.h>

__global__ void hist_kernel(char* d1, char* d2, int* ftable, int* source1, int* source2, char* dmatch1, char* dmatch2, int sz1, int sz2, int gap, int reward) {
    int collen = sz1;
    int rowlen = sz2;
    int tablesize = max(sz1, sz2);
    //int ftable[rowlen*collen];
    //int source1[rowlen*collen];
    //int source2[rowlen*collen];
    // initialize edges of ftable
    // sz1 = num rows, sz2 = num cols
    for (int i = 0; i < sz1; i++) {
        for (int j = 0; j < sz2; j++) {
            ftable[i*sz2+j] = 0;
        }
    }
    for (int i = threadIdx.x + blockIdx.x * blockDim.x+1; i < sz1; i += blockDim.x * gridDim.x) {
	for (int j = threadIdx.y + blockIdx.y * blockDim.y+1; j < sz2; j += blockDim.y * gridDim.y) {
	    int score = 0;
	    // calculate ftable entry at this point
	    if (d1[i] == d2[j]) {
	        score = reward;
	    } else {
	        score = 0;
	    }
	    if ((gap+ftable[i*sz2+j-1] >= score+ftable[(i-1)*sz2+j-1]) 
                && (gap+ftable[i*sz2+j-1] >= gap+ftable[(i-1)*sz2+j])
		&& (gap+ftable[i*sz2+j-1] > 0)){
		    source1[i*sz2+j] = i;
		    source2[i*sz2+j] = j-1;
	    }
            else if ((score+ftable[(i-1)*sz2+j-1] >= gap+ftable[(i-1)*sz2+j]) 
		      && (score+ftable[(i-1)*sz2+j-1] > 0)) {
	        source1[i*sz2+j] = i-1;
		source2[i*sz2+j] = j-1;
		ftable[i*sz2+j] = score+ftable[(i-1)*sz2+j-1];
	    }
	    else if (gap+ftable[(i-1)*sz2+j] > 0) {
	        source1[i*sz2+j] = i-1;
		source2[i*sz2+j] = j;
		ftable[i*sz2+j] = gap+ftable[(i-1)*sz2+j];
	    }
	    else {
	        ftable[i*sz2+j] = 0;
	    }
	}	
    }
    // now go through whole ftable and find max value - 
    // this should be parallelized too
    int maxval = -1000;
    int spot1 = 0;
    int spot2 = 0;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x+1; i < sz1; i += blockDim.x * gridDim.x) {
        for (int j = threadIdx.y + blockIdx.y * blockDim.y+1; j < sz2; j += blockDim.y * gridDim.y) {
            if (ftable[i*sz2+j] > maxval) {
	        maxval = ftable[i*sz2+j];
		spot1 = i;
		spot2 = j;
	    }
	}
    }

    // now we rebuild the output sequences
    printf("%d %d\n", spot1, spot2);
}

int main(int argc, char* argv[]){
    FILE *seq1;
    FILE *seq2;
    int gap;
    int reward;
    int grid;
    int block;
    if (argc == 5) {
        seq1 = fopen(argv[1], "rb");
	seq2 = fopen(argv[2], "rb");
        gap = atoi(argv[3]);
	reward = atoi(argv[4]);
	grid = atoi(argv[5]);
	block = atoi(argv[6]);
    }
    else {
        printf("Incorrect number of arguments. Arguments should take form: sequence 1, sequence 2, gap penalty, score for a match, grid dim, block dim.\n");
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
    hist_kernel<<<grid, block>>>(d1, d2, dmatch1, dmatch2, sz1, sz2, gap, reward);
    cudaEventRecord(tock, 0);
    cudaEventSynchronize(tock);
    float time;
    cudaEventElapsedTime(&time, tick, tock);
    printf("time elapsed: %0.6f ms\n", time);
}
