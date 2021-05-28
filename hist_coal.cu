#include <stdio.h>
#include <stdlib.h>

__global__ void find_kernel(int* ftablefind, int* d_spots, int sz1, int sz2){
    int maxval = -10000;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x+1; i < sz1; i += blockDim.x * gridDim.x) {
        for (int j = threadIdx.y + blockIdx.y * blockDim.y+1; j < sz2; j += blockDim.y * gridDim.y) {
	    if (ftablefind[i*sz2+j] > maxval) {
                maxval = ftablefind[i*sz2+j];
		d_spots[0] = i;
		d_spots[1] = j;
		d_spots[2] = maxval;
	    }
 	}
    }	
}


__global__ void align_kernel(char* d1, char* d2, int* ftable, int* source1, int* source2, int sz1, int sz2, int gap, int reward, int penalty) {
    //int collen = sz1;
    //int rowlen = sz2;
    //int tablesize = max(sz1, sz2);
    //int ftable[rowlen*collen];
    //int source1[rowlen*collen];
    //int source2[rowlen*collen];
    // initialize edges of ftable
    // sz1 = num rows, sz2 = num cols
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < sz1; i += blockDim.x * gridDim.x) {
	for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < sz2; j += blockDim.y * gridDim.y) {
	    int score = 0;
	    // calculate ftable entry at this point
	    if (d1[i] == d2[j]) {
	        score = reward;
	    } else {
	        score = penalty;
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
	    /*
	    if (ftable[i*sz2+j]>0) {
	        printf("%d\n", i*sz2+j);
	    }
	    */
	}	
    }
    // now go through whole ftable and find max value - 
    // this should be parallelized too
    // maybe I will have another kernel for this
    /*
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
    */
}

int main(int argc, char* argv[]){
    FILE *seq1;
    FILE *seq2;
    int gap;
    int reward;
    int penalty;
    int grid;
    int block;
    if (argc == 8) {
        seq1 = fopen(argv[1], "rb");
	seq2 = fopen(argv[2], "rb");
        gap = atoi(argv[3]);
	reward = atoi(argv[4]);
	penalty = atoi(argv[5]);
	grid = atoi(argv[6]);
	block = atoi(argv[7]);
    }
    else {
        printf("Incorrect number of arguments. Arguments should take form: sequence 1, sequence 2, gap penalty, score for a match, penalty for a mismatch, grid dim, block dim.\n");
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
    /*
    I don't actually need these yet
    char* match1 = (char*) malloc(sz1*sizeof(char));
    char* match2 = (char*) malloc(sz2*sizeof(char));
    char* dmatch1;
    char* dmatch2;
    cudaMalloc((void**)&dmatch1, sz1*sizeof(char));
    cudaMalloc((void**)&dmatch2, sz2*sizeof(char));
    */
    int* Fftable = (int*) malloc(sz1*sz2*sizeof(int));
    int* ftable;
    cudaMalloc((void**)&ftable, sz1*sz2*sizeof(int));
    cudaMemset(ftable, 0, sz1*sz2*sizeof(int));
    
    int* Ssource1 = (int*) malloc(sz1*sz2*sizeof(int));
    int* source1;
    cudaMalloc((void**)&source1, sz1*sz2*sizeof(int));
    cudaMemset(source1, 0, sz1*sz2*sizeof(int));

    int* Ssource2 = (int*) malloc(sz1*sz2*sizeof(int));
    int* source2;
    cudaMalloc((void**)&source2, sz1*sz2*sizeof(int));
    cudaMemset(source2, 0, sz1*sz2*sizeof(int));
    

    // create start and end events
    
    cudaEvent_t tick, tock;
    cudaEventCreate(&tick);
    cudaEventCreate(&tock);
    // call funciton and report time
    cudaEventRecord(tick, 0);
    align_kernel<<<grid, block>>>(d1, d2, ftable, source1, source2, sz1, sz2, gap, reward, penalty);
    cudaEventRecord(tock, 0);
    cudaEventSynchronize(tock);
    float time;
    cudaEventElapsedTime(&time, tick, tock);
    printf("time to align: %0.6f ms\n", time);

    cudaMemcpy(Fftable, ftable, sz1*sz2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ssource1, source1, sz1*sz2*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ssource2, source2, sz1*sz2*sizeof(int), cudaMemcpyDeviceToHost);
    // create new ftable for finding max val
    int* ftablefind;
    cudaMalloc((void**)&ftablefind, sz1*sz2*sizeof(int));
    cudaMemcpy(ftablefind, ftable, sz1*sz2*sizeof(int), cudaMemcpyHostToDevice);
    // create spot place
    int* spots = (int*) malloc(3*sizeof(int));
    int* d_spots;
    cudaMalloc((void**)&d_spots, 3*sizeof(int));
    cudaMemset(d_spots, 0, 3*sizeof(int));

    cudaEvent_t tick1, tock1;
    cudaEventCreate(&tick1);
    cudaEventCreate(&tock1);
    // call funciton and report time
    cudaEventRecord(tick1, 0);
    find_kernel<<<grid, block>>>(ftablefind, d_spots, sz1, sz2);
    cudaEventRecord(tock1, 0);
    cudaEventSynchronize(tock1);
    float time1;
    cudaEventElapsedTime(&time1, tick1, tock1);
    printf("time to find optimal sequence: %0.6f ms\n", time1);
    cudaMemcpy(spots, d_spots, 3*sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Best spot: %d %d\n", spots[0], spots[1]);
    //printf("Max score: %d \n", spots[2]);

    // at this point we have our optimally alinged spot. 
    // now we just want to reconstruct the sequence. parallelizing this
    // will be a nightmare and we've already taken care of the most
    // computationally expensive part, so this can just be a straight 
    // while loop.

    // I don't care if this is inefficient I just don't know how long these 
    // sequences will be
    
    /*
    for (int i = 0; i < sz1; i++) {
        for (int j = 0; j < sz2; j++) {
            printf("(%d %d) ", Ssource1[i*sz2+j], Ssource2[i*sz2+j]);
	}
	printf("\n");
    }
    */
    char align1[sz1+sz2];
    char align2[sz1+sz2];
    int newspot[3];
    int aligncount = 0;
    while ((spots[0] != 0) && (spots[1] != 0)) {
	//printf("%s\n%s\n", align1, align2);
	newspot[0] = Ssource1[spots[0]*sz2+spots[1]];
	newspot[1] = Ssource2[spots[0]*sz2+spots[1]];
	//printf("(%d, %d) \n", spots[0], spots[1]);
	//printf("%c\n%c\n", text1[spots[0]-1], text2[spots[1]-1]);
	//printf("%s\n%s\n", align1, align2);
	if ((newspot[0]+1 == spots[0]) && (newspot[1]+1==spots[1])) {
	    //printf("aaaaa\n");
	    //printf("%c\n", text2[spots[1]-1]);
            align1[aligncount] = text1[spots[0]-1];
            align2[aligncount] = text2[spots[1]-1];
	    //printf("%s\n", align2);
	    //aligncount = aligncount + 1;
        }
	else if (newspot[0] == spots[0]) {
	    //printf("bbbbb\n");
            align1[aligncount] = '_';
	    align2[aligncount] = text2[spots[1]-1];
	    //aligncount = aligncount + 1;
        }
	else if (newspot[1] == spots[1]) {
	    //printf("cccc\n");
	    align1[aligncount] = text1[spots[0]-1];
	    align2[aligncount] = '_';
	    //aligncount = aligncount + 1;
	}
	//printf("%c", align1[aligncount-1]);
	//printf("%s\n%s\n", align1, align2);
	aligncount = aligncount + 1;
	align1[aligncount] = '\0';
	align2[aligncount] = '\0';
	spots[0] = newspot[0];
	spots[1] = newspot[1];
	//printf("%d %d \n", spots[0], spots[1]);
    }
    //printf("aa %s aa \n", align1);
    //align1[aligncount] = '\0';
    //align2[aligncount] = '\0';
    //printf("%s", align1);
    printf("Visual representation of alignment (sequences are reversed):\n");
    printf("%s\n%s\n", align1, align2);

    float identity = 0.0;
    for (int i = 0; i < aligncount; i++) {
        if (align1[i] == align2[i]) {
            identity += 1;
        }
    }
    identity = identity / aligncount;
    printf("Identity percent:\n");
    printf("%.2f\n", identity*100);
    printf("Alignment score:\n");
    printf("%d\n", spots[2]);
}
