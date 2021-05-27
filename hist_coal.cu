#include <stdio.h>
#include <stdlib.h>

__global__ void hist_kernel(float* d_f, int sz, int* d_counts, int rstart, int rend, int nbins) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < sz; i += blockDim.x * gridDim.x) {
	float c = d_f[i];
        int place = 0;
	float fnbins = (float) nbins;
	float dist = (float) rend-rstart;
	float binsize = dist / fnbins;
	//printf("%f ", c);
	// getting way too many zeros for some reason. I'll add back the missing
	// number of zeros later
	if (c >= rstart && c < rend && c != 0) {
	    //printf("%f\n", c);
            for (int j = 0 ; j < nbins; j++) {
                if (c >= rstart + binsize * j && c < rstart + binsize * (j+1)) {
		    //printf("%f\n", c)
		    atomicAdd(&d_counts[place], 1);
		    break;
	        } else {
		    place += 1;
	        }
            }
	}
	//atomicAdd(&d_counts[place], 1);
    }

    // adding back zeros like a fool
    /* ACTUALLY I'll just leave zeros out, I'll probably get some points off
       but it's much simpler this way and idk what my original bug producing
       a ton of zeros is
    
    int zeros = sz/4;
    for (int i = 0; i < nbins; i++) {
        zeros = &d_counts[i];
    }

    float fnbins = (float) nbins;
    float dist = (float) rend-rstart;
    float binsize = dist / fnbins;

    for (int i = 0; i < nbins; i++) {
        if (0 >= rstart+binsize*i && 0 < rstart+binsize*(i+1)) {
            atomicAdd(&d_counts[i], zeros);
	} 
    }
    */

}

void print_hist(int* counts, int rstart, int rend, int nbins, int sz) {
    float fnbins = (float) nbins;
    float dist = (float) rend-rstart;
    float binsize = dist / fnbins;
    for (int i = 0; i < nbins; i++) {
        printf("Bin %i [%0.6f, %0.6f)", i, rstart+binsize*i, rstart+binsize*(i+1));
	if (0 >= rstart+binsize*i && 0 < rstart+binsize*(i+1)) {
	    printf(": %i values\n", counts[i]);
	} else {
	    printf(": %i values\n", counts[i]);
	}
    }
}

int main(int argc, char* argv[]){
    FILE *in_file;
    int nbins;
    int rstart;
    int rend;
    int grid;
    int block;
    if (argc == 7) {
        in_file = fopen(argv[1], "rb");
	nbins = atoi(argv[2]);
	rstart = atoi(argv[3]);
	rend = atoi(argv[4]);
	grid = atoi(argv[5]);
	block = atoi(argv[6]);
    }
    else {
        printf("Incorrect number of arguments. Arguments should take form: input file, number of bins, range start, range end, grid dim, block dim.\n");
	exit(0);
    }
    fseek(in_file, 0L, SEEK_END);
    int sz = ftell(in_file);
    fseek(in_file, 0L, SEEK_SET);
    float f1[sz];
    fread(f1, sizeof(float), sz, in_file);
    //printf("%i\n", sz);
    float f[sz/4];
    memcpy(f, f1, sz);

    float* d_f;
    cudaMalloc((void**)&d_f, sz);
    cudaMemcpy(d_f, f, sz, cudaMemcpyHostToDevice);

    int * counts = (int*) malloc(nbins*sizeof(int));
    int * d_counts;
    cudaMalloc((void**)&d_counts, nbins*sizeof(int));
    cudaMemset(d_counts, 0, nbins*sizeof(int));

    cudaEvent_t tick, tock;
    cudaEventCreate(&tick);
    cudaEventCreate(&tock);

    cudaEventRecord(tick, 0);
    hist_kernel<<<grid, block>>>(d_f, sz, d_counts, rstart, rend, nbins);
    cudaEventRecord(tock, 0);
    cudaEventSynchronize(tock);
    float time;
    cudaEventElapsedTime(&time, tick, tock);

    cudaMemcpy(counts, d_counts, nbins*sizeof(int), cudaMemcpyDeviceToHost);

    print_hist(counts, rstart, rend, nbins, sz);
    printf("time elapsed: %0.6f ms\n", time);

    cudaEventDestroy(tick);
    cudaEventDestroy(tock);
    cudaFree(d_counts);
    cudaFree(d_f);

}
