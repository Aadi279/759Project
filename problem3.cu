#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/functional.h>

__device__ __host__ bool moreThan5(int amount) {
    return amount > 5;
}

int main(int argc, char* argv[]) {

    float time = 0.f;

    const int N = 15;
    int day[N]         = {0, 0, 1, 2, 5, 5, 6, 6, 7, 8, 9, 9, 9, 10, 11};
    int site[N]        = {2, 3, 0, 1, 1, 2, 0, 1, 2, 1, 3, 4, 0, 1,  2 };
    int measurement[N] = {9, 5, 6, 3, 3, 8, 2, 6, 5,10, 9,11, 8, 4,  1 };

    // START MY CODE
//    thrust::day<int> H(N, 0);
//    thrust::site<int> H(N, 0);
//    thrust::measurement<int> H(N, 0);

//    thrust::copy(day_a, day_a+dfasdfN, day.begin());
//    thrust::copy(site_a, site_a+N, site.begin());
//    thrust::copy(measurement_a, measurement_a+N, measurement.begin());

    // Timing things
    cudaEvent_t startEvent_inc, stopEvent_inc;
    cudaEventCreate(&startEvent_inc);
    cudaEventCreate(&stopEvent_inc);
    cudaEventRecord(startEvent_inc,0);

    thrust::equal_to<int> binary_pred;
    thrust::plus<int> binary_op;
    thrust::maximum<int> max_fn;

    int day_out[N];
    int measurement_maxes[N];
    ////Part A:
    thrust::reduce_by_key(day, day+N, measurement, day_out, measurement_maxes, binary_pred, max_fn);

    int numSites = thrust::count_if(measurement_maxes, measurement_maxes+11, moreThan5);

    printf("%d\n", numSites);

    ////Part B:
    thrust::sort_by_key(site, site + N, measurement);


    thrust::reduce_by_key(thrust::host,
                          site, site + N, measurement,
                          site, measurement,
                          binary_pred, binary_op);

    for(int i=0; i<5; i++) {
        printf("%d ", measurement[i]);
    }
    printf("\n");

    ////CODE MEAT END

    // Finish timing
    cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(&time, startEvent_inc, stopEvent_inc);

    // END MY CODE

    //printf("%d\n%f\n%f\n\n",N,cuda_out[N-1],time);

    //free resources
    //free(in); free(out); free(cuda_out);
    return 0;
}