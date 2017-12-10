#include <iostream>
#include <iomanip>
#include <iterator>

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


//__device__ __host__ bool moreThan5(int amount) {
//    return amount > 5;
//}

///__device__ void fuun(int x) {
///    printf("%d\n", x);
///}


__global__ void testy(int n){
    int z = 2;
    printf("%d", z);
    return; 
    //fuun(z);
}

//__device__ float dot(float* xs, float* ys, float* zs, int i0, int i1) {
//
//}

/**
 *
 * @param x0
 * @param x1
 * @param y0
 * @param y1
 * @param z0
 * @param z1
 * @param zp z of the plane
 * @param x_r result of the intersection
 * @param y_r result of the intersection
 * @return
 */
int main(int argc, char* argv[]) {


    printf("ehll");
    
    testy<<<1,1>>>(2);

//    float x[N] = {0.,  0., 1., 0., 1., 1., 0., 0., 0.};
//    float y[N] = {0.,  0., 0., 0., 0., 0., 0., 1., 1.};
//    float z[N] = {.25, 1.25, 1.25, 0.25, 0.25, 1.25, 1.25, 2.6, 2.6};

    //const float layerHeight = .2;
    //float x[N] = {.5,  .5, .5};
    //float y[N] = {1.5, 2.5, 3.5};
    //float z[N] = {.5,   1.5, .5};

    //const float layerHeight = 1.;
    //float x[N] = {0., 0., 0.,   0., 0., 2.,   0., 0., 2.};
    //float y[N] = {0., 2., 0.,   0., 0., 0.,   0., 2., 0.};
    //float z[N] = {0., 2., 2.,   0., 2., 2.,   0., 2., 2.};


    // END MY CODE

    //printf("%d\n%f\n%f\n\n",N,cuda_out[N-1],time);

    //free resources
    //free(in); free(out); free(cuda_out);
    return 0;
}
