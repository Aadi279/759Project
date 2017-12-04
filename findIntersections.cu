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


/**
 * Generates an array of ints representing the height (in number of layers) which each triangle spans
 * @param zs
 * @param layersInTris
 * @param numberOfTris
 * @return
 */
__global__ void layersInEachTriangle(float* zs, int* layersInTris, int numberOfTris, const float lH) {
    // Represents the triangle we're checking currently
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    float z0 = zs[gtid*3];
    float z1 = zs[gtid*3+1];
    float z2 = zs[gtid*3+2];
    int layersContained;
    if(gtid < numberOfTris) {
        float z_max = max(max(z0, z1), z2);
        float z_min = min(min(z0, z1), z2);
        layersContained = ceil(z_max/lH) - floor(z_min / lH);
        if (layersContained*lH + z_min == z_max)
            layersContained++;
        layersInTris[gtid] = layersContained;
    }
}

//__device__ float dot(float* xs, float* ys, float* zs, int i0, int i1) {
//
//}

__global__ void calculateIntersections(float* xs, float* ys, float* zs, float* seg_x, float* seg_y, float* seg_l) {

}

// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
    typedef typename Vector::value_type T;
    std::cout << "  " << std::setw(20) << name << "  ";
    thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {

    float time = 0.f;

    const int n = 3;
    const int N = n*3;
    const float layerHeight = .5;


    float x[N] = {0., 0., 1., 0., 1., 1., 0., 0., 0.};
    float y[N] = {0., 0., 0., 0., 0., 0., 0., 1., 1.};
    float z[N] = {0., 1., 1., 0., 0., 1., 1., 2.5, 2.5};


    // Timing things
    cudaEvent_t startEvent_inc, stopEvent_inc;
    cudaEventCreate(&startEvent_inc);
    cudaEventCreate(&stopEvent_inc);
    cudaEventRecord(startEvent_inc,0);

    thrust::device_vector<float> dx(N, 0);
    thrust::device_vector<float> dy(N, 0);
    thrust::device_vector<float> dz(N, 0);
    thrust::copy(x, x+N, dx.begin());
    thrust::copy(y, y+N, dy.begin());
    thrust::copy(z, z+N, dz.begin());

    print_vector("x", dx);

    thrust::device_vector<int> layersInTris(n, 0);
    int*  layersInTris_p = thrust::raw_pointer_cast( &layersInTris[0] );
    float*  z_p = thrust::raw_pointer_cast( &dz[0] );
    layersInEachTriangle<<<2, 8>>>(z_p, layersInTris_p, n, layerHeight);

    print_vector("layersInTris", layersInTris);

    thrust::device_vector<int> intersectionSegmentsIndexStart(n, 0);
    thrust::inclusive_scan(layersInTris.begin(), layersInTris.end(), intersectionSegmentsIndexStart.begin());

    print_vector("layersInTris", intersectionSegmentsIndexStart);

    int totalIntersections = intersectionSegmentsIndexStart[intersectionSegmentsIndexStart.size()-1];

    printf("totalIntersections: %d", totalIntersections);

    // Intersection segment coordinate arrays
    thrust::device_vector<float> iscx(totalIntersections, 0);
    thrust::device_vector<float> iscy(totalIntersections, 0);
    thrust::device_vector<float> iscl(totalIntersections, 0);
    float* iscx_p = thrust::raw_pointer_cast( &iscx[0] );
    float* iscy_p = thrust::raw_pointer_cast( &iscy[0] );
    float* iscl_p = thrust::raw_pointer_cast( &iscl[0] );
    float* x_p = thrust::raw_pointer_cast( &x[0] );
    float* y_p = thrust::raw_pointer_cast( &y[0] );

    calculateIntersections<<<2, 8>>>(x_p, y_p, z_p, iscx_p, iscy_p, iscl_p);


//    thrust::equal_to<int> binary_pred;
//    thrust::plus<int> binary_op;
//    thrust::maximum<int> max_fn;
//
//    int day_out[N];
//    int measurement_maxes[N];
//    ////Part A:
//    thrust::reduce_by_key(day, day+N, measurement, day_out, measurement_maxes, binary_pred, max_fn);
//
//    int numSites = thrust::count_if(measurement_maxes, measurement_maxes+11, moreThan5);
//
//    printf("%d\n", numSites);
//
//    ////Part B:
//    thrust::sort_by_key(site, site + N, measurement);
//
//
//    thrust::reduce_by_key(thrust::host,
//                          site, site + N, measurement,
//                          site, measurement,
//                          binary_pred, binary_op);
//
//    for(int i=0; i<5; i++) {
//        printf("%d ", measurement[i]);
//    }
//    printf("\n");
//

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