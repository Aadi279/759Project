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
        //printf("floor(z_max/lH): %f\n", floor(z_max/lH));
        //printf("ceil(z_min/lH): %f\n", ceil(z_min/lH));
        layersContained = floor(z_max/lH) - ceil(z_min / lH) + 1;
        //if (layersContained*lH + z_min == z_max)
        //layersContained++;
        layersInTris[gtid] = layersContained;
    }

    //TODO: Handle boundary case of triangles which intersect layer at a point or are completely coplanar
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
__device__ void get_intersection(float x0, float x1,
                                 float y0, float y1,
                                 float z0, float z1, float zp,
                                 float* x_r, float* y_r,
                                 bool* parallel,
                                 bool* non_intersecting) {

    //printf("Getting intersection...\nx0:%f\nx1:%f\ny0:%f\ny1:%f\nz0:%f\nz1:%f\nzp:%f\n", x0, x1, y0, y1, z0, z1, zp);

    //TODO: Put this check outside and handle by putting both points into the segment list. This case represents a planar line segment
    float denom = (z1 - z0);

    *non_intersecting = false;
    *parallel = false;

    if(denom == 0) {
        *parallel = true;
        return;
    }

    float t = (zp - z0) / denom;

    if(t < 0 || t > 1) {
        *non_intersecting = true;
        return;
    }

    *x_r = x0 + t * (x1 - x0);
    *y_r = y0 + t * (y1 - y0);
    return;
}

__global__ void calculateIntersections(float* xs, float* ys, float* zs, int* layersInTri, int* startIndexInSegments, float* seg_x, float* seg_y, float* seg_l, const float lH, const int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    //TODO: Needs an external loop for `triIndex` instead of using gtid directly
    // Because we may have more tris than threads
    if(gtid < n) {
        int stri = gtid*3;
        float x0 = xs[stri];
        float y0 = ys[stri];
        float z0 = zs[stri];
        float x1 = xs[stri + 1];
        float y1 = ys[stri + 1];
        float z1 = zs[stri + 1];
        float x2 = xs[stri + 2];
        float y2 = ys[stri + 2];
        float z2 = zs[stri + 2];
        int bottomLayer = ceil(min(min(z0, z1), z2) / lH);


        int localStartIndexInSegments;
        if(gtid < 1) {
            localStartIndexInSegments = 0;
        } else {
            localStartIndexInSegments = startIndexInSegments[gtid-1]*2;
        }


        // Iterate through layers
        int layer; float zp;
        float* x_r = (float*)malloc(sizeof(float)); //TODO: Put this in shared memory to boost performance
        float* y_r = (float*)malloc(sizeof(float));
        bool* non_intersecting = (bool*)malloc(sizeof(float));
        bool* parallel = (bool*)malloc(sizeof(float));
        int intersectionsFound;

        int segmentIndex;
        for(int i = 0; i < layersInTri[gtid]; i++) {
            layer = bottomLayer + i;
            zp = layer * lH;

            intersectionsFound = 0;

            segmentIndex = localStartIndexInSegments + 2*i;


            //if(layer == 1) {
            //    printf("\nlayer 1...\n");
            //    printf("i: %d\n", i);
            //    printf("Bottom Layer: %d\n", bottomLayer);
            ////    printf("Layers in tri: %d\n", layersInTri[gtid]);
            //}

            //if(i==1) {
            //    printf("\n at i=1...\n");
            //    printf("layer=%d\n", layer);
            //    printf("bottomLayer=%d\n", bottomLayer);
            //    printf("gtid=%d\n", gtid);
            //}

            get_intersection(x0, x1, y0, y1, z0, z1, zp, x_r, y_r, parallel, non_intersecting);
            if(!(*non_intersecting)){
                seg_x[segmentIndex] = *x_r;
                seg_y[segmentIndex] = *y_r;
                seg_l[segmentIndex] = layer;
                segmentIndex++;
            }

            get_intersection(x1, x2, y1, y2, z1, z2, zp, x_r, y_r, parallel, non_intersecting);
            if(!(*non_intersecting)){
                seg_x[segmentIndex] = *x_r;
                seg_y[segmentIndex] = *y_r;
                seg_l[segmentIndex] = layer;
                segmentIndex++;
            }

            get_intersection(x2, x0, y2, y0, z2, z0, zp, x_r, y_r, parallel, non_intersecting);
            if(!(*non_intersecting)){
                seg_x[segmentIndex] = *x_r;
                seg_y[segmentIndex] = *y_r;
                seg_l[segmentIndex] = layer;
                segmentIndex++;
            }

            //get_intersection(x0, x1, y0, y1, z0, z1, zp, x_r, y_r, parallel, non_intersecting);
            //if(!(*non_intersecting)){
            //    seg_x[localStartIndexInSegments+i] = *x_r;
            //    seg_y[localStartIndexInSegments+i] = *y_r;
            //    seg_l[localStartIndexInSegments+i] = layer;
            //    intersectionsFound++;
            //}

            //get_intersection(x1, x2, y1, y2, z1, z2, zp, x_r, y_r, parallel, non_intersecting);
            //if(!(*non_intersecting)){
            //    seg_x[localStartIndexInSegments+i+intersectionsFound] = *x_r;
            //    seg_y[localStartIndexInSegments+i+intersectionsFound] = *y_r;
            //    seg_l[localStartIndexInSegments+i+intersectionsFound] = layer;
            //    intersectionsFound++;
            //}

            //get_intersection(x2, x0, y2, y0, z2, z0, zp, x_r, y_r, parallel, non_intersecting);
            //if(!(*non_intersecting)){
            //    seg_x[localStartIndexInSegments+i+intersectionsFound] = *x_r;
            //    seg_y[localStartIndexInSegments+i+intersectionsFound] = *y_r;
            //    seg_l[localStartIndexInSegments+i+intersectionsFound] = layer;
            //    intersectionsFound++;
            //}
            // TODO: Handle boundary cases for planar triangles and tangentially intersecting triangles
        }
    }
//    for(int i=0; i < 3; i++) {
//        sx[stri + i]
//    }

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

    const int n = 1;
    const int N = n*3;


//    float x[N] = {0.,  0., 1., 0., 1., 1., 0., 0., 0.};
//    float y[N] = {0.,  0., 0., 0., 0., 0., 0., 1., 1.};
//    float z[N] = {.25, 1.25, 1.25, 0.25, 0.25, 1.25, 1.25, 2.6, 2.6};
    const float layerHeight = .2;
    float x[N] = {.5,  .5, .5};
    float y[N] = {1.5, 2.5, 3.5};
    float z[N] = {.5,   1.5, .5};


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

    //print_vector("layersInTris", layersInTris);

    thrust::device_vector<int> intersectionSegmentsIndexStart(n, 0);
    thrust::inclusive_scan(layersInTris.begin(), layersInTris.end(), intersectionSegmentsIndexStart.begin());

    //print_vector("intersectionSegmentsIndexStart", intersectionSegmentsIndexStart);

    int totalIntersections = intersectionSegmentsIndexStart[intersectionSegmentsIndexStart.size()-1];

    //printf("totalIntersections: %d\n", totalIntersections);

    // Intersection segment coordinate arrays
    thrust::device_vector<float> iscx(totalIntersections*2, 0);
    thrust::device_vector<float> iscy(totalIntersections*2, 0);
    thrust::device_vector<float> iscl(totalIntersections*2, 0);
    float* iscx_p = thrust::raw_pointer_cast( &iscx[0] );
    float* iscy_p = thrust::raw_pointer_cast( &iscy[0] );
    float* iscl_p = thrust::raw_pointer_cast( &iscl[0] );
    float* x_p = thrust::raw_pointer_cast( &dx[0] );
    float* y_p = thrust::raw_pointer_cast( &dy[0] );
    int* intersectionSegmentsIndexStart_p = thrust::raw_pointer_cast( &intersectionSegmentsIndexStart[0]);

    calculateIntersections<<<2, 8>>>(x_p, y_p, z_p, layersInTris_p, intersectionSegmentsIndexStart_p, iscx_p, iscy_p, iscl_p, layerHeight, n);

    printf("totalIntersections: %d\n", totalIntersections);
    print_vector("layersInTris", layersInTris);

    print_vector("iscx", iscx);
    print_vector("iscy", iscy);
    print_vector("iscl", iscl);


    // Copy over our layer vector so that we can use it as a key vector twice for stable_sort_by_key
    thrust::device_vector<float> iscl2(totalIntersections*2, 0);
    thrust::copy(iscl.begin(), iscl.end(), iscl2.begin());

    thrust::stable_sort_by_key(thrust::device, iscl.begin(), iscl.end(), iscx.begin());
    thrust::stable_sort_by_key(thrust::device, iscl2.begin(), iscl2.end(), iscy.begin());

    print_vector("iscx", iscx);
    print_vector("iscy", iscy);
    print_vector("iscl", iscl);


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