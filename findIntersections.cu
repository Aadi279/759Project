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
                                 float &x_r, float &y_r,
                                 bool &parallel,
                                 bool &non_intersecting) {

    //printf("Getting intersection...\nx0:%f\nx1:%f\ny0:%f\ny1:%f\nz0:%f\nz1:%f\nzp:%f\n", x0, x1, y0, y1, z0, z1, zp);

    //TODO: Put this check outside and handle by putting both points into the segment list. This case represents a planar line segment
    float denom = (z1 - z0);

    non_intersecting = false;
    parallel = false;

    if(denom == 0) {
        parallel = true;
        if(z1 != zp) {
            non_intersecting = true;
        }
        return;
    }

    float t = (zp - z0) / denom;

    if(t < 0 || t > 1) {
        non_intersecting = true;
        //printf("t=%f, x=%f, y=%f, non_intersecting=%d\n", t, *x_r, *y_r, *non_intersecting);
        return;
    }

    x_r = x0 + t * (x1 - x0);
    y_r = y0 + t * (y1 - y0);

    //printf("t=%f, x=%f, y=%f, non_intersecting=%d\n", t, *x_r, *y_r, *non_intersecting);
    return;
}

__device__ void addToSegments(const float x0, const float x1, const float y0, const float y1, const float z0, const float z1, int layer, const float zp,  int &si, float* seg_x, float* seg_y, float* seg_l) {
    float x_r; float y_r;
    bool parallel; bool non_intersecting;
    get_intersection(x0, x1, y0, y1, z0, z1, zp, x_r, y_r, parallel, non_intersecting);
    if(!(non_intersecting)){
        //if(parallel) {
        //    seg_x[si] = x0;
        //    seg_y[si] = y0;
        //    seg_l[si] = layer;
        //    si++;
        //    seg_x[si] = x1;
        //    seg_y[si] = y1;
        //    seg_l[si] = layer;
        //    si++;
        //} else {
        seg_x[si] = x_r;
        seg_y[si] = y_r;
        seg_l[si] = layer;
        si++;
        //}
    }
}

__device__ bool isMiddle(const float z0, const float z1, const float z2) {
    return ((z0 < z1) && (z1 < z2)) || ((z2 < z1) && (z1 < z0));
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
        float x_r;
        float y_r;
        bool non_intersecting;
        bool parallel;

        int segmentOffset;
        int si=localStartIndexInSegments;

        char pointsOnLayerPlane;
        for(int i = 0; i < layersInTri[gtid]; i++) {
            layer = bottomLayer + i;
            zp = layer * lH;
            pointsOnLayerPlane = 0;
            if(zp == z0) {
                pointsOnLayerPlane++;
            }
            if(zp == z1) {
                pointsOnLayerPlane++;
            }
            if(zp == z2) {
                pointsOnLayerPlane++;
            }
            if(pointsOnLayerPlane == 0) {
                // TODO: We'll also have to handle the case where a triangle crosses a layer
                // and the layer lies right on the point
                // Honestly, I think the thing to do would be to just check whether the triangle
                // has any points on this layer in a separate handler

                addToSegments(x0, x1, y0, y1, z0, z1, layer, zp, si, seg_x, seg_y, seg_l);
                addToSegments(x1, x2, y1, y2, z1, z2, layer, zp, si, seg_x, seg_y, seg_l);
                addToSegments(x2, x0, y2, y0, z2, z0, layer, zp, si, seg_x, seg_y, seg_l);

                // TODO: Handle boundary cases for planar triangles and tangentially intersecting triangles
            } else if(pointsOnLayerPlane < 3) {
                if(zp == z0) {
                    seg_x[si] = x0;
                    seg_y[si] = y0;
                    seg_l[si] = layer;
                    si++;
                    if(pointsOnLayerPlane == 1) {
                        if(isMiddle(z1,z0,z2)){
                            // TODO: Have to add a check here to see if it's the middle point which sits on the edge
                            addToSegments(x1, x2, y1, y2, z1, z2, layer, zp, si, seg_x, seg_y, seg_l);
                        } else {
                            seg_x[si] = x0;
                            seg_y[si] = y0;
                            seg_l[si] = layer;
                            si++;
                        }
                    }
                }
                if(zp == z1) {
                    seg_x[si] = x1;
                    seg_y[si] = y1;
                    seg_l[si] = layer;
                    si++;
                    if(pointsOnLayerPlane == 1) {
                        if(isMiddle(z0, z1, z2)) {
                            addToSegments(x2, x0, y2, y0, z2, z0, layer, zp, si, seg_x, seg_y, seg_l);
                        } else {
                            seg_x[si] = x1;
                            seg_y[si] = y1;
                            seg_l[si] = layer;
                            si++;
                        }
                    }
                }
                if(zp == z2) {
                    seg_x[si] = x2;
                    seg_y[si] = y2;
                    seg_l[si] = layer;
                    si++;
                    if(pointsOnLayerPlane == 1) {
                        if(isMiddle(z0, z1, z2)) {
                            addToSegments(x0, x2, y0, y2, z0, z2, layer, zp, si, seg_x, seg_y, seg_l);
                        } else {
                            seg_x[si] = x2;
                            seg_y[si] = y2;
                            seg_l[si] = layer;
                            si++;
                        }
                    }
                }
            }
        }
    }
}

__global__ void calculateScanPointsPerSegment(float* iscy_p, float* numScanPointsPerSegment, const float scanHeight, int totalIntersections)
{
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid < totalIntersections) {
        int y0 = iscy_p[gtid*2];
        int y1 = iscy_p[gtid*2+1];
        int ymax = max(y0, y1);
        int ymin = min(y1, y0);
        numScanPointsPerSegment[gtid] = floor(ymax / scanHeight) - ceil(ymin / scanHeight);
    }
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

    const int n = 4;
    const int N = n*3;


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

    const float layerHeight = 1.;
    const float scanHeight = .5;
    float x[N] = {0., 0., 0.,   0., 0., 2.,   0., 0., 2.,   0., 0., 2.,};
    float y[N] = {0., 2., 0.,   0., 0., 0.,   0., 2., 0.,   0., 2., 0.,};
    float z[N] = {0., 2., 4.,   0., 4., 2.,   0., 2., 2.,   4., 2., 2.,};

    //const float layerHeight = 1.;
    //float x[N] = {0., 0., 0.,   0., 0., 2.};
    //float y[N] = {0., 2., 0.,   0., 0., 0.};
    //float z[N] = {0., 2., 2.,   0., 2., 2.};


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
    print_vector("y", dy);
    print_vector("z", dz);

    thrust::device_vector<int> layersInTris(n, 0);
    int*  layersInTris_p = thrust::raw_pointer_cast( &layersInTris[0] );
    float*  z_p = thrust::raw_pointer_cast( &dz[0] );
    layersInEachTriangle<<<2, 8>>>(z_p, layersInTris_p, n, layerHeight);

    //print_vector("layersInTris", layersInTris);

    thrust::device_vector<int> intersectionSegmentsIndexStart(n, 0);
    thrust::inclusive_scan(layersInTris.begin(), layersInTris.end(), intersectionSegmentsIndexStart.begin());

    //print_vector("intersectionSegmentsIndexStart", intersectionSegmentsIndexStart);

    // Might want to do this asyncronously? Could be a costly operation
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

    //thrust::stable_sort_by_key(thrust::device, iscx.begin(), iscx.end(), iscl.begin());
    //thrust::stable_sort_by_key(thrust::device, iscy.begin(), iscy.end(), iscl2.begin());
    thrust::stable_sort_by_key(thrust::device, iscl.begin(), iscl.end(), iscx.begin());
    thrust::stable_sort_by_key(thrust::device, iscl2.begin(), iscl2.end(), iscy.begin());

    // TODO: we might put a pruning algorithm here to get rid of duplicate segments
    // This would spare us in instances where we have non-manifold shapes in 3D space
    // and allow us to say that more than 2 sequential duplicates indicates a non-manifold junction and we should
    // continue through

    print_vector("iscx", iscx);
    print_vector("iscy", iscy);
    print_vector("iscl", iscl);

    thrust::device_vector<float> scanPointsPerSegment(totalIntersections, 0);
    float* scanPointsPerSegment_p = thrust::raw_pointer_cast( &scanPointsPerSegment[0] );
    calculateScanPointsPerSegment<<<2, 32>>>(iscy_p, scanPointsPerSegment_p, scanHeight, totalIntersections); // Gotta get the total intersections
    print_vector("scanPointsPerSegment", scanPointsPerSegment);


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