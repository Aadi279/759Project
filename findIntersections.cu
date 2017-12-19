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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

//TODO: It would be great to split this into separate files, compile times are getting a bit long and it's
// more difficult to work with on Euler as a huge piece like this.

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
        layersContained = floor(z_max/lH) - ceil(z_min / lH) + 1;
        layersInTris[gtid] = layersContained;
    }

    //TODO: Handle boundary case of triangles which intersect layer at a point or are completely coplanar (This might be comprehensive now, test!)
}

/**
 * Finds the intersection between a triangle and a plane
 *
 * @param x0 x coordinate of the current triangle edge's startpoint
 * @param x1 x coordinate of the current triangle edge's endpoint
 * @param y0 y coordinate of the current triangle edge's startpoint
 * @param y1 y coordinate of the current triangle edge's endpoint
 * @param z0 z coordinate of the current triangle edge's startpoint
 * @param z1 z coordinate of the current triangle edge's endpoint
 * @param zp z coordinate of the plane
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
        return;
    }

    x_r = x0 + t * (x1 - x0);
    y_r = y0 + t * (y1 - y0);
    return;
}

/**
 * Given a triangle edge and a layer spec, find the intersection between the edge and the layer if there is one
 * and place its coordinates into the boundary segments arrays
 *
 * @param x0 x coordinate of the current triangle edge's startpoint
 * @param x1 x coordinate of the current triangle edge's endpoint
 * @param y0 y coordinate of the current triangle edge's startpoint
 * @param y1 y coordinate of the current triangle edge's endpoint
 * @param z0 z coordinate of the current triangle edge's startpoint
 * @param z1 z coordinate of the current triangle edge's endpoint
 * @param layer Integer index of the current layer
 * @param zp Float z coordinate of the currrent layer
 * @param si Current index within each of the boundary segment arrays
 * @param seg_x List of boundary segment x coordinates (stride 2)
 * @param seg_y List of boundary segment y coordinates (stride 2)
 * @param seg_l List of boundary segment z coordinates (stride 2)
 * @return
 */
__device__ void updateContourSegmentsIfIntersects(const float x0, const float x1, const float y0, const float y1, const float z0, const float z1, int layer, const float zp,  int &si, float* seg_x, float* seg_y, int* seg_l) {
    float x_r; float y_r;
    bool parallel; bool non_intersecting;
    get_intersection(x0, x1, y0, y1, z0, z1, zp, x_r, y_r, parallel, non_intersecting);
    if(!(non_intersecting)){
        seg_x[si] = x_r;
        seg_y[si] = y_r;
        seg_l[si] = layer;
        si++;
    }
}

/**
 * Returns true if the 2nd argument is between the exterior arguments
 * @param z0
 * @param z1
 * @param z2
 * @return
 */
__device__ bool isMiddle(const float z0, const float z1, const float z2) {
    return ((z0 < z1) && (z1 < z2)) || ((z2 < z1) && (z1 < z0));
}

/**
 * Iterate through every triangle, find intersections for each triangle with the layers they span
 * Populate the contour segment arrays with all found intersections
 *
 * Note that the resulting segment coordinate arrays will be unsorted, but the data contained should be interpreted as stride-2
 * arrays for which the odd-indexed elements represent endpoints and even-indexed elements represent coordinates of startpoints
 *
 * @param xs The x coordinates of triangle vertices (stride-3)
 * @param ys The y coordinates of triangle vertices (stride-3)
 * @param zs The z coordinates of triangle vertices (stride-3)
 * @param layersInTri
 * @param startIndexInSegments
 * @param seg_x The boundary array x coordinate
 * @param seg_y The boundary array y coordinate
 * @param seg_l The boundary array layer coordinate
 * @param lH The distance between layers
 * @param n The total number of triangles to be analyzed
 * @return
 */
__global__ void findContourSegmentsForEachTriangle(float* xs, float* ys, float* zs, int* layersInTri, int* startIndexInSegments, float* seg_x, float* seg_y, int* seg_l, const float lH, const int n) {
    // TODO: !IMPORTANT: A couple ways this could be optimized:
    // - Processing tris with points lying on the layerplane in a totally separate kernel so that "normal" tris don't have to go through all these contingencies every time for boundary conditions
    // - Sorting tris by number of layers spanned. This could allow us to process "short" tris on the same block as each other, having roughly the same loop size allows for smaller idle time
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

                updateContourSegmentsIfIntersects(x0, x1, y0, y1, z0, z1, layer, zp, si, seg_x, seg_y, seg_l);
                updateContourSegmentsIfIntersects(x1, x2, y1, y2, z1, z2, layer, zp, si, seg_x, seg_y, seg_l);
                updateContourSegmentsIfIntersects(x2, x0, y2, y0, z2, z0, layer, zp, si, seg_x, seg_y, seg_l);

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
                            updateContourSegmentsIfIntersects(x1, x2, y1, y2, z1, z2, layer, zp, si, seg_x, seg_y, seg_l);
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
                            updateContourSegmentsIfIntersects(x2, x0, y2, y0, z2, z0, layer, zp, si, seg_x, seg_y, seg_l);
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
                            updateContourSegmentsIfIntersects(x0, x2, y0, y2, z0, z2, layer, zp, si, seg_x, seg_y, seg_l);
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

__global__ void calculateScanPointsPerSegment(float* iscy_p, int* numScanPointsPerSegment, const float scanHeight, int totalIntersections)
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

// TODO: this could be done inline, y1-y0 and x1-x0 should be reused between iterations
__device__ float getXIntersection2D(float atY, float x0, float x1, float y0, float y1)
{
    printf("\nx0: %f\n\
            y0: %f\n\
            x1: %f\n\
            y1: %f\n\n", x0, y0, x1, y1);
    float t = (atY - y0) / (y1 - y0);
    return t * (x1 - x0) + x0;
}

__global__ void calculateScanIntersections(float* iscx, float* iscy, int* iscl, int* numScanPointsPerSegment, float* scanIntersections_x, float* scanIntersections_y, int* scanIntersections_l, int* intersectionSegmentsIndexStart, const int totalIntersections, const float scanHeight)
{
    // TODO: Needs outer loop, gtid != scanLineNumber

    // TODO: I believe we could just not concern ourselves with perfectly horizontal lines, but this is not yet confirmed
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gtid < totalIntersections) {
        int numScanPoints = numScanPointsPerSegment[gtid];
        int startIndex =  intersectionSegmentsIndexStart[gtid];
        float x0 = iscx[gtid*2];
        float x1 = iscx[gtid*2+1];
        float y_0 = iscy[gtid*2]; // temp before we know max/min
        float y_1 = iscy[gtid*2+1];
        float y0 = min(y_0, y_1);
        float y1 = max(y_0, y_1);
        //printf("\ngtid: %d\n\
        //        numScanPoints: %d\n\
        //        x0: %f\n\
        //        y0: %f\n\
        //        x1: %f\n\
        //        y1: %f\n\n", gtid, numScanPoints, x0, y0, x1, y1);

        int scanIntersectionIndex; float curY; float resx;
        //printf("numscanPoints:%d\n", numScanPoints);
        //printf("totalIntersections:%d\n", totalIntersections);
        for(int i=0; i < numScanPoints; i++) {
            scanIntersectionIndex = i+startIndex;
            curY = y0 + i * scanHeight;
            //printf("scanIntersectionIndex: %d\n", scanIntersectionIndex);
            resx = x0 + (curY - y0) / (y1-y0) * (x1 - x0);
            scanIntersections_x[scanIntersectionIndex] = resx;
            scanIntersections_y[scanIntersectionIndex] = curY;
            scanIntersections_l[scanIntersectionIndex] = iscl[gtid*2];
        }
    }
}

typedef thrust::tuple<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<int>::iterator> IteratorTupleXYL;
typedef thrust::tuple<float, float, int> TupleXYL;
typedef thrust::zip_iterator<IteratorTupleXYL> ZipXYL;

struct xyl_predicate
{
    __device__ bool operator()(TupleXYL xyl0, TupleXYL xyl1){
        if(thrust::get<2>(xyl0) == thrust::get<2>(xyl1)) {
            if(thrust::get<1>(xyl0) == thrust::get<1>(xyl1)) {
                return thrust::get<0>(xyl0) <= thrust::get<0>(xyl1);
            }

                // If same layer, different scan line, sort higher scan first
            else {
                return thrust::get<1>(xyl0) < thrust::get<1>(xyl1);
            }
        }
            // If not on the same layer, sort lower layer first
        else {
            return thrust::get<2>(xyl0) < thrust::get<2>(xyl1);
        }
    }
};

struct xyl_eq
{
    __device__ bool operator()(TupleXYL xyl0, TupleXYL xyl1){
        return (thrust::get<2>(xyl0) == thrust::get<2>(xyl1)) && (thrust::get<1>(xyl0) == thrust::get<1>(xyl1)) && (thrust::get<0>(xyl0) == thrust::get<0>(xyl1));
    }
};

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

    thrust::device_vector<int> intersectionSegmentsIndexStart(n, 0);
    thrust::inclusive_scan(layersInTris.begin(), layersInTris.end(), intersectionSegmentsIndexStart.begin());

    // Might want to do this asyncronously? Could be a costly operation
    int totalIntersections = intersectionSegmentsIndexStart[intersectionSegmentsIndexStart.size()-1];

    // Intersection segment coordinate arrays
    thrust::device_vector<float> iscx(totalIntersections*2, 0);
    thrust::device_vector<float> iscy(totalIntersections*2, 0);
    thrust::device_vector<int> iscl(totalIntersections*2, 0);
    float* iscx_p = thrust::raw_pointer_cast( &iscx[0] );
    float* iscy_p = thrust::raw_pointer_cast( &iscy[0] );
    // TODO: Should probably convert to int everywhere
    int* iscl_p = thrust::raw_pointer_cast( &iscl[0] );
    float* x_p = thrust::raw_pointer_cast( &dx[0] );
    float* y_p = thrust::raw_pointer_cast( &dy[0] );
    int* intersectionSegmentsIndexStart_p = thrust::raw_pointer_cast( &intersectionSegmentsIndexStart[0]);

    findContourSegmentsForEachTriangle<<<2, 8>>>(x_p, y_p, z_p, layersInTris_p, intersectionSegmentsIndexStart_p, iscx_p, iscy_p, iscl_p, layerHeight, n);

    printf("totalIntersections: %d\n", totalIntersections);
    print_vector("layersInTris", layersInTris);

    print_vector("iscx", iscx);
    print_vector("iscy", iscy);
    print_vector("iscl", iscl);


    // Copy over our layer vector so that we can use it as a key vector twice for stable_sort_by_key
    thrust::device_vector<int> iscl2(totalIntersections*2, 0);
    thrust::copy(iscl.begin(), iscl.end(), iscl2.begin());

    // TODO: A custom implementation of sort by key here would allow us to avoid copying
    // A zip iterator might also do the trick
    // TODO: It may not even be necessary to sort here
    thrust::stable_sort_by_key(thrust::device, iscl.begin(), iscl.end(), iscx.begin());
    thrust::stable_sort_by_key(thrust::device, iscl2.begin(), iscl2.end(), iscy.begin());

    // TODO: we might put a pruning algorithm here to get rid of duplicate segments
    // This would spare us in instances where we have non-manifold shapes in 3D space
    // and allow us to say that more than 2 sequential duplicates indicates a non-manifold junction and we should
    // continue through

    print_vector("iscx", iscx);
    print_vector("iscy", iscy);
    print_vector("iscl", iscl);

    // PART 2: Linear scan of layer planes
    // We begin with our list of segment points representing a set of unordered boundaries

    thrust::device_vector<int> scanPointsPerSegment(totalIntersections, 0);
    int* scanPointsPerSegment_p = thrust::raw_pointer_cast( &scanPointsPerSegment[0] );
    calculateScanPointsPerSegment<<<2, 32>>>(iscy_p, scanPointsPerSegment_p, scanHeight, totalIntersections); // Gotta get the total intersections
    print_vector("scanPointsPerSegment", scanPointsPerSegment);

    // TODO: It may make sense here to sort boundary lines by scanPointsPerSegment in order to make sure that the smallest
    // lines (lines with least iterations required) get done on the same warp. This would also allow for the smallest
    // warps to pick up larger warps at the end quicker once the external loop is added

    thrust::device_vector<int> scanIntersectionIndexStart(totalIntersections, 0);
    thrust::inclusive_scan(scanPointsPerSegment.begin(), scanPointsPerSegment.end(), scanIntersectionIndexStart.begin());
    int totalScanPoints = scanIntersectionIndexStart[scanIntersectionIndexStart.size()-1];

    print_vector("scanIntersectionIndexStart", scanIntersectionIndexStart);


    thrust::device_vector<float> SIx(totalScanPoints, 0); // Scan intersections
    thrust::device_vector<float> SIy(totalScanPoints, 0); // Scan intersections
    thrust::device_vector<int> SIl(totalScanPoints, 0); // Scan intersections
    float* SIx_p = thrust::raw_pointer_cast( &SIx[0] );
    float* SIy_p = thrust::raw_pointer_cast( &SIy[0] );
    int* SIl_p = thrust::raw_pointer_cast( &SIl[0] );
    int* scanIntersectionIndexStart_p = thrust::raw_pointer_cast( &scanIntersectionIndexStart[0] );


    calculateScanIntersections<<<2, 32>>>(iscx_p, iscy_p, iscl_p, scanPointsPerSegment_p, SIx_p, SIy_p, SIl_p, scanIntersectionIndexStart_p, totalIntersections, scanHeight);

    print_vector("SIx", SIx);
    print_vector("SIy", SIy);
    print_vector("SIl", SIl);

    // Sort such that points on the same layer are grouped, within that, points on the same scanline are grouped, and within THAT points are grouped by x coordinate
    thrust::sort(thrust::make_zip_iterator(thrust::make_tuple(SIx.begin(), SIy.begin(), SIl.begin())),
                 thrust::make_zip_iterator(thrust::make_tuple(SIx.end(), SIy.end(), SIl.end())),
                 xyl_predicate());

    print_vector("SIx", SIx);
    print_vector("SIy", SIy);
    print_vector("SIl", SIl);

    // Remove all duplicate points
    xyl_eq predicate;

    ZipXYL newEnd = thrust::unique(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(SIx.begin(), SIy.begin(), SIl.begin())),
                                   thrust::make_zip_iterator(thrust::make_tuple(SIx.end(), SIy.end(), SIl.end())),
                                   predicate);

    IteratorTupleXYL endTuple = newEnd.get_iterator_tuple();
    SIx.erase( thrust::get<0>( endTuple ), SIx.end() );
    SIy.erase( thrust::get<1>( endTuple ), SIy.end() );
    SIl.erase( thrust::get<2>( endTuple ), SIl.end() );

    print_vector("SIx", SIx);
    print_vector("SIy", SIy);
    print_vector("SIl", SIl);

    // Finish timing
    cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
    cudaEventSynchronize(stopEvent_inc);
    cudaEventElapsedTime(&time, startEvent_inc, stopEvent_inc);

    // END MY CODE

    // TODO: Unroll loops *using pragmas* to keep code clean.
    // TODO: Use optimized operations in cuda code where possible (i.e. `__add`

    // TODO: Free all resources *as soon as we can*
    // We might hit memory bounds on larger models

    //free resources
    //free(in); free(out); free(cuda_out);
    return 0;
}