def sortContourForLayer(self, inp, outp, blockSums, N):

    # Create a N array of neighbors, initialize to -1
    # Each thread takes a start point
    # While we haven't found a neighbor
        # If we've already found a neighbor for this point, stop the loop. Note this is volatile and must be checked every time
        # Check the next point
        # If this point doesn't have a neighbor yet
            # Then check if its adjacent.
            # If it is adjacent
                # Write the results to this thread's position on the neighbor array
                # Also write to the neighbors position

# What do we have?
    # Constant function to check the number of intersections for a given triangle
    # We could also

# Intersection pre-allocation algorithm
    # Check each triangle


# Intersection algorithm
    # Take our list of triangles
    # Generate an array of heights (custom kernel) on each triangle
    # Generate intersectionSegmentsIndexStart as an additive inclusive reduction over heights array
    # (custom kernel) For each triangle, calculate intersection with each layer, place in intersections array, place layer in layers array
    # Sort the intersections array by layer
    # Count the instances of each layer in the layers array, store in `intersectionsPerLayer` vector

