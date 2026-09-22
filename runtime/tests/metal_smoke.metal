#include <metal_stdlib>
using namespace metal;

kernel void add_one(device uint *values [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    values[index] += 1;
}
