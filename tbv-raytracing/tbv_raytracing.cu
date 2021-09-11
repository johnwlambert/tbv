// Copyright (c) 2021
// Argo AI, LLC, All Rights Reserved.
// 
// Notice: All information contained herein is, and remains the property
// of Argo AI. The intellectual and technical concepts contained herein
// are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
// Patents, patents in process, and are protected by trade secret or
// copyright law. This work is licensed under a CC BY-NC-SA 4.0 
// International License.
// 
// Originating Authors: John Lambert


#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>
#include <limits> // for std::numeric limits
#include <stdio.h>


// #include <pybind11/eigen.h>


/** add 
* @{
*/
__device__ float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 operator+(const float3& a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__device__ float3 operator+(const float a, const float3& b)
{
  return make_float3(a + b.x, a + b.y, a + b.z);
}
__device__ void operator+=(float3& a, const float3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
__device__ float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 operator-(const float3& a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
__device__ float3 operator-(const float a, const float3& b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
__device__ void operator-=(float3& a, const float3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
__device__ float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ float3 operator*(const float3& a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ void operator*=(float3& a, const float3& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
__device__ void operator*=(float3& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
__device__ float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__device__ float3 operator/(const float3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
__device__ float3 operator/(const float s, const float3& a)
{
  return make_float3( s/a.x, s/a.y, s/a.z );
}
__device__ void operator/=(float3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** dot product from optixu/optixu_math_namespace.h */
__device__ float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product from optixu/optixu_math_namespace.h */
__device__ float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/** normalize from optixu/optixu_math_namespace.h */
__device__ float3 normalize(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}



__device__ float3 ray_triangle_intersect_kernel(
  float3 origin,
  float3 ray_dir,
  float3 v0,
  float3 v1,
  float3 v2)
{
  float NO_HIT_VALUE = -99999;

  float3 v0v1 = v1 - v0;
  float3 v0v2 = v2 - v0;
  float3 pvec = cross(ray_dir,v0v2);
  double det = dot(v0v1,pvec);

  float kEpsilon = 1e-10;
  if (det < kEpsilon)
  {
    return make_float3(NO_HIT_VALUE,NO_HIT_VALUE,NO_HIT_VALUE);
  }

  double invDet = 1 / det;
  float3 tvec = origin - v0;
  
  double u = dot(tvec,pvec) * invDet;
  if ((u < 0) || (u > 1))
  {
    return make_float3(NO_HIT_VALUE,NO_HIT_VALUE,NO_HIT_VALUE);
  }

  float3 qvec = cross(tvec,v0v1);
  float v = dot(ray_dir,qvec) * invDet;
  if ((v < 0) || (u + v > 1))
  {
    return make_float3(NO_HIT_VALUE,NO_HIT_VALUE,NO_HIT_VALUE);
  }

  float t = dot(v0v2,qvec) * invDet;

  float3 P = origin + t * ray_dir;
  return P;
}




/*
moller_trombore algorithm
Will have num_rays global thread indices

Each thread handles 1 ray

Args
  ray_dirs: float array with 3 * num_rays elements
  hits: float array with 3 * num_rays * num_triangles elements
  triangles: float array with 9 * num_triangles elements

  int size_bytes = num_rays*N_COORDS_PER_RAY*sizeof(double);
  ray_dirs_gpu, 

  size_bytes = num_triangles*N_VERTS_PER_TRI*N_COORDS_PER_TRI_VERTEX*sizeof(double);
  triangles_gpu

  size_bytes = num_triangles*num_rays*3*sizeof(double);
  hits_gpu

*/
__global__ void ray_mesh_intersect_kernel(
  const float3 origin,
  const double* ray_dirs,
  const double* triangles,
  double* hits,
  const int num_rays,
  const int num_triangles)
{
  unsigned int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx < (num_rays) )
  {
    float NO_HIT_VALUE = -99999;
    float kEpsilon = 1e-10;

    unsigned int ray_offs = 3 * ray_idx; // n'th element to take from the array
    unsigned int hit_offs = ray_offs;
    float3 ray_dir = make_float3(ray_dirs[ray_offs], ray_dirs[ray_offs+1], ray_dirs[ray_offs+2]);

    for (int tri_idx=0; tri_idx < num_triangles; tri_idx++)
    {
      unsigned int tri_offs = 9 * tri_idx; // n'th element to take from the array
  
      float3 v0 = make_float3(triangles[tri_offs  ], triangles[tri_offs+1], triangles[tri_offs+2]);
      float3 v1 = make_float3(triangles[tri_offs+3], triangles[tri_offs+4], triangles[tri_offs+5]);
      float3 v2 = make_float3(triangles[tri_offs+6], triangles[tri_offs+7], triangles[tri_offs+8]);

      float3 hit = ray_triangle_intersect_kernel(origin, ray_dir, v0, v1, v2);
      if (
           ( fabs(hit.x - NO_HIT_VALUE) > kEpsilon) 
        && ( fabs(hit.y - NO_HIT_VALUE) > kEpsilon) 
        && ( fabs(hit.z - NO_HIT_VALUE) > kEpsilon)
        )
      {
        hits[hit_offs] = hit.x;
        hits[hit_offs + 1] = hit.y;
        hits[hit_offs + 2] = hit.z;
        return;
      }
    }
    // this ray never hit any triangle
    hits[hit_offs] = NO_HIT_VALUE;
    hits[hit_offs + 1] = NO_HIT_VALUE;
    hits[hit_offs + 2] = NO_HIT_VALUE;

  }
}



// each ray tries to hit a triangle, then exits 
// otherwise, send out all triangle-ray pairs
void run_intersection_kernel(
  const float3 & origin,
  const double* ray_dirs, // on gpu
  const double* triangles, // on gpu
  double* hits, // on gpu
  int num_rays,
  int num_triangles)
{
  int n_thread = 512; // also try 256
  dim3 dimBlock(n_thread, 1, 1); // threads per block
  int num_blocks_reqd = ceil( float(num_rays) / float(dimBlock.x) );
  dim3 dimGrid(num_blocks_reqd); // number of blocks

  //std::cout << "Num rays: " << num_rays << std::endl;
  //std::cout << "Num triangles: " << num_triangles << std::endl;
  //std::cout << "n thread: " << n_thread << std::endl;
  //std::cout << "Num blocks required:" << num_blocks_reqd << std::endl;

  ray_mesh_intersect_kernel<<<dimGrid, dimBlock>>>
    (
      origin,
      ray_dirs,
      triangles,
      hits,
      num_rays,
      num_triangles
    );

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}


/*
TODO: 8000 images over 1000 logs is 8M calls of this function
Make one class, and prevent 8M mallocs

Also try 1 thread per triangle-ray pair, and compare runtime.
*/

//std::tuple<Eigen::VectorXb, Eigen::MatrixXd>
void intersect_rays_with_tri_mesh(
  pybind11::array_t<double> triangles,
  pybind11::array_t<double> origin,
  pybind11::array_t<double> ray_dirs,
  pybind11::array_t<double> hits)
{
  pybind11::buffer_info tri_info = triangles.request();
  pybind11::buffer_info ray_info = ray_dirs.request();


  if (tri_info.ndim != 2) {
    std::stringstream strstr;
    strstr << "triangles.ndim != 2" << std::endl;
    strstr << "triangles.ndim: " << tri_info.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  if (tri_info.shape[1] != 9) {
    std::stringstream strstr;
    strstr << "triangles.shape[1] != 9" << std::endl;
    strstr << "triangles.shape[1]: " << tri_info.shape[1] << std::endl;
    throw std::runtime_error(strstr.str());
  }

  if (ray_info.ndim != 2) {
    std::stringstream strstr;
    strstr << "ray_dirs.ndim != 2" << std::endl;
    strstr << "ray_dirs.ndim: " << ray_info.ndim << std::endl;
    throw std::runtime_error(strstr.str());
  }

  if (ray_info.shape[1] != 3) {
    std::stringstream strstr;
    strstr << "ray_dirs.shape[1] != 9" << std::endl;
    strstr << "ray_dirs.shape[1]: " << ray_info.shape[1] << std::endl;
    throw std::runtime_error(strstr.str());
  }

  int num_rays = ray_info.shape[0];
  int num_triangles = tri_info.shape[0];

  double ox = origin.data()[0];
  double oy = origin.data()[1];
  double oz = origin.data()[2];
  float3 origin_vec = make_float3(ox, oy, oz);
  
  double* ray_dirs_gpu;
  double* triangles_gpu;
  double* hits_gpu;

  int N_COORDS_PER_TRI_VERTEX = 3;
  int N_VERTS_PER_TRI = 3;
  int N_COORDS_PER_RAY = 3;
  int N_COORDS_PER_HIT = 3;
  
  int size_bytes = num_rays*N_COORDS_PER_RAY*sizeof(double);
  cudaError_t error = cudaMalloc(&ray_dirs_gpu, size_bytes);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
  double* ray_dirs_ptr = reinterpret_cast<double*>(ray_info.ptr);
  error = cudaMemcpy(ray_dirs_gpu, ray_dirs_ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  size_bytes = num_triangles*N_VERTS_PER_TRI*N_COORDS_PER_TRI_VERTEX*sizeof(double);
  error = cudaMalloc(&triangles_gpu, size_bytes);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
  double* triangles_ptr = reinterpret_cast<double*>(tri_info.ptr);
  error = cudaMemcpy(triangles_gpu, triangles_ptr, size_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  size_bytes = num_rays*N_COORDS_PER_HIT*sizeof(double);
  //std::cout << "Malloc " << size_bytes << " for hits_gpu" << std::endl;
  error = cudaMalloc(&hits_gpu, size_bytes);
  if (error != cudaSuccess) {
    std::cout << "Malloc failed!" << std::endl;
    throw std::runtime_error(cudaGetErrorString(error));
  }

  run_intersection_kernel(
    origin_vec,
    ray_dirs_gpu,
    triangles_gpu,
    hits_gpu,
    num_rays,
    num_triangles
  );

  //std::cout << "kernel ran successfully!" << std::endl;
  //std::cout << "Ray_dirs size: " << ray_info.size << std::endl;

  /* No pointer is passed, so NumPy will allocate the buffer */
  // auto inter_points_arr = pybind11::array_t<double>(ray_info.size);

  pybind11::buffer_info hits_info = hits.request();
  double *hits_cpu_ptr = static_cast<double *>(hits_info.ptr);

  error = cudaMemcpy(hits_cpu_ptr, hits_gpu, size_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  // for (int i=0; i<ray_info.size; i++)
  // {
  //   std::cout << "Hit " << i << ": " << hits_cpu_ptr[i]  << std::endl;
  // }

  // Eigen::VectorXb inter_exists_arr = Eigen::VectorXb(num_rays);
  // inter_exists_arr.setConstant(false); 

  // Eigen::MatrixXd inter_points_arr = Eigen::MatrixXd(num_rays,3);
  // inter_points_arr.setZero();

  error = cudaFree(ray_dirs_gpu);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(triangles_gpu);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  error = cudaFree(hits_gpu);
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }

  // return std::make_tuple(inter_exists_arr, inter_points_arr);
  //return inter_points_arr;
};



PYBIND11_MODULE(tbv_raytracing, m)
{
  m.def("intersect_rays_with_tri_mesh", &intersect_rays_with_tri_mesh);
}
