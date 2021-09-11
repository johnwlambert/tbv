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


#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <vector>

namespace Eigen {
    typedef Matrix<bool, Dynamic, 1> VectorXb;
}

/*
See for reference:
https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates

t is the distance along the ray from the origin

Args:
	origin: shape (3,)
	ray_dir: ray direction shape (3,)
	v0: triangle vertex 0, ordered counter-clockwise (CCW)
	v1: triangle vertex 1
	v2: triangle vertex 2

Returns:
	boolean whether intersection is valid
	P: intersection point if exists, otherwise None
*/
std::tuple<bool,Eigen::Vector3d> ray_triangle_intersect_moller_trombore(
	const Eigen::Vector3d & origin,
	const Eigen::Vector3d & ray_dir,
	const Eigen::Vector3d & v0,
	const Eigen::Vector3d & v1,
	const Eigen::Vector3d & v2)
{
	Eigen::Vector3d v0v1 = v1 - v0;
	Eigen::Vector3d v0v2 = v2 - v0;
	Eigen::Vector3d pvec = ray_dir.cross(v0v2);
	float det = v0v1.dot(pvec);

	Eigen::Vector3d null_vec(0.0,0.0,0.0);

	// CULLING 
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	float kEpsilon = 1e-10;
	if (det < kEpsilon) return std::make_tuple(false, null_vec);

	float invDet = 1 / det;

	Eigen::Vector3d tvec = origin - v0;
	float u = tvec.dot(pvec) * invDet;
	if ((u < 0) || (u > 1)) return std::make_tuple(false, null_vec);

	Eigen::Vector3d qvec = tvec.cross(v0v1);
	float v = ray_dir.dot(qvec) * invDet;
	if ((v < 0) || (u + v > 1)) return std::make_tuple(false, null_vec);

	float t = v0v2.dot(qvec) * invDet;

	Eigen::Vector3d P = origin + t * ray_dir;
	return std::make_tuple(true, P);
}


/* 
In the egovehicle frame, intersect triangles up to some distance away 

triangles is N x 9 matrix, with 9 cols for 3 vertices
	first 3 cols as v0, then next 3 cols as v1, then v2 coords

Args:
	triangles: N x 9 
	origin: length-3 array
	ray_dirs: M x 3

Returns:
	inter_exists_arr: M boolean vals
	inter_points: M 3d points
*/
std::tuple<Eigen::VectorXb, Eigen::MatrixXd> intersect_rays_with_tri_mesh(
	const Eigen::MatrixXd & triangles,
	const Eigen::Vector3d & origin,
	const Eigen::MatrixXd & ray_dirs)
{
	int n_triangles = triangles.rows();
	int n_rays = ray_dirs.rows();

	Eigen::VectorXb inter_exists_arr = Eigen::VectorXb(n_rays);
	inter_exists_arr.setConstant(false); 

	Eigen::MatrixXd inter_points_arr = Eigen::MatrixXd(n_rays,3);
	inter_points_arr.setZero();

	for (int j=0; j < n_rays; j++)
	{
		for (int i=0; i < n_triangles; i++)
		{
			std::tuple<bool,Eigen::Vector3d> hit_data = ray_triangle_intersect_moller_trombore(
				origin,
				ray_dirs.row(j),
				triangles.row(i).segment<3>(0), // v0
				triangles.row(i).segment<3>(3), // v1
				triangles.row(i).segment<3>(6) // v2
			);
			inter_exists_arr(j) = std::get<0>(hit_data); // boolean whether we hit
			inter_points_arr.row(j) = std::get<1>(hit_data);
			if ( inter_exists_arr(j) ) break;
		}
	}
	return std::make_tuple(inter_exists_arr, inter_points_arr);
}


PYBIND11_MODULE(tbv_raytracing, m) {
    m.doc() = "pybind11 raytracing plugin"; // optional module docstring

    m.def(
    	"ray_triangle_intersect_moller_trombore",
    	&ray_triangle_intersect_moller_trombore,
    	"Moller-Trombore ray-triangle intersection"
    );

	m.def(
		"intersect_rays_with_tri_mesh",
		&intersect_rays_with_tri_mesh,
		"Ray-mesh intersection w/ Moller-Trombore subroutine"
	);
}

