/*
 *
 *  Copyright (c) 2015, Facebook, Inc. All rights reserved.
 *
 *  Licensed under the Creative Commons Attribution-NonCommercial 3.0
 *  License (the "License"). You may obtain a copy of the License at
 *  https://creativecommons.org/licenses/by-nc/3.0/.
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  License for the specific language governing permissions and limitations
 *  under the License.
 *
 *
 */

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/video_3d_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {
	
  // Copy (one line per thread) from one array to another, with arbitrary 
 // strides in the last two dimensions. 
 template <typename Dtype> 
 __global__ void copy_kernel(const int n, const int height, const int width, 
     const int src_outer_stride, const int src_inner_stride, 
     const int dest_outer_stride, const int dest_inner_stride, 
     const Dtype* src, Dtype* dest) { 
   CUDA_KERNEL_LOOP(index, n) { 
     int src_start = index / height * src_outer_stride 
                   + index % height * src_inner_stride; 
     int dest_start = index / height * dest_outer_stride 
                    + index % height * dest_inner_stride; 
     for (int i = 0; i < width; ++i) { 
       dest[dest_start + i] = src[src_start + i]; 
     } 
   } 
 } 

 template <typename Dtype> 
 __global__ void copy_kernel_3d(const int n, const int length, const int height, const int width,
	 const int src_stride1, const int src_stride2, const int src_stride3,
	 const int dest_stride1, const int dest_stride2, const int dest_stride3,
	 const Dtype* src, Dtype* dest) {
	 CUDA_KERNEL_LOOP(index, n) {
		 int src_start = index % height * src_stride3 + index / height % length * src_stride2 + index / (height * length) * src_stride1;
		 int dest_start = index % height * dest_stride3 + index / height % length * dest_stride2 + index / (height * length) * dest_stride1;

		 for (int i = 0; i < width; ++i) {
			 dest[dest_start + i] = src[src_start + i];
		 }
	 }
 }

template <typename Dtype>
Dtype Crop3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
   const Dtype* bottom_data = bottom[0]->gpu_data(); 
   Dtype* top_data = (*top)[0]->mutable_gpu_data(); 
   const int lines = (*top)[0]->count() / (*top)[0]->width();

   // NOLINT_NEXT_LINE(whitespace/operators) 
   copy_kernel_3d << <CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS >> >(
	   lines, (*top)[0]->length(), (*top)[0]->height(), (*top)[0]->width(),
       bottom[0]->length() * bottom[0]->height() * bottom[0]->width(), bottom[0]->height() * bottom[0]->width(), bottom[0]->width(), 
	   (*top)[0]->length() * (*top)[0]->height() * (*top)[0]->width(), (*top)[0]->height() * (*top)[0]->width(), (*top)[0]->width(),
       bottom_data + bottom[0]->offset(0, 0, crop_l_, crop_h_, crop_w_), top_data); 

  return Dtype(0.);
}

template <typename Dtype>
void Crop3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff(); 
   Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff(); 
   const int lines = top[0]->count() / top[0]->width(); 
 
   if (propagate_down) { 
	   caffe_gpu_set((*bottom)[0]->count(), static_cast<Dtype>(0), bottom_diff);
	   caffe_gpu_set((*bottom)[1]->count(), static_cast<Dtype>(0), (*bottom)[1]->mutable_gpu_diff());
     // NOLINT_NEXT_LINE(whitespace/operators) 
	   copy_kernel_3d << <CAFFE_GET_BLOCKS(lines), CAFFE_CUDA_NUM_THREADS >> >(
         lines, top[0]->length(), top[0]->height(), top[0]->width(), 
         top[0]->length() * top[0]->height() * top[0]->width(), top[0]->height() * top[0]->width(), top[0]->width(), 
		 (*bottom)[0]->length() * (*bottom)[0]->height() * (*bottom)[0]->width(), (*bottom)[0]->height() * (*bottom)[0]->width(), (*bottom)[0]->width(),
		 top_diff, bottom_diff + (*bottom)[0]->offset(0, 0, crop_l_, crop_h_, crop_w_));
   } 

}

INSTANTIATE_CLASS(Crop3DLayer);

}  // namespace caffe
