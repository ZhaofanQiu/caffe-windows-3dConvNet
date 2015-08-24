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


#include <vector>
#include "caffe/layer.hpp"
#include "caffe/video_3d_layers.hpp"
#include "caffe/util/vol2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype Deconvolution3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (*top)[0]->mutable_gpu_data();
	Dtype* col_data = col_buffer_.mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();

	for (int n = 0; n < num_; ++n) {
		// First, inner-product
		for (int g = 0; g < filter_group_; ++g) {
			caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / filter_group_,
				conv_out_spatial_dim_, channels_ / filter_group_,
				(Dtype)1., weight + weight_offset_ * g, bottom_data + bottom[0]->offset(n) + output_offset_ * g,
				(Dtype)0., col_data + col_offset_ * g);

		}
		//Second, col2vol
		col2vol_gpu(col_data, num_output_, length_out_, height_out_, width_out_, kernel_size_, kernel_depth_, pad_,
			temporal_pad_, stride_, temporal_stride_, top_data + (*top)[0]->offset(n));

		//Third, add bias
		if (bias_term_) {
			caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
				length_out_ * height_out_ * width_out_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
				(Dtype)1., top_data + (*top)[0]->offset(n));
		}
	}
  return Dtype(0.);
}

template <typename Dtype>
void Deconvolution3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	const Dtype* bottom_data = (*bottom)[0]->gpu_data();
	Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
	Dtype* col_data = col_buffer_.mutable_gpu_data();
	// bias gradient if necessary
	Dtype* bias_diff = NULL;

	if (bias_term_) {
		bias_diff = this->blobs_[1]->mutable_gpu_diff();
		CUDA_CHECK(cudaMemset(bias_diff, 0,
			sizeof(Dtype) * this->blobs_[1]->count()));
		for (int n = 0; n < num_; ++n) {
			caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, length_out_ * height_out_ * width_out_,
				1., top_diff + top[0]->offset(n),
				reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), 1.,
				bias_diff);
		}
	}

  CUDA_CHECK(cudaMemset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));

  for (int n = 0; n < num_; ++n) {
	  // since we saved memory in the forward pass by not storing all col data,
	  // we will need to recompute them.
	  vol2col_gpu(top_diff + top[0]->offset(n), num_output_, length_out_, height_out_,
		  width_out_, kernel_size_, kernel_depth_, pad_, temporal_pad_, stride_,
		  temporal_stride_, col_data);

	  // gradient w.r.t. weight. Note that we will accumulate diffs.
	  for (int g = 0; g<filter_group_; ++g){
		  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_ / filter_group_,
			  kernel_dim_ / filter_group_, conv_out_spatial_dim_,
			  (Dtype)1., bottom_data + (*bottom)[0]->offset(n) + g * output_offset_,
			  col_data + col_offset_ * g, (Dtype)1.,
			  weight_diff + g * weight_offset_);
	  }

	  // gradient w.r.t. bottom data, if necessary
	  if (propagate_down) {

		  // accumulate the other filter groups -> col_diff
		  for (int g = 0; g<filter_group_; ++g){
			  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ / filter_group_,
				  conv_out_spatial_dim_, kernel_dim_ / filter_group_,
				  (Dtype)1., weight + g * weight_offset_,
				  col_data + g * col_offset_,
				  (Dtype)0., bottom_diff + (*bottom)[0]->offset(n) + g * output_offset_);
		  }
	  }
  }
}


INSTANTIATE_CLASS(Deconvolution3DLayer);

}  // namespace caffe
