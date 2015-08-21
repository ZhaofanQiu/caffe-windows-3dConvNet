#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_3d_layers.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void StretchForward(const int nthreads, const Dtype* bottom_data,
		const int channels, const int offset, Dtype* top_data) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int o = index % offset;
			int c = index / offset;
			top_data[o * channels + c] =
				bottom_data[c * offset + o];
		}
	}

	template <typename Dtype>
	Dtype Stretch3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
		Dtype* top_data = (*top)[0]->mutable_gpu_data();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		int count = count_ / num_;
		int offset = length_ * height_ * width_;
		for (int n = 0; n < num_; ++n)
		{
			StretchForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, bottom_data + bottom[0]->offset(n), channels_, offset ,
				top_data + (*top)[0]->offset(n * offset));
		}
		return Dtype(0.);
	}

	template <typename Dtype>
	__global__ void StretchBackward(const int nthreads, const Dtype* top_diff,
		const int channels, const int offset, Dtype* bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int o = index % offset;
			int c = index / offset;
			bottom_diff[c * offset + o] =
				top_diff[o * channels + c];
		}
	}

	template <typename Dtype>
	void Stretch3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
		int count = count_ / num_;
		int offset = length_ * width_ * height_;
		
		if (!propagate_down) { return; }
		for (int n = 0; n < num_; ++n)
		{
			StretchBackward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
				count, top_diff + top[0]->offset(n * offset), channels_, offset,
				bottom_diff + (*bottom)[0]->offset(n));
		}
	}

	INSTANTIATE_CLASS(Stretch3DLayer);
}  // namespace caffe
