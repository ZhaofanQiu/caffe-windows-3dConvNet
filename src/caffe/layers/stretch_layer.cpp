#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/video_3d_layers.hpp"

namespace caffe {

	template <typename Dtype>
	void Stretch3DLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {

		CHECK_EQ(top->size(), 1) <<
			"StretchLayer takes a single blob as output.";
		CHECK_EQ(bottom.size(), 1) <<
			"StretchLayer takes a single blob as input.";

		// Initialize with the first blob.
		count_ = bottom[0]->count();
		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		length_ = bottom[0]->length();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();

		(*top)[0]->Reshape(count_ / channels_, channels_, 1, 1, 1);
	}

	template <typename Dtype>
	Dtype Stretch3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = (*top)[0]->mutable_cpu_data();

		int offset = length_ * height_ * width_;
		for (int n = 0; n < num_; ++n)
		{
			for (int c = 0; c < channels_; ++c)
			{
				for (int o = 0; o < offset; ++o)
				{
					*(top_data + (*top)[0]->offset(n * offset + o, c)) =
						*(bottom_data + bottom[0]->offset(n, c) + o);
				}
			}
		}
		return Dtype(0.);
	}

	template <typename Dtype>
	void Stretch3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
		if (!propagate_down) { return; }
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		int offset = length_ * height_ * width_;
		for (int n = 0; n < num_; ++n)
		{
			for (int c = 0; c < channels_; ++c)
			{
				for (int o = 0; o < offset; ++o)
				{
					*(bottom_diff + (*bottom)[0]->offset(n, c) + o) =
						*(top_diff + top[0]->offset(n * offset + o, c));
				}
			}
		}
	}

	INSTANTIATE_CLASS(Stretch3DLayer);

}  // namespace caffe