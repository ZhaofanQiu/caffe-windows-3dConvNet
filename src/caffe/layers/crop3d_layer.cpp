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

template <typename Dtype>
void Crop3DLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Crop3DLayer takes two blobs as input.";
	CHECK_EQ(top->size(), 1) << "Crop3DLayer takes a single blob as output.";

	//fake a easy crop ____by zhaofan
	CHECK_GE(bottom[0]->length(), bottom[1]->length());
	CHECK_GE(bottom[0]->height(), bottom[1]->height());
	CHECK_GE(bottom[0]->width(), bottom[1]->width());
	

   crop_h_ = round((bottom[0]->height() - bottom[1]->height()) / 2); 
   crop_w_ = round((bottom[0]->width() - bottom[1]->width()) / 2);
   crop_l_ = round((bottom[0]->length() - bottom[1]->length()) / 2);

   (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->length(), bottom[1]->height(),
	          bottom[1]->width());
}

template <typename Dtype>
Dtype Crop3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
   const Dtype* bottom_data = bottom[0]->cpu_data(); 
   Dtype* top_data = (*top)[0]->mutable_cpu_data(); 
   for (int n = 0; n < (*top)[0]->num(); ++n) {
	   for (int c = 0; c < (*top)[0]->channels(); ++c) {
		   for (int l = 0; l < (*top)[0]->length(); ++l)
		   {
			   for (int h = 0; h < (*top)[0]->height(); ++h) {
				   caffe_copy((*top)[0]->width(),
					   bottom_data + bottom[0]->offset(n, c, crop_l_ + l, crop_h_ + h, crop_w_),
					   top_data + (*top)[0]->offset(n, c, l, h));
			   }
		   }
     } 
   } 
  return Dtype(0.);
}

template <typename Dtype>
void Crop3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	  if (!propagate_down) {
	    return;
	  }
   const Dtype* top_diff = top[0]->cpu_diff(); 
   Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff(); 
   if (propagate_down) { 
	   caffe_set((*bottom)[0]->count(), static_cast<Dtype>(0), bottom_diff);
     for (int n = 0; n < top[0]->num(); ++n) { 
       for (int c = 0; c < top[0]->channels(); ++c) { 
		   for (int l = 0; l < top[0]->length(); ++l)
		   {
			   for (int h = 0; h < top[0]->height(); ++h) {
				   caffe_copy(top[0]->width(),
					   top_diff + top[0]->offset(n, c, l, h),
					   bottom_diff + (*bottom)[0]->offset(n, c, crop_l_ + l, crop_h_ + h, crop_w_));
			   }
         } 
       } 
     } 
   } 

}

INSTANTIATE_CLASS(Crop3DLayer);

}  // namespace caffe
