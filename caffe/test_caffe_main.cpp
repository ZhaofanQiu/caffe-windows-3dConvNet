#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/video_3d_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe{

	template <typename Dtype>
	class Deconvolution3DLayerTest {
	public:
		Deconvolution3DLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}
		~Deconvolution3DLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(2, 2, 3, 4, 4);
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
			//propagate to bottom
			propagate_down = true;
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		bool propagate_down;

	public:
		void StartTest(int device_id){
			LayerParameter layer_param;
			ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
			convolution_param->set_kernel_size(4);
			convolution_param->set_kernel_depth(1);
			convolution_param->set_stride(2);
			convolution_param->set_num_output(2);
			convolution_param->mutable_weight_filler()->set_type("gaussian");
			convolution_param->mutable_bias_filler()->set_type("gaussian");

			shared_ptr<Layer<Dtype>> layer(new Deconvolution3DLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->num(), 2);
			EXPECT_EQ(this->blob_top_->channels(), 2);
			EXPECT_EQ(this->blob_top_->length(), 3);
			EXPECT_EQ(this->blob_top_->height(), 10);
			EXPECT_EQ(this->blob_top_->width(), 10);

			if (device_id < 0)
			{
				Caffe::set_mode(Caffe::CPU);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
			else
			{
				Caffe::set_mode(Caffe::GPU);
				Caffe::SetDevice(device_id);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
		}
	};

	template <typename Dtype>
	class Crop3DLayerTest {
	public:
		Crop3DLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_bottom2_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}

		~Crop3DLayerTest(){ delete blob_bottom_; delete blob_bottom2_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(2, 2, 3, 4, 4);
			blob_bottom2_->Reshape(2, 2, 2, 3, 3);
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			filler.Fill(this->blob_bottom2_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom2_);
			blob_top_vec_.push_back(blob_top_);
			//propagate to bottom
			propagate_down = true;
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_bottom2_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		bool propagate_down;

	public:
		void StartTest(int device_id){
			LayerParameter layer_param;
			shared_ptr<Layer<Dtype>> layer(new Crop3DLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->num(), 2);
			EXPECT_EQ(this->blob_top_->channels(), 2);
			EXPECT_EQ(this->blob_top_->length(), 2);
			EXPECT_EQ(this->blob_top_->height(), 3);
			EXPECT_EQ(this->blob_top_->width(), 3);

			if (device_id < 0)
			{
				Caffe::set_mode(Caffe::CPU);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
			else
			{
				Caffe::set_mode(Caffe::GPU);
				Caffe::SetDevice(device_id);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
		}
	};

	template <typename Dtype>
	class EltwiseLayerTest {
	public:
		EltwiseLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_bottom2_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}

		~EltwiseLayerTest(){ delete blob_bottom_; delete blob_bottom2_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(2, 2, 3, 4, 4);
			blob_bottom2_->Reshape(2, 2, 3, 4, 4);
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			filler.Fill(this->blob_bottom2_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom2_);
			blob_top_vec_.push_back(blob_top_);
			//propagate to bottom
			propagate_down = true;
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_bottom2_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		bool propagate_down;

	public:
		void StartTest(int device_id){
			LayerParameter layer_param;
			EltwiseParameter* eltwise_param = layer_param.mutable_eltwise_param();
			eltwise_param->set_operation(EltwiseParameter_EltwiseOp_SUM);

			shared_ptr<Layer<Dtype>> layer(new Crop3DLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->num(), 2);
			EXPECT_EQ(this->blob_top_->channels(), 2);
			EXPECT_EQ(this->blob_top_->length(), 3);
			EXPECT_EQ(this->blob_top_->height(), 4);
			EXPECT_EQ(this->blob_top_->width(), 4);

			if (device_id < 0)
			{
				Caffe::set_mode(Caffe::CPU);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
			else
			{
				Caffe::set_mode(Caffe::GPU);
				Caffe::SetDevice(device_id);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
		}
	};

	template <typename Dtype>
	class Stretch3DLayerTest {
	public:
		Stretch3DLayerTest() :blob_bottom_(new Blob<Dtype>()),
			blob_top_(new Blob<Dtype>()){
			this->SetUp();
		}

		~Stretch3DLayerTest(){ delete blob_bottom_; delete blob_top_; }

	protected:
		void SetUp(){
			blob_bottom_->Reshape(2, 2, 3, 4, 4);
			//fill the values
			FillerParameter filler_param;
			filler_param.set_value(1.);
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_bottom_vec_.push_back(blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
			//propagate to bottom
			propagate_down = true;
		}
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
		bool propagate_down;

	public:
		void StartTest(int device_id){
			LayerParameter layer_param;

			shared_ptr<Layer<Dtype>> layer(new Stretch3DLayer<Dtype>(layer_param));
			layer->SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
			EXPECT_EQ(this->blob_top_->num(), 96);
			EXPECT_EQ(this->blob_top_->channels(), 2);
			EXPECT_EQ(this->blob_top_->length(), 1);
			EXPECT_EQ(this->blob_top_->height(), 1);
			EXPECT_EQ(this->blob_top_->width(), 1);

			if (device_id < 0)
			{
				Caffe::set_mode(Caffe::CPU);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
			else
			{
				Caffe::set_mode(Caffe::GPU);
				Caffe::SetDevice(device_id);
				GradientChecker<Dtype> checker(1e-2, 1e-3);
				checker.CheckGradientExhaustive(layer.get(), &this->blob_bottom_vec_, &this->blob_top_vec_);
			}
		}
	};
}

int main(int argc, char** argv){
	FLAGS_logtostderr = 1;
	caffe::Deconvolution3DLayerTest<float> test1;
	test1.StartTest(15);
	caffe::Crop3DLayerTest<float> test2;
	test2.StartTest(15);
	caffe::EltwiseLayerTest<float> test3;
	test3.StartTest(15);
	caffe::Stretch3DLayerTest<float> test4;
	test4.StartTest(15);
	return 0;
}
