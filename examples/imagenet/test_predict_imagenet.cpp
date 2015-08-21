#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/smart_ptr/shared_ptr.hpp"

using namespace caffe;
using namespace std;
using namespace cv;
using namespace boost;

int main(int argc, char** argv)
{
    // Set GPU
    Caffe::set_mode(Caffe::GPU);
    int device_id = 0;
    Caffe::SetDevice(device_id);
    LOG(INFO) << "Using GPU";

    // Set to TEST Phase
    Caffe::set_phase(Caffe::TEST);

    // Load net
    Net<float> net("../models/bvlc_reference_caffenet/deploy_memory_data.prototxt");

    // Load pre-trained net (binary proto)
    net.CopyTrainedLayersFrom("../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel");

    // Load image
    //string imgName = "../examples/images/fish-bike.jpg";
    string imgName = "../examples/images/cat.jpg";
    Mat image = imread(imgName);
    imshow("image", image);
    waitKey(1);

    // Set vector for image
    vector<cv::Mat> imageVector;
    imageVector.push_back(image);

    // Set vector for label
    vector<int> labelVector;
    labelVector.push_back(0);//push_back 0 for initialize purpose

    // Net initialization
    float loss = 0.0;
    boost::shared_ptr<MemoryDataLayer<float> > memory_data_layer;
    memory_data_layer = boost::static_pointer_cast<MemoryDataLayer<float>>(net.layer_by_name("data"));

    //memory_data_layer->AddMatVector(imageVector, labelVector);
    Datum datum;
    ReadImageToDatum(imgName, 1, 227, 227, &datum);

    vector<Datum> datums;
    for (int i = 0; i < 1; i++)
        datums.push_back(datum);

    memory_data_layer->AddDatumVector(datums);
    const vector<Blob<float>*>& results = net.ForwardPrefilled(&loss);

    // Two free output layer in deploy_memory_data.prototxt: 'label' and 'output'.
    // The second output should be 282 which is corresponding to cat.
    for (int i = 0; i < results.size(); i++) {
        for (int j = 0; j < results[i]->channels(); j++) {
            LOG(INFO) << results[i]->mutable_cpu_data()[j];
        }
    }

    // Load data layer out for display
    // To-Do: Not working right now
    /*const boost::shared_ptr<Blob<float> > feature_blob = net.blob_by_name("data");
    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    const float* feature_blob_data = feature_blob->mutable_cpu_data() + feature_blob->offset(0);

    Mat feaImg0(Size(28, 28), CV_8UC3);
    for (int chn = 0; chn < 3; chn++) {
        for (int row = 0; row < 28; row++) {
            for (int col = 0; col < 28; col++) {
                int d = chn * 28 * 28 + row * 28 + col;
                uchar val = uchar(feature_blob_data[d] * 255);
                Vec3b(feaImg0.at<uchar>(row, col)).val[chn] = uchar(feature_blob_data[d] * 255);
            }
        }
    }
    imshow("data", feaImg0);
    waitKey(1);*/

    system("pause");
    return 0;
}