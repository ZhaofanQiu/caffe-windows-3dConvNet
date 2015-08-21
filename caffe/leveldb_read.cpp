// Copyright 2014 BVLC and contributors.

#include <glog/logging.h>
#include <gflags\gflags.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <algorithm>
#include <string>
#include <iostream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::BlobProto;
using std::max;
using std::cout;

DEFINE_string(out_data_file, "", "path of file to store data in db.");

int main(int argc, char** argv) 
{
	::google::InitGoogleLogging(argv[0]);
	FLAGS_alsologtostderr = 1;

	if (argc < 3) {
		LOG(ERROR) << "Usage: leveldb_read input_leveldb img_info_output_file [data_output_file]";
		return 1;
	}
	::gflags::ParseCommandLineFlags(&argc, &argv, true);
	const std::string out_data_file = FLAGS_out_data_file;
	FILE* output_data = fopen(out_data_file.c_str(),"w");
	FILE* output_imginfo = fopen(argv[2], "w");
	CHECK(output_imginfo != NULL) << "open file: " << argv[3] << " failed.";

	leveldb::DB* db;
	leveldb::Options options;
	options.create_if_missing = false;

	LOG(INFO) << "Opening leveldb " << argv[1];
	leveldb::Status status = leveldb::DB::Open(options, argv[1], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[1];

	leveldb::ReadOptions read_options;
	read_options.fill_cache = false;
	leveldb::Iterator* it = db->NewIterator(read_options);

	caffe::VolumeDatum datum;
	int count = 0;

	LOG(INFO) << "Starting Iteration";
	for (it->SeekToFirst(); it->Valid(); it->Next()) 
	{
		// just a dummy operation
		datum.ParseFromString(it->value().ToString());
		const int label = datum.label();
		const std::string& imgname = it->key().ToString();
		//write imgname and label into file
		fprintf(output_imginfo, "%s\t%d\n", imgname.c_str(), label);
//		fflush(output_imginfo);
		if (out_data_file.size()){
			const int data_size = datum.data().size();
			const int float_data_size = datum.float_data_size();
			int size_in_datum = data_size > float_data_size ? data_size : float_data_size;
			CHECK(size_in_datum > 0) << "Empty data in image " << imgname;
			if (float_data_size > data_size){
				for (int i = 0; i < size_in_datum; ++i){
					fprintf(output_data, "%f\t", static_cast<float>(datum.float_data(i)));
				}
			}
			else{
				for (int i = 0; i < size_in_datum; ++i){
					fprintf(output_data, "%f\t", static_cast<float>((datum.data()[i])));
				}
			}
			fprintf(output_data, "\n");
		}
		++count;
		if (count % 10 == 0){
			LOG(INFO) <<"No."<< count <<": "<< imgname << "\t" << label;
		}
		if (count % 1000 == 0)
		{
			LOG(ERROR) << "Have read: " << count << " files.";
		}
	}
	if (count % 1000 != 0) 
	{
		LOG(ERROR) << "Processed " << count << " files.";
	}
	if (out_data_file.size()){
		fclose(output_data);
	}
	fclose(output_imginfo);
	delete db;
	return 0;
}
