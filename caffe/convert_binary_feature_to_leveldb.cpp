//added by Qing Li, 2014-12-13
// This program converts a set of features to a leveldb by storing them as Datum
// proto buffers.
// Usage:
//    convert_featureset ROOTFOLDER/ LISTFILE DB_NAME [0/1]
// where ROOTFOLDER is the root folder that holds all the features, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.fisher 7
//   ....
// if the last argument is 1, a random shuffle will be carried out before we
// process the file lines.

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using std::string;

#define CHANNELS 1 
#define LENGTH 1 
#define HEIGHT 240 
#define WIDTH 320

float buf[CHANNELS * LENGTH * HEIGHT * WIDTH];

bool ReadNoSparseFeatureToVolumeDatum(const string& filename, const int label, VolumeDatum* datum)
{
	LOG(INFO) << filename << '\t' << label;

	FILE* fp;
	fp = fopen(filename.c_str(), "rb");
	if (fp == NULL)
	{
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}
	datum->set_channels(CHANNELS);
	datum->set_length(LENGTH);
	datum->set_height(HEIGHT);
	datum->set_width(WIDTH);
	datum->set_label(label);
	datum->clear_data();
	datum->clear_float_data();

	CHECK_EQ(CHANNELS * LENGTH * HEIGHT * WIDTH, fread(buf, sizeof(float), CHANNELS * LENGTH * HEIGHT * WIDTH, fp)) << "file not full read";

	fclose(fp);

	auto datum_ptr = datum->mutable_float_data();
	for (int dim = 0; dim < CHANNELS * LENGTH * HEIGHT * WIDTH; dim++)
	{
		datum_ptr->Add(buf[dim]);
	}
	return true;
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 4 || argc > 5) {
		printf("Convert a set of features to the leveldb format used\n"
			"as input for Caffe.\n"
			"Usage:\n"
			"    convert_featureset ROOTFOLDER/ LISTFILE DB_NAME"
			" RANDOM_SHUFFLE_DATA[0 or 1] \n");
		return 1;
	}
	std::ifstream infile(argv[2]);
	std::vector<std::pair<string, int> > lines;
	string filename;
	int label;
	while (infile >> filename >> label) {
		lines.push_back(std::make_pair(filename, label));
	}
	if (argc == 5 && argv[4][0] == '1') {
		// randomly shuffle data
		LOG(INFO) << "Shuffling data";
		std::random_shuffle(lines.begin(), lines.end());
	}
	LOG(INFO) << "A total of " << lines.size() << " features.";

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(INFO) << "Opening leveldb " << argv[3];
	leveldb::Status status = leveldb::DB::Open(
		options, argv[3], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

	string root_folder(argv[1]);
	VolumeDatum datum;
	int count = 0;
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;
	/*std::stringstream str2int;
	str2int<<argv[4];
	int channel;
	str2int>>channel;*/

	for (int line_id = 0; line_id < lines.size(); ++line_id) {
		if (!ReadNoSparseFeatureToVolumeDatum(root_folder + lines[line_id].first,
			lines[line_id].second, &datum))
		{
			continue;
		}
		if (!data_size_initialized) {
			data_size = datum.channels() * datum.length() * datum.height() * datum.width();
			data_size_initialized = true;
		}
		else {
			//const string& data = datum.float_data();
			CHECK_EQ(datum.float_data().size(), data_size) << "Incorrect data field size "
				<< datum.float_data().size();
		}
		// sequential
		_snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
			lines[line_id].first.c_str());
		string value;
		// get the value
		datum.SerializeToString(&value);
		batch->Put(string(key_cstr), value);
		if (++count % 100 == 0) {
			db->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR) << "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	// write the last batch
	if (count % 100 != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR) << "Processed " << count << " files.";
	}

	delete batch;
	delete db;
	return 0;
}
