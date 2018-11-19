#ifndef DATA_SET_H
#define DATA_SET_H

#include <vector>
#include <string>
#include <initializer_list>

// Meta data used to normalize the data set. Useful to
// go back and forth between normalized data.
class DataSetMetaData {
friend class DataSet;
private:
  float mean_km;
  float std_km;
  float mean_age;
  float std_age;
  float min_price;
  float max_price;
};

enum class Fuel {
    DIESEL,
    GAZOLINE
};

class DataSet {
public:
  // Construct a data set from the given csv file path.
  DataSet(std::string dir, std::string file_name) {
    ReadCSVFile(dir, file_name);
  }

  // getters
  std::vector<float>& x() { return x_; }
  std::vector<float>& y() { return y_; }

  // read the given csv file and complete x_ and y_
  void ReadCSVFile(std::string dir, std::string file_name);

  // convert one csv line to a vector of float
  std::vector<float> ReadCSVLine(std::string line);

  // normalize a human input using the data set metadata
  std::initializer_list<float> input(float km, Fuel fuel, float age);
  std::vector<float> input_vector(float km, Fuel fuel, float age);

  // convert a price outputted by the DNN to a human price
  float output(float price);
private:
  DataSetMetaData data_set_metadata;
  std::vector<float> x_;
  std::vector<float> y_;
};

#endif // DATA_SET_H
