// -------------- gen txt-------------
#include <boost/timer.hpp>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/viz.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int num_pic = 5000;

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Please enter your kitti data path" << endl
         << "For example:           ./slam_dataset/kitti/07/" << endl;
    return 1;
  }
  string filepath = argv[1];
  string outfilename = "rgb.txt";
  string timefilename = "times.txt";
  ofstream txtgenerator;
  ifstream timereader;
  txtgenerator.open(filepath + outfilename, ios::out | ios::trunc);
  timereader.open(filepath + timefilename);

  for (int i = 0; i < num_pic; i++) {
    string tmp_time;
    timereader >> tmp_time;
    txtgenerator << tmp_time;

    txtgenerator << " image_0/";
    txtgenerator.fill('0'); //设置填充字符
    txtgenerator.width(6);  //设置域宽
    txtgenerator << i << ".png" << endl;
  }

  txtgenerator.close();
  return 0;
}
