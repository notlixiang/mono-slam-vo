/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H

#include "myslam/common_include.h"
#include "myslam/map.h"

#include <opencv2/features2d/features2d.hpp>

namespace myslam 
{
class VisualOdometry
{
public:
    typedef shared_ptr<VisualOdometry> Ptr;
    enum VOState {
        STARTING=-2,
        INITIALIZING=-1,
        OK=0,
        LOST
    };
    
    VOState     state_;     // current VO status
    Map::Ptr    map_;       // map with all frames and map points
    
    Frame::Ptr  ref_;       // reference key-frame
    Frame::Ptr  curr_;      // current frame
    
    cv::Ptr<cv::ORB> orb_;  // orb detector and computer
    vector<cv::KeyPoint>    keypoints_curr_;    // keypoints in current frame
    Mat                     descriptors_curr_;  // descriptor in current frame
    
    vector<cv::KeyPoint>    keypoints_ref_;
    Mat                     descriptors_ref_;


    cv::FlannBasedMatcher   matcher_flann_;     // flann matcher
    vector<MapPoint::Ptr>   match_3dpts_;       // matched 3d points
    vector<cv::DMatch> good_matches_2d2d_;
    vector<cv::DMatch> good_matches_3d2d_;
    vector< Point3d > tiangulation_points_buff_;

    vector<int>             match_2dkp_index_;  // matched 2d pixels (index of kp_curr)

    vector<int>             match_2dkp_index_ref_;
    vector<int>             match_2dkp_index_curr_;
vector<bool>  tiangulation_points_good_;
vector<bool>  bool_inlier_3d2d_;


    SE3 T_c_w_estimated_;    // the estimated pose of current frame
    int num_inliers_;        // number of inlier features in icp

    int num_inliers_2d2d_;

    int num_lost_;           // number of lost times
    
    // parameters
    int num_of_features_;   // number of features
    double scale_factor_;   // scale in image pyramid
    int level_pyramid_;     // number of pyramid levels
    float match_ratio_;     // ratio for selecting  good matches
    int max_num_lost_;      // max number of continuous lost times
    int min_inliers_;       // minimum inliers
    int min_inliers_2d2d_;
    int min_good_matches_;
    int triangulation_point_angle_enough_num_;
    int min_triangulation_point_angle_enough_num_;
    double key_frame_min_rot;   // minimal rotation of two key-frames
    double key_frame_min_trans; // minimal translation of two key-frames
    double map_point_erase_ratio_; // remove map point ratio
    double num_inliers_2d2d_ratio_;
    double mean_view_error_ref_triangulation_;
    double mean_view_error_cur_triangulation_;
    double mean_view_angle_triangulation_;
    double min_view_angle_triangulation_;
    double min_view_angle_triangulation_init_;
    double max_mean_view_error_triangulation_;
    double max_matching_distance_;
    int max_init_frame_refresh_num_;
public: // functions 
    VisualOdometry();
    ~VisualOdometry();
    
    bool addFrame( Frame::Ptr frame );      // add a new frame
    int init_cnt_frame=0;
    int max_map_points_;

protected:  
    // inner operation
    void extractKeyPoints();
    void computeDescriptors();

    void extractKeyPointsRef();
    void computeDescriptorsRef();

    void featureMatching();
    void featureMatching2d2d();//used for initialization
    void poseEstimation2d2d ();


    void poseEstimationPnP();
    void optimizeMap();
    
    void addKeyFrame();
    //    void addMapPoints();
    bool checkEstimatedPose();
    bool checkEstimatedInitPose();
    bool checkKeyFrame();
    
    double getViewAngle( Frame::Ptr frame, MapPoint::Ptr point );
    
    void triangulation ();
    int Cnt0(Mat src);
    int Cnt255(Mat src);
    double distanceEuclidean2d(double x1,double y1,double x2,double y2);
    double distanceEuclidean3d(double x1,double y1,double z1,double x2,double y2,double z2);
    double length3d(double x,double y,double z);
    double angleRadFromAcos(double l1,double l2,double d);
    void addMapPointsTriangulation();
void globalBundleAdjustment();
void removeMultiPoints();
void *ptrGlobalBundleAdjustment();
};
}

#endif // VISUALODOMETRY_H
