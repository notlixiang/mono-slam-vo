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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <algorithm>
#include <boost/timer.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"
#include <algorithm>

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( STARTING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_ ( new cv::flann::LshIndexParams ( 5,10,2 ) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    min_inliers_2d2d_=Config::get<int> ( "min_inliers_2d2d" );
    num_inliers_2d2d_ratio_ = Config::get<double> ( "num_inliers_2d2d_ratio" );
    min_good_matches_= Config::get<double> ( "min_good_matches" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ( "map_point_erase_ratio" );
    min_view_angle_triangulation_= Config::get<double> ( "min_view_angle_triangulation" );
    min_view_angle_triangulation_init_= Config::get<double> ( "min_view_angle_triangulation_init" );
    max_init_frame_refresh_num_= Config::get<double> ( "max_init_frame_refresh_num" );
    max_mean_view_error_triangulation_= Config::get<double> ( "max_mean_view_error_triangulation" );
    min_triangulation_point_angle_enough_num_= Config::get<double> ( "min_triangulation_point_angle_enough_num" );
    max_matching_distance_= Config::get<double> ( "max_matching_distance" );
    orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{

    switch ( state_ )
    {
    case STARTING:
    {
        cout<<"state_ : "<<state_<<endl;
        curr_ = ref_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        extractKeyPointsRef();//prepare for initialization
        computeDescriptorsRef();
        //        if(init_cnt_frame<5)// initialization result judgement
        //        {
        //            init_cnt_frame++;
        //            break;
        //        }
        addKeyFrame();      // the first frame is a key-frame
        state_ = INITIALIZING;
        break;
    }
    case INITIALIZING:
    {
        cout<<"state_ : "<<state_<<endl;
        curr_ = frame;
        // extract features from first frame and add them into map
        extractKeyPoints();
        computeDescriptors();
        featureMatching2d2d();
        poseEstimation2d2d();
        triangulation();
        bool res=checkEstimatedInitPose();
        if(res==true)// initialization result judgement
        {

            addMapPointsTriangulation();
            addKeyFrame();      // the success init frame is a key-frame
            globalBundleAdjustment();
            ref_ = frame;
            cout<<"--------------------------------------------------------------------------"
               <<"Initialization finished !"
              <<"--------------------------------------------------------------------------"<<endl;

            //            pthread_t tid ;
            //            ret = pthread_create(&tid, NULL,ptrGlobalBundleAdjustment,NULL);
            //            pthread_join(tid,NULL);
            //            for ( auto& allpoints: map_->map_points_ )
            //            {
            //                MapPoint::Ptr& p = allpoints.second;
            //                // check if p in curr frame image
            //                cout<<p->pos_.transpose()<<endl;
            //            }

            state_ = OK;
            //            waitKey();
            break;
        }
        cout<<"init_cnt_frame :"<<init_cnt_frame<<endl;
        if(init_cnt_frame<max_init_frame_refresh_num_)// initialization result judgement
        {
            init_cnt_frame++;
            break;
        }
        ref_ = frame;
        init_cnt_frame=0;
        break;
    }
    case OK:
    {
        cout<<"state_ : "<<state_<<endl;
        curr_ = frame;
        //        curr_=ref_;
        curr_->T_c_w_ = ref_->T_c_w_;
        extractKeyPoints();
        computeDescriptors();
        featureMatching();
        poseEstimationPnP();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_w_estimated_;
            optimizeMap();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                //                waitKey();
                //                optimizeMap();
                extractKeyPointsRef();//prepare for initialization
                computeDescriptorsRef();
                featureMatching2d2d();
                triangulation();
                //                waitKey();
                addMapPointsTriangulation();
                //                                removeMultiPoints();
                addKeyFrame();
                globalBundleAdjustment();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;

                //                num_lost_=0;
                //                state_ = STARTING;
            }
            break;
            return false;
        }
        cout<<"map_->map_points_.size() : "<<map_->map_points_.size()<<endl;
        cout<<"map_->keyframes_.size() : "<<map_->keyframes_.size()<<endl;
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;

        break;
        return false;
    }
    }
    return true;
}

void VisualOdometry::extractKeyPoints()
{

    boost::timer timer;
int col_0_4=0;
int col_1_4=int(curr_->color_.cols*1/4);
int col_2_4=int(curr_->color_.cols*2/4);
int col_3_4=int(curr_->color_.cols*3/4);
int col_4_4=int(curr_->color_.cols*4/4);

    Mat part1= curr_->color_.colRange(col_0_4,col_1_4).clone();
    Mat part2= curr_->color_.colRange(col_1_4,col_2_4).clone();
    Mat part3= curr_->color_.colRange(col_2_4,col_3_4).clone();
    Mat part4= curr_->color_.colRange(col_3_4,col_4_4).clone();

    vector<cv::KeyPoint>    keypoints_part1_;    // keypoints in current frame
    vector<cv::KeyPoint>    keypoints_part2_;
    vector<cv::KeyPoint>    keypoints_part3_;
    vector<cv::KeyPoint>    keypoints_part4_;

    orb_->detect ( part1, keypoints_part1_ );
    orb_->detect ( part2, keypoints_part2_ );
    orb_->detect ( part3, keypoints_part3_ );
    orb_->detect ( part4, keypoints_part4_ );


    keypoints_curr_.clear();

    for(auto point:keypoints_part2_)
    {
        point.pt.x+=col_1_4;
        keypoints_curr_.push_back(point);
    }
    for(auto point:keypoints_part3_)
    {
        point.pt.x+=col_2_4;
        keypoints_curr_.push_back(point);
    }
    for(auto point:keypoints_part4_)
    {
        point.pt.x+=col_3_4;
        keypoints_curr_.push_back(point);
    }

    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
    cout<<"keypoints_curr_.size() : "<<keypoints_curr_.size()<<endl;
}

void VisualOdometry::computeDescriptors()
{
    descriptors_curr_=Mat();
    boost::timer timer;
    orb_->compute ( curr_->color_, keypoints_curr_, descriptors_curr_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
}
///------------new
void VisualOdometry::extractKeyPointsRef()
{
    boost::timer timer;
    int col_0_4=0;
    int col_1_4=int(ref_->color_.cols*1/4);
    int col_2_4=int(ref_->color_.cols*2/4);
    int col_3_4=int(ref_->color_.cols*3/4);
    int col_4_4=int(ref_->color_.cols*4/4);

        Mat part1= ref_->color_.colRange(col_0_4,col_1_4).clone();
        Mat part2= ref_->color_.colRange(col_1_4,col_2_4).clone();
        Mat part3= ref_->color_.colRange(col_2_4,col_3_4).clone();
        Mat part4= ref_->color_.colRange(col_3_4,col_4_4).clone();

        vector<cv::KeyPoint>    keypoints_part1_;    // keypoints in current frame
        vector<cv::KeyPoint>    keypoints_part2_;
        vector<cv::KeyPoint>    keypoints_part3_;
        vector<cv::KeyPoint>    keypoints_part4_;

        orb_->detect ( part1, keypoints_part1_ );
        orb_->detect ( part2, keypoints_part2_ );
        orb_->detect ( part3, keypoints_part3_ );
        orb_->detect ( part4, keypoints_part4_ );


        keypoints_ref_.clear();

        for(auto point:keypoints_part2_)
        {
            point.pt.x+=col_1_4;
            keypoints_ref_.push_back(point);
        }
        for(auto point:keypoints_part3_)
        {
            point.pt.x+=col_2_4;
            keypoints_ref_.push_back(point);
        }
        for(auto point:keypoints_part4_)
        {
            point.pt.x+=col_3_4;
            keypoints_ref_.push_back(point);
        }
//    orb_->detect ( ref_->color_, keypoints_ref_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::computeDescriptorsRef()
{
    boost::timer timer;
    orb_->compute ( ref_->color_, keypoints_ref_, descriptors_ref_ );
    cout<<"descriptor computation cost time: "<<timer.elapsed() <<endl;
}
///------------!-new

void VisualOdometry::featureMatching2d2d()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    // select the candidates in map
    //    Mat desp_map;
    //    vector<MapPoint::Ptr> candidate;
    //    for ( auto& allpoints: map_->map_points_ )
    //    {
    //        MapPoint::Ptr& p = allpoints.second;
    //        // check if p in curr frame image
    //        if ( curr_->isInFrame(p->pos_) )
    //        {
    //            // add to candidate
    //            p->visible_times_++;
    //            candidate.push_back( p );
    //            desp_map.push_back( p->descriptor_ );
    //        }
    //    }

    matcher_flann_.match ( descriptors_ref_, descriptors_curr_, matches );


    // select the best matches
    float min_dis = std::min_element (
                matches.begin(), matches.end(),
                [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;



    if(state_ == INITIALIZING)
    {
        match_2dkp_index_ref_.clear();
        match_2dkp_index_curr_.clear();
        good_matches_2d2d_.clear();
        for ( cv::DMatch& m : matches )
        {
            if ( m.distance < max<float> ( min_dis*match_ratio_, max_matching_distance_ ) )
            {
                good_matches_2d2d_.push_back(m);
                match_2dkp_index_ref_.push_back( m.queryIdx );
                match_2dkp_index_curr_.push_back( m.trainIdx );
            }
        }

        if(good_matches_2d2d_.size()<100)
        {
            match_2dkp_index_ref_.clear();
            match_2dkp_index_curr_.clear();
            good_matches_2d2d_.clear();
            for ( cv::DMatch& m : matches )
            {
                if ( m.distance < max<float> ( min_dis*match_ratio_, max_matching_distance_*1.5 ) )
                {
                    good_matches_2d2d_.push_back(m);
                    match_2dkp_index_ref_.push_back( m.queryIdx );
                    match_2dkp_index_curr_.push_back( m.trainIdx );
                }
            }
        }

    }
    else
    {
        match_2dkp_index_ref_.clear();
        match_2dkp_index_curr_.clear();
        good_matches_2d2d_.clear();
        for ( cv::DMatch& m : matches )
        {
            if ( m.distance < max<float> ( min_dis*match_ratio_, max_matching_distance_ ) )
            {
                good_matches_2d2d_.push_back(m);
                match_2dkp_index_ref_.push_back( m.queryIdx );
                match_2dkp_index_curr_.push_back( m.trainIdx );
            }
        }

        if(good_matches_2d2d_.size()<100)
        {
            match_2dkp_index_ref_.clear();
            match_2dkp_index_curr_.clear();
            good_matches_2d2d_.clear();
            for ( cv::DMatch& m : matches )
            {
                if ( m.distance < max<float> ( min_dis*match_ratio_, max_matching_distance_*1.5 ) )
                {
                    good_matches_2d2d_.push_back(m);
                    match_2dkp_index_ref_.push_back( m.queryIdx );
                    match_2dkp_index_curr_.push_back( m.trainIdx );
                }
            }
        }
    }

//    Mat ShowMatches;
//    drawMatches(ref_->color_,keypoints_ref_, curr_->color_,keypoints_curr_,good_matches_2d2d_,ShowMatches);
//    imshow("matches", ShowMatches);

    cout<<"good matches 2d2d: "<<match_2dkp_index_ref_.size() <<endl;
    cout<<"match cost time 2d2d: "<<timer.elapsed() <<endl;
}


void VisualOdometry::featureMatching()
{
    boost::timer timer;
    vector<cv::DMatch> matches;
    // select the candidates in map
    Mat desp_map;
    vector<MapPoint::Ptr> candidate;
    cout<<"Add candidate "<<endl;
    cout<<"map_->map_points_.size() : "<<map_->map_points_.size()<<endl;
    for ( auto& allpoints: map_->map_points_ )
    {
        //        cout<<"desp_map ready : "<<candidate.size()<<endl;
        MapPoint::Ptr& p = allpoints.second;
        // check if p in curr frame image
        if ( ref_->isInFrame(p->pos_) )
        {
            // add to candidate

            p->visible_times_++;
            candidate.push_back( p );
            desp_map.push_back( p->descriptor_ );
        }
    }
    cout<<"desp_map ready : "<<candidate.size()<<endl;
    if (candidate.size()==0){return;}
    matcher_flann_.match ( desp_map, descriptors_curr_, matches );
    // select the best matches
    float min_dis = std::min_element (
                matches.begin(), matches.end(),
                [] ( const cv::DMatch& m1, const cv::DMatch& m2 )
    {
        return m1.distance < m2.distance;
    } )->distance;

    good_matches_3d2d_.clear();
    match_3dpts_.clear();
    match_2dkp_index_.clear();
    for ( cv::DMatch& m : matches )
    {
        if ( m.distance < max<float> ( min_dis*match_ratio_, max_matching_distance_) )
        {
            good_matches_3d2d_.push_back(m);
            match_3dpts_.push_back( candidate[m.queryIdx] );
            match_2dkp_index_.push_back( m.trainIdx );
        }
    }

    cout<<"good matches: "<<match_3dpts_.size() <<endl;
    cout<<"match cost time: "<<timer.elapsed() <<endl;
}

void VisualOdometry::poseEstimation2d2d ()
{
    // 相机内参
    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
              );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<cv::Point2f> points_ref;
    vector<cv::Point2f> points_curr;

    for ( int i = 0; i < ( int ) match_2dkp_index_ref_.size(); i++ )
    {
        points_ref.push_back ( keypoints_ref_[match_2dkp_index_ref_[i]].pt );
        points_curr.push_back ( keypoints_curr_[match_2dkp_index_curr_[i]].pt );
    }

    //-- 计算基础矩阵
    //    cv::Mat fundamental_matrix;
    //    fundamental_matrix = findFundamentalMat ( points_ref, points_curr, CV_FM_8POINT );
    //    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    cv::Point2f principal_point ( ref_->camera_->cx_, ref_->camera_->cy_ );	//相机光心,
    double focal_length = (ref_->camera_->fx_+ref_->camera_->fy_)/2.0;			//相机焦距,
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points_ref, points_curr, focal_length, principal_point );
    //    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //    //-- 计算单应矩阵
    //    Mat homography_matrix;
    //    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    //    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    Mat R,t,inliers;
    recoverPose ( essential_matrix, points_ref, points_curr, R, t, focal_length, principal_point,inliers );

    num_inliers_2d2d_=Cnt255(inliers);

    cout<<"Init num_inliers_2d2d_: "<<num_inliers_2d2d_<<endl;
    //cout<<inliers<<endl;

    Mat rvec;
    Rodrigues(R,rvec);
    T_c_w_estimated_ = SE3 (
                SO3 (rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                );
    curr_->T_c_w_ =T_c_w_estimated_;
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t.t()<<endl;

}


int VisualOdometry::Cnt255(Mat src)
{

    int counter = 0;
    //迭代器访问像素点
    Mat_<uchar>::iterator it = src.begin<uchar>();
    Mat_<uchar>::iterator itend = src.end<uchar>();
    for (; it!=itend; ++it)
    {
        if((*it)==255) counter+=1;//二值化后，像素点是0或者255
    }
    return counter;
}

int VisualOdometry::Cnt0(Mat src)
{

    int counter = 0;
    //迭代器访问像素点
    Mat_<uchar>::iterator it = src.begin<uchar>();
    Mat_<uchar>::iterator itend = src.end<uchar>();
    for (; it!=itend; ++it)
    {
        if((*it)==0) counter+=1;//二值化后，像素点是0或者255
    }
    return counter;
}


void VisualOdometry::poseEstimationPnP()
{
    // construct the 3d 2d observations
    vector<cv::Point3f> pts3d;
    vector<cv::Point2f> pts2d;

    for ( int index:match_2dkp_index_ )
    {
        pts2d.push_back ( keypoints_curr_[index].pt );
    }
    for ( MapPoint::Ptr pt:match_3dpts_ )
    {
        pts3d.push_back( pt->getPositionCV() );
    }

    cout<<"pts2d.size() : "<<pts2d.size()<<endl;
    cout<<"pts3d.size() : "<<pts3d.size()<<endl;

    if(pts3d.size()<4)
    {
        return;
    }

    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
              );
    Mat rvec, tvec, inliers;
    cv::solvePnPRansac ( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 3.0, 0.99, inliers);
    num_inliers_ = inliers.rows;
    //    cout<<inliers<<endl;
    cout<<"pnp inliers: "<<num_inliers_<<endl;
    //    cout<<inliers<<endl;

    T_c_w_estimated_ = SE3 (
                SO3 ( rvec.at<double> ( 0,0 ), rvec.at<double> ( 1,0 ), rvec.at<double> ( 2,0 ) ),
                Vector3d ( tvec.at<double> ( 0,0 ), tvec.at<double> ( 1,0 ), tvec.at<double> ( 2,0 ) )
                );
    cout<<"T_c_w_estimated_ 0 : "<<endl<<T_c_w_estimated_.matrix()<<endl;

    // using bundle adjustment to optimize the pose
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            T_c_w_estimated_.rotation_matrix(), T_c_w_estimated_.translation()
                            ));
    optimizer.addVertex ( pose );

    //        bool_inlier_3d2d_.clear();
    //        vector<bool> tmp(good_matches_2d2d_.size(), false);
    //        tiangulation_points_good_.swap(tmp);
    // edges
    for ( int i=0; i<inliers.rows; i++ )
    {
//        cout<<inliers.row(i)<<endl;
        int index = inliers.at<int> ( i,0 );
        // 3D -> 2D projection
        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
        edge->setId ( i );
        edge->setVertex ( 0, pose );
        edge->camera_ = curr_->camera_.get();
        edge->point_ = Vector3d ( pts3d[index].x, pts3d[index].y, pts3d[index].z );
        edge->setMeasurement ( Vector2d ( pts2d[index].x, pts2d[index].y ) );
        edge->setInformation ( Eigen::Matrix2d::Identity());
        optimizer.addEdge ( edge );
        // set the inlier map points
        match_3dpts_[index]->matched_times_++;
        match_3dpts_[index]->descriptor_= descriptors_curr_.row( match_2dkp_index_[index]).clone();
//        cout<<"match_3dpts_[index]->matched_times_ : "<<match_3dpts_[index]->matched_times_<<"  "<<match_3dpts_[index]->visible_times_<<endl;
    }

    optimizer.initializeOptimization();
    optimizer.optimize (100 );

    T_c_w_estimated_ = SE3 (
                pose->estimate().rotation(),
                pose->estimate().translation()
                );

    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;
}

void VisualOdometry::triangulation()
{
    //    Mat T1 = (cv::Mat_<float> (3,4) <<
    //              1,0,0,0,
    //              0,1,0,0,
    //              0,0,1,0);
    Eigen::Matrix3d R=T_c_w_estimated_.rotation_matrix();
    Eigen::Vector3d t=T_c_w_estimated_.translation();
    Mat T2 = (cv::Mat_<float> (3,4) <<
              R(0,0), R(0,1), R(0,2), t(0,0),
              R(1,0), R(1,1), R(1,2), t(1,0),
              R(2,0), R(2,1), R(2,2), t(2,0)
              );

    R=ref_->T_c_w_.rotation_matrix();
    t=ref_->T_c_w_.translation();
    Mat T1 = (cv::Mat_<float> (3,4) <<
              R(0,0), R(0,1), R(0,2), t(0,0),
              R(1,0), R(1,1), R(1,2), t(1,0),
              R(2,0), R(2,1), R(2,2), t(2,0)
              //              R[0,0], R[0,1], R[0,2], t[0,0],
              //            R[1,0], R[1,1], R[1,2], t[1,0],
              //            R[2,0], R[2,1], R[2,2], t[2,0]
              );


    Mat K = ( cv::Mat_<double> ( 3,3 ) <<
              ref_->camera_->fx_, 0, ref_->camera_->cx_,
              0, ref_->camera_->fy_, ref_->camera_->cy_,
              0,0,1
              );

    //    vector<cv::Point2f> pts_1, pts_2;
    //    for ( DMatch m:matches )
    //    {
    //        // 将像素坐标转换至相机坐标
    //        pts_1.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
    //        pts_2.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
    //    }

    vector<cv::Point2f> points_ref;
    vector<cv::Point2f> points_curr;

    //载入goodmatch里的点对
    points_ref.clear();
    points_curr.clear();

    //    for ( int i = 0; i < ( int ) match_2dkp_index_ref_.size(); i++ )
    //    {
    //        points_ref.push_back ( keypoints_ref_[match_2dkp_index_ref_[i]].pt );
    //        points_curr.push_back ( keypoints_curr_[match_2dkp_index_curr_[i]].pt );
    //    }
    //    cout<<"points_ref.size() : "<<points_ref.size()<<endl;


    for ( int i = 0; i < ( int ) good_matches_2d2d_.size(); i++ )
    {
        points_ref.push_back ( ref_->camera_->pixel2cam(keypoints_ref_[good_matches_2d2d_[i].queryIdx].pt));
        points_curr.push_back ( curr_->camera_->pixel2cam(keypoints_curr_[good_matches_2d2d_[i].trainIdx].pt));
    }

    //    for ( int i = 0; i < ( int ) keypoints_ref_.size(); i++ )//把所有点push
    //    {
    //        points_ref.push_back ( ref_->camera_->pixel2cam(keypoints_ref_[i].pt));
    //        points_curr.push_back ( curr_->camera_->pixel2cam(keypoints_curr_[i].pt));
    //    }


    cout<<"points_ref.size() : "<<points_ref.size()<<endl;
    //调用opencv函数进行三角化
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, points_ref, points_curr, pts_4d );

    // 将三角化还原的三维点转换成非齐次坐标
    tiangulation_points_buff_.clear();
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
                    x.at<float>(0,0),
                    x.at<float>(1,0),
                    x.at<float>(2,0)
                    );
        //将三角化结果存入缓冲区
        tiangulation_points_buff_.push_back( p );
    }

    tiangulation_points_good_.clear();
    vector<bool> tmp(good_matches_2d2d_.size(), false);
    tiangulation_points_good_.swap(tmp);
    //验证三角化结果，计算平均误差与平均视角，用于评估三角化的适用性
    double mean_error_ref=0;
    double mean_error_cur=0;
    double mean_view_angle=0;
    triangulation_point_angle_enough_num_=0;
    for ( int i=0; i<good_matches_2d2d_.size(); i++ )
    {
        Eigen::Vector3d ptworld= Eigen::Vector3d( tiangulation_points_buff_[i].x, tiangulation_points_buff_[i].y, tiangulation_points_buff_[i].z );

        Point2d pt1_cam = ref_->camera_->pixel2cam( keypoints_ref_[ good_matches_2d2d_[i].queryIdx ].pt);
//        Point2d pt1_cam_3d(
//                    tiangulation_points_buff_[i].x/tiangulation_points_buff_[i].z,
//                    tiangulation_points_buff_[i].y/tiangulation_points_buff_[i].z
//                    );

        Eigen::Vector3d pt1_trans = ref_->T_c_w_.rotation_matrix()*( ptworld) + ref_->T_c_w_.translation();

        double l1=length3d(pt1_trans(0),pt1_trans(1),pt1_trans(2));
        double error_ref=distanceEuclidean2d(pt1_cam.x,pt1_cam.y,pt1_trans(0),pt1_trans(1));
        //        double error_ref=sqrt((pt1_cam.x-pt1_cam_3d.x)*(pt1_cam.x-pt1_cam_3d.x)+(pt1_cam.y-pt1_cam_3d.y)*(pt1_cam.y-pt1_cam_3d.y));
        mean_error_ref=(mean_error_ref*i+ error_ref)/(i+1);
        //        cout<<"point in the first camera frame: "<<pt1_cam<<endl;
        //        cout<<"point projected from 3D "<<pt1_cam_3d<<", d="<<tiangulation_points_buff[i].z<<endl;

        // 第二个图
        Point2f pt2_cam = ref_->camera_->pixel2cam( keypoints_curr_[ good_matches_2d2d_[i].trainIdx ].pt);

        Eigen::Vector3d pt2_trans = T_c_w_estimated_.rotation_matrix()*( ptworld) + T_c_w_estimated_.translation();
        double l2=length3d(pt2_trans(0),pt2_trans(1),pt2_trans(2));
        pt2_trans /= pt2_trans(2,0);

        double error_cur=distanceEuclidean2d(pt2_cam.x,pt2_cam.y,pt2_trans(0),pt2_trans(1));
        //        double error_cur=sqrt((pt2_cam.x-pt2_trans(0))*(pt2_cam.x-pt2_trans(0))+(pt2_cam.y-pt2_trans(1))*(pt2_cam.y-pt2_trans(1)));
        mean_error_cur=(mean_error_cur*i+error_cur )/(i+1);

        //        cout<<"point in the second camera frame: "<<pt2_cam<<endl;
        //        cout<<"point reprojected from second frame: "<<pt2_trans.transpose()<<endl;
        //        cout<<endl;

        //double d=T_c_w_estimated_.translation().norm();
        double d=length3d( T_c_w_estimated_.translation()(0), T_c_w_estimated_.translation()(1), T_c_w_estimated_.translation()(2));
        double angleRad=angleRadFromAcos(l1,l2,d);
        //        cout<<"error_cur : "<<error_cur<<"error_ref : "<<error_ref<<endl;

        if(state_ == INITIALIZING)
        {
            if(angleRad>min_view_angle_triangulation_init_&&error_cur<max_mean_view_error_triangulation_&&error_ref<max_mean_view_error_triangulation_)
            {
                tiangulation_points_good_[i]=true;
                triangulation_point_angle_enough_num_++;
            }
        }
        else
        {
            if(angleRad>min_view_angle_triangulation_&&error_cur<max_mean_view_error_triangulation_&&error_ref<max_mean_view_error_triangulation_)
            {
                tiangulation_points_good_[i]=true;
                triangulation_point_angle_enough_num_++;
            }
        }
        mean_view_angle=(mean_view_angle*i+angleRad )/(i+1);
        //cout<<" T_c_w_estimated_.translation()(0) "<<T_c_w_estimated_.translation()(0)<<" (1) "<<T_c_w_estimated_.translation()(1)<<" (2) "<<T_c_w_estimated_.translation()(2)<<endl;
        //cout<<" l1 "<<l1<<" l2 "<<l2<<" d "<<d<<endl;
        //cout<<"angleRad  :  "<<angleRad<<endl;
        //        cout<<"error_ref  :  "<<error_ref<<endl;
        //        cout<<"error_cur  :  "<<error_cur<<endl;
    }
    //存储并输出结果到命令行
    cout<<"good_matches_2d2d.size()  :  "<<good_matches_2d2d_.size()<<endl;
    mean_view_error_ref_triangulation_=mean_error_ref;
    mean_view_error_cur_triangulation_=mean_error_cur;
    mean_view_angle_triangulation_=mean_view_angle;
    cout<<"mean_error_ref  :  "<<mean_error_ref<<endl;
    cout<<"mean_error_cur  :  "<<mean_error_cur<<endl;
    cout<<"mean_view_angle  :   "<<mean_view_angle<<endl;
    cout<<"triangulation_point_angle_enough_num_ : "<<triangulation_point_angle_enough_num_<<endl;

    //    points_ref.clear();
    //    points_curr.clear();
    //    for ( int i = 0; i < ( int ) match_2dkp_index_ref_.size(); i++ )
    //    {
    //        points_ref.push_back ( ref_->camera_->pixel2cam(keypoints_ref_[match_2dkp_index_ref_[i]].pt));
    //        points_curr.push_back ( ref_->camera_->pixel2cam(keypoints_curr_[match_2dkp_index_curr_[i]].pt));
    //    }
    //    cout<<"points_ref.size() : "<<points_ref.size()<<endl;
}

bool VisualOdometry::checkEstimatedPose()
{

    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    cout<<"  inlier   : "<<num_inliers_<<endl;
    cout<<"  motion   : "<<d.norm() <<endl;
    if ( d.norm() > 1.0+0.2*(num_lost_+1) )
    {
        cout<<"reject because motion is too large: " <<endl;
        return false;
    }
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<endl;
        return false;
    }

    return true;
}


bool VisualOdometry::checkEstimatedInitPose()
{

    if ( match_2dkp_index_ref_.size() < min_good_matches_ )
    {
        cout<<"reject because num of good matches is too small: "<<match_2dkp_index_ref_.size() <<endl;
        return false;
    }
    // check if the estimated pose is good
    cout<<"Init checkEstimatedInitPose num_inliers_2d2d_: "<<num_inliers_2d2d_<<endl;
    if ( num_inliers_2d2d_ < match_2dkp_index_ref_.size()*num_inliers_2d2d_ratio_ )
    {
        cout<<"Init rejected because inlier is too small: "<<num_inliers_2d2d_<<endl;
        return false;
    }
    //    if ( mean_view_error_ref_triangulation_ >max_mean_view_error_triangulation_||mean_view_error_cur_triangulation_>max_mean_view_error_triangulation_)
    //    {
    //        cout<<"Init rejected because view error too large: "<<mean_view_error_ref_triangulation_<<" "<<mean_view_error_cur_triangulation_<<endl;
    //        return false;
    //    }

    if ( triangulation_point_angle_enough_num_<min_triangulation_point_angle_enough_num_)
    {
        cout<<"Init rejected because angle points too small: "<<triangulation_point_angle_enough_num_<<endl;
        return false;
    }

    //        if ( mean_view_angle_triangulation_ <min_mean_view_angle_triangulation_/3)
    //        {
    //            cout<<"Init rejected because angle too small: "<<mean_view_angle_triangulation_<<endl;
    //            return false;
    //        }

    // if the motion is too large, it is probably wrong
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    if ( d.norm() > 5.0 )//-----------------------------------------------------------------------/////////////////////////////////////////////////////
    {
        cout<<"Init reject because motion is too large: "<<d.norm() <<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    SE3 T_r_c = ref_->T_c_w_ * T_c_w_estimated_.inverse();
    Sophus::Vector6d d = T_r_c.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    cout<<"rot.norm() : "<<rot.norm()<<endl;
    cout<<"trans.norm() : "<<trans.norm()<<endl;
    if ( d.norm() > 1.0 )//-----------------------------------------------------------------------/////////////////////////////////////////////////////
    {
        return true;
    }
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    //    if ( map_->keyframes_.empty() )
    //    {
    //        // first key-frame, add all 3d points into map
    //        for ( size_t i=0; i<keypoints_curr_.size(); i++ )
    //        {
    //           double d =1.0;// curr_->findDepth ( keypoints_curr_[i] );
    //            if ( d < 0 )
    //                continue;
    //            Vector3d p_world = ref_->camera_->pixel2world (//not supported
    //                Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_c_w_, d
    //            );
    //            Vector3d n = p_world - ref_->getCamCenter();
    //            n.normalize();
    //            MapPoint::Ptr map_point = MapPoint::createMapPoint(
    //                p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
    //            );
    //            map_->insertMapPoint( map_point );
    //        }
    //    }
    if ( !map_->keyframes_.empty() )
    {
        curr_->T_c_w_=T_c_w_estimated_;
    }
    map_->insertKeyFrame ( curr_ );
    ref_ = curr_;
    cout<<"addKeyFrame"<<endl;
}

void VisualOdometry::addMapPointsTriangulation()
{
    // add the new map points into map
    if(state_ == INITIALIZING)//初始化，将三角化所得的点加入地图，将初始化点加入初始两帧的匹配点集
    {
        //    vector<bool> matched(keypoints_curr_.size(), false);
        //    for ( int index:match_2dkp_index_curr_ )
        //        matched[index] = true;

        //        for ( int i = 0; i < ( int ) good_matches_2d2d_.size(); i++ )
        //        {
        //            points_ref.push_back ( ref_->camera_->pixel2cam(keypoints_ref_[good_matches_2d2d_[i].queryIdx].pt));
        //            points_curr.push_back ( curr_->camera_->pixel2cam(keypoints_curr_[good_matches_2d2d_[i].trainIdx].pt));
        //        }

        //只有2d2d匹配结果好的结果才能作为地图点
        cout<<"keypoints_curr_.size() : "<<keypoints_curr_.size()<<endl;
        cout<<"tiangulation_points_buff_.size() : "<<tiangulation_points_buff_.size()<<endl;
        cout<<"descriptors_curr_.rows : "<<descriptors_curr_.rows<<endl;

        //        vector<bool> matched2d2d(keypoints_curr_.size(), false);
        //        for ( int index:match_2dkp_index_curr_ )
        //            matched2d2d[index] = true;
        for ( int i=0; i<good_matches_2d2d_.size(); i++ )
        {
            //        if ( matched[i] == true )
            //            continue;
            //        double d =1.0;// ref_->findDepth ( keypoints_curr_[i] );
            //        if ( d<0 )
            //            continue;
            //            if ( matched2d2d[i] == false )
            //                continue;
            if(tiangulation_points_good_[i]==true)
            {
                Vector3d p_world = Eigen::Vector3d( tiangulation_points_buff_[i].x, tiangulation_points_buff_[i].y, tiangulation_points_buff_[i].z );
                /*ref_->camera_->pixel2world (
                    Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
                    curr_->T_c_w_, d
                    );*/
                if(curr_->isDepthPositive(p_world))
                {
                    Vector3d n = p_world - ref_->getCamCenter();
                    n.normalize();
                    MapPoint::Ptr map_point = MapPoint::createMapPoint(
                                p_world, n, descriptors_curr_.row(good_matches_2d2d_[i].trainIdx).clone(), curr_.get()
                                );

                    map_->insertMapPoint( map_point);
                    ref_->insertFramePoint(map_point,keypoints_ref_[good_matches_2d2d_[i].queryIdx].pt);
                    curr_->insertFramePoint(map_point,keypoints_curr_[good_matches_2d2d_[i].trainIdx].pt);
                    cout<<tiangulation_points_buff_[i]<<endl;//初始化结果输出
                }
            }
        }
    }
    else
    {
        cout<<"keypoints_curr_.size() : "<<keypoints_curr_.size()<<endl;
        cout<<"tiangulation_points_buff_.size() : "<<tiangulation_points_buff_.size()<<endl;
        cout<<"descriptors_curr_.rows : "<<descriptors_curr_.rows<<endl;
        //只有2d2d匹配上而且3d2d没有匹配的点才能加入地图

        //        vector<bool> matched3d2d(good_matches_2d2d_.size(), false);
        //        for ( int index:match_2dkp_index_ )
        //            matched3d2d[index] = true;
        //        vector<bool> matched2d2d(keypoints_curr_.size(), false);
        //        for ( int index:match_2dkp_index_curr_ )
        //            matched2d2d[index] = true;
        for ( int i=0; i<good_matches_2d2d_.size(); i++ )
        {
            //            if ( matched3d2d[i] == true )
            //                continue;
            //            if ( matched2d2d[i] == false )
            //                continue;
            if(tiangulation_points_good_[i]==true)
            {
                vector<int>::iterator ret;
                ret = std::find(match_2dkp_index_.begin(), match_2dkp_index_.end(), good_matches_2d2d_[i].trainIdx);
                if(ret != match_2dkp_index_.end())
                {
                    //                    cout<<"Point already in map"<<endl;
                    continue;
                }
                else{
                    Vector3d p_world = Eigen::Vector3d( tiangulation_points_buff_[i].x, tiangulation_points_buff_[i].y, tiangulation_points_buff_[i].z );
                    /*ref_->camera_->pixel2world (
                        Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
                        curr_->T_c_w_, d
                        );*/
                    if(curr_->isDepthPositive(p_world))
                    {
                        Vector3d n = p_world - ref_->getCamCenter();
                        n.normalize();
                        MapPoint::Ptr map_point = MapPoint::createMapPoint(
                                    p_world, n, descriptors_curr_.row(good_matches_2d2d_[i].trainIdx).clone(), curr_.get()
                                    );

                        //                    cout<<tiangulation_points_buff_[i]<<endl;//结果输出
                        map_->insertMapPoint( map_point );
                        //                    ref_->insertFramePoint(map_point);
                        //                    curr_->insertFramePoint(map_point);
                        ref_->insertFramePoint(map_point,keypoints_ref_[good_matches_2d2d_[i].queryIdx].pt);
                        curr_->insertFramePoint(map_point,keypoints_curr_[good_matches_2d2d_[i].trainIdx].pt);
                    }
                }
            }
        }
    }
}


//void VisualOdometry::addMapPoints()
//{
//    // add the new map points into map
//    vector<bool> matched(keypoints_curr_.size(), false);
//    for ( int index:match_2dkp_index_ )
//        matched[index] = true;
//    for ( int i=0; i<keypoints_curr_.size(); i++ )
//    {
//        if ( matched[i] == true )
//            continue;
//        double d =1.0;// ref_->findDepth ( keypoints_curr_[i] );
//        if ( d<0 )
//            continue;
//        Vector3d p_world = ref_->camera_->pixel2world (
//                    Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ),
//                    curr_->T_c_w_, d
//                    );
//        Vector3d n = p_world - ref_->getCamCenter();
//        n.normalize();
//        MapPoint::Ptr map_point = MapPoint::createMapPoint(
//                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
//                    );
//        map_->insertMapPoint( map_point );
//    }
//}

void VisualOdometry::optimizeMap()
{
    // remove the hardly seen and no visible points
    int k1=0;
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end(); )
    {
        if ( !ref_->isInFrame(iter->second->pos_) /*&&k1%10!=0*/)
        {
            iter = map_->map_points_.erase(iter);
            k1++;
            continue;
        }

        if(state_ == OK)
        {
            if ((iter->second->matched_times_ < 3)&&(iter->second->visible_times_> 4))
            {
                iter = map_->map_points_.erase(iter);
                continue;
            }
        }

        float match_ratio = float(iter->second->matched_times_)/iter->second->visible_times_;
        if ( match_ratio < map_point_erase_ratio_ )
        {
            iter = map_->map_points_.erase(iter);
            continue;
        }


        //        double angle = getViewAngle( curr_, iter->second );
        //        if ( angle > M_PI/6. )
        //        {
        //            iter = map_->map_points_.erase(iter);
        //            continue;
        //        }

        if ( iter->second->good_ == false )
        {
            // TODO try triangulate this map point
        }
        iter++;
    }

    //    if ( match_2dkp_index_.size()<100 )
    //addMapPoints();
    if ( map_->map_points_.size() > 200 &&map_point_erase_ratio_<0.85)
    {
        // TODO map is too large, remove some one
        cout<<"map_point_erase_ratio_ : "<<map_point_erase_ratio_<<endl;
        map_point_erase_ratio_ += 0.1;
    }
    else
    {map_point_erase_ratio_ = 0.1;}
    cout<<"map points: "<<map_->map_points_.size()<<endl;
}

void VisualOdometry::globalBundleAdjustment()
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> Block;
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block ( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    long unsigned int maxKFid = 0;
    long unsigned int minKFid = 999999;
    int deltaf = 20;

    for(auto iter = map_->keyframes_.begin(); iter != map_->keyframes_.end();iter++)
    {
        Frame::Ptr f= iter->second;
        if(f->id_>maxKFid)
        {maxKFid=f->id_;}
    }
    for(auto iter = map_->keyframes_.begin(); iter != map_->keyframes_.end();iter++)
    {
        Frame::Ptr f= iter->second;
        if((f->id_<minKFid)&&(int(f->id_)>(int(maxKFid)-deltaf)))
        {minKFid=f->id_;}
    }
    cout<<"minKFid : "<<minKFid<<endl;
    for(auto iter = map_->keyframes_.begin(); iter != map_->keyframes_.end();iter++)
    {
        Frame::Ptr f= iter->second;

        //        if(f->id_==0||f->id_>=minKFid)
        //        {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(g2o::SE3Quat (
                              f->T_c_w_.rotation_matrix(), f->T_c_w_.translation()
                              ));
        vSE3->setId(f->id_);
        vSE3->setFixed(f->id_==minKFid);
        optimizer.addVertex(vSE3);
//        cout<<"f->id_ : "<<f->id_<<endl;
        //        }

        //            cout<<"T_c_w_ : "<<endl<<f->T_c_w_.matrix()<<endl;
    }


    int nEdges = 0 ;
    for(auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();iter++)
    {
        MapPoint::Ptr& pMP =  iter->second;//某个地图点
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(pMP->pos_);
        const int id = pMP->id_+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        //        vector<int> observations  =  pMP->getObs();
        for(size_t j = 0;j<iter->second->observed_frames_.size();j++)
        {
            //            int obsframeid = observations[j] ;//某观测到地图点的关键帧

            Frame* ff = iter->second->observed_frames_[j];
            for(auto iter_point_ptr = ff->matched_map_points_.begin(); iter_point_ptr != map_->map_points_.end();iter_point_ptr++)
            {
                if(iter_point_ptr->second->id_ == pMP->id_)//此关键帧观测到的点
                {
                    Eigen::Matrix<double,2,1> obs ;
                    Point2f& obs_tmp= ff->matched_map_points_position_.at(iter_point_ptr->first);
                    obs << obs_tmp.x,obs_tmp.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
                    //                    EdgeProjectXYZ2UVPoseOnly* e  = new EdgeProjectXYZ2UVPoseOnly();
                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(ff->id_)));
                    e->setMeasurement(obs);
                    e->setInformation(Eigen::Matrix2d::Identity());

                    //                        g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    //                        e->setRobustKernel(rk);
                    //                        rk->setDelta(thHuber2D);
                    e->fx = ff->camera_->fx_;
                    e->fy = ff->camera_->fy_;
                    e->cx =ff->camera_->cx_;
                    e->cy = ff->camera_->cy_;
                    optimizer.addEdge(e);
                    nEdges++;
                }
            }
        }
    }
    cout<<"nEdges : "<<nEdges<<endl;
    if(nEdges==0)
    {
        cout<<" no possible edges  return !! "<<endl;
        return ;
    }

    optimizer.initializeOptimization();
    optimizer.optimize(500);


    for(auto iter = map_->keyframes_.begin(); iter != map_->keyframes_.end();iter++)
    {
        Frame::Ptr f= iter->second;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(f->id_));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        f->T_c_w_= Sophus::SE3(SE3quat.rotation(),SE3quat.translation());

        //        cout<<"T_c_w_ : "<<endl<<f->T_c_w_.matrix()<<endl;
    }

    for(auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();iter++)
    {
        MapPoint::Ptr& pMP =  iter->second;//某个地图点
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->id_+maxKFid+1));
        pMP->pos_=vPoint->estimate();
        //        pMP->UpdateWorldPos(Convert::toCvMat(vPoint->estimate()));
    }
    sleep(0.001);
    //    cout<<"T_c_w_estimated_: "<<endl<<T_c_w_estimated_.matrix()<<endl;

}


void VisualOdometry::removeMultiPoints()
{

    vector<cv::DMatch> matches;
    // select the candidates in map
    Mat desp_map_1,desp_map_2;
    vector<unordered_map<unsigned long, MapPoint::Ptr>::iterator> candidate;
    cout<<"removeMultiPoints start"<<endl;
    for ( auto iter = map_->map_points_.begin(); iter != map_->map_points_.end();iter++)
    {
        MapPoint::Ptr& p = iter->second;
        candidate.push_back( iter );
        //        cout<<p->id_<<endl;
        desp_map_1.push_back( p->descriptor_ );
        desp_map_2.push_back( p->descriptor_ );
    }

    matcher_flann_.match ( desp_map_1, desp_map_2, matches );

    vector<unordered_map<unsigned long, MapPoint::Ptr>::iterator> pointsToRemove;
    int goodmatchcnt=0;
    cout<<"matches.size() : "<<matches.size()<<endl;
    for ( cv::DMatch& m : matches )
    {
        if(m.distance!=0)
        {
            goodmatchcnt++;
            Point3f pt1=candidate[m.queryIdx]->second->getPositionCV();
            Point3f pt2=candidate[m.trainIdx]->second->getPositionCV();
            double pointDistance=distanceEuclidean3d(pt1.x,pt1.y,pt1.z,pt2.x,pt2.y,pt2.z);
            if ( m.distance < 30&& pointDistance<5.0)
            {
                int matched_times_1=candidate[m.queryIdx]->second->matched_times_;
                int matched_times_2=candidate[m.trainIdx]->second->matched_times_;
                if(matched_times_1>matched_times_2)//移除匹配次数更少的点
                {
                    pointsToRemove.push_back(candidate[m.trainIdx]);
                    continue;
                }
                if(matched_times_1<matched_times_2)
                {
                    pointsToRemove.push_back(candidate[m.queryIdx]);
                    continue;
                }
            }
        }
    }
    cout<<"goodmatchcnt : "<<goodmatchcnt<<endl;

    for ( auto& iter:pointsToRemove)
    {
        map_->map_points_.erase(iter);
    }
    cout<<"pointsToRemove.size() : "<<pointsToRemove.size()<<endl;
    //    vector<cv::DMatch> matches;
    //    // select the candidates in map
    //    Mat desp_map_1,desp_map_2;
    //    vector<MapPoint::Ptr> candidate;
    //    for ( auto& allpoints: map_->map_points_ )
    //    {
    //        MapPoint::Ptr& p = allpoints.second;
    //        candidate.push_back( p );
    //        desp_map_1.push_back( p->descriptor_ );
    //        desp_map_2.push_back( p->descriptor_ );
    //    }
    //    matcher_flann_.match ( desp_map_1, desp_map_2, matches );
    //    vector<MapPoint::Ptr> pointsToRemove;
    //    for ( cv::DMatch& m : matches )
    //    {
    //        if(m.distance!=0)
    //        {
    //            Point3f pt1=candidate[m.queryIdx]->getPositionCV();
    //            Point3f pt2=candidate[m.trainIdx]->getPositionCV();
    //            double pointDistance=distanceEuclidean3d(pt1.x,pt1.y,pt1.z,pt2.x,pt2.y,pt2.z);
    //            if ( m.distance < 30&& pointDistance<1.0)
    //            {
    //                int matched_times_1=candidate[m.queryIdx]->matched_times_;
    //                int matched_times_2=candidate[m.trainIdx]->matched_times_;
    //                if(matched_times_1>matched_times_2)//移除匹配次数更少的点
    //                {
    //                    pointsToRemove.push_back(candidate[m.trainIdx]);
    //                    continue;
    //                }
    //                if(matched_times_1<matched_times_2)
    //                {
    //                    pointsToRemove.push_back(candidate[m.queryIdx]);
    //                    continue;
    //                }
    //            }
    //        }
    //    }

    //移除重复点

}

double VisualOdometry::getViewAngle ( Frame::Ptr frame, MapPoint::Ptr point )
{
    Vector3d n = point->pos_ - frame->getCamCenter();
    n.normalize();
    return acos( n.transpose()*point->norm_ );
}

double VisualOdometry::distanceEuclidean2d(double x1,double y1,double x2,double y2)
{
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

double VisualOdometry::distanceEuclidean3d(double x1,double y1,double z1,double x2,double y2,double z2)
{
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
}

double VisualOdometry::length3d(double x,double y,double z)
{
    return sqrt(x*x+y*y+z*z);
}

double VisualOdometry::angleRadFromAcos(double l1,double l2,double d)
{
    return acos((l1*l1+l2*l2-d*d)/(2*l1*l2));
}
void *VisualOdometry::ptrGlobalBundleAdjustment()
{
    globalBundleAdjustment();
}
}
