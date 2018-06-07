// -------------- test the visual odometry -------------
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

string int2string(int value)
{
    stringstream ss;
    ss<<value;
    return ss.str();
}

int main ( int argc, char** argv )
{
    if ( argc != 2 )
    {
        cout<<"usage: run_vo parameter_file"<<endl;
        return 1;
    }

    myslam::Config::setParameterFile ( argv[1] );
    myslam::VisualOdometry::Ptr vo ( new myslam::VisualOdometry );

    string dataset_dir = myslam::Config::get<string> ( "dataset_dir" );
    cout<<"dataset: "<<dataset_dir<<endl;
    ifstream fin ( dataset_dir+"/rgb.txt" );
    if ( !fin )
    {
        //        cout<<"please generate the associate file called associate.txt!"<<endl;
        cout<<"rgb.txt not found!"<<endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    while ( !fin.eof() )
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file;//>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        //        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        //        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }



    time_t t=std::time(0);
    struct tm * now = std::localtime( & t );
    //    tm* p = localtime(&timep);
    //    char log_file[100] ;//= {0};
    string logfilepath = "/home/null/slam/log/";
    //    string outfilename = "rgb.txt";
    string log_file(logfilepath+"slamlog"+
                    '-'+int2string(now->tm_year + 1900)+
                    '-'+int2string(now->tm_mon + 1)+
                    '-'+int2string(now->tm_mday)+
                    '-'+int2string(now->tm_hour)+
                    '-'+int2string(now->tm_min)+
                    '-'+int2string(now->tm_sec)+".log");
    ofstream slam_log;
    slam_log.open(log_file/*.data()*/,ios::out|ios::trunc);

    int map_width=800;
    int map_height=800;
    cv::namedWindow("Trajectory", cv::WINDOW_AUTOSIZE);
    cv::Mat traj = cv::Mat::zeros(map_width, map_height, CV_8UC3);// 用于绘制轨迹



    myslam::Camera::Ptr camera ( new myslam::Camera );

    // visualization
    cv::viz::Viz3d vis ( "Visual Odometry" );
    cv::viz::WCoordinateSystem world_coor ( 1.0 ), camera_coor ( 0.5 );
    cv::Point3d cam_pos ( 0, -1.0, -1.0 ), cam_focal_point ( 0,0,0 ), cam_y_dir ( 0,1,0 );
    cv::Affine3d cam_pose = cv::viz::makeCameraPose ( cam_pos, cam_focal_point, cam_y_dir );
    vis.setViewerPose ( cam_pose );

    world_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 2.0 );
    camera_coor.setRenderingProperty ( cv::viz::LINE_WIDTH, 1.0 );
//    vis.showWidget ( "World", world_coor );
//    vis.showWidget ( "Camera", camera_coor );

    cout<<"read total "<<rgb_files.size() <<" entries"<<endl;
    vo->init_cnt_frame=0;
    for ( int i=0; i<rgb_files.size(); i++ )
    {
        cout<<"****** loop "<<i<<" ******"<<endl;
        Mat color = cv::imread ( rgb_files[i] );
        cout<<"rgb_files[i] : "<<rgb_files[i]<<endl;
        //        Mat depth = cv::imread ( depth_files[i], -1 );
        if ( color.data==nullptr/* || depth.data==nullptr*/ )
            break;
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        //        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame ( pFrame );
        cout<<"VO costs time: "<<timer.elapsed() <<endl;

        if ( vo->state_ == myslam::VisualOdometry::LOST )
            break;
        SE3 Twc = pFrame->T_c_w_.inverse();

        double scale=5;

        double pos_x=Twc.translation()(0);
        double pos_z=Twc.translation()(2);

        int draw_x = int(-pos_x*scale) + 400;
        int draw_y = int(pos_z*scale) + 100;

        cv::circle(traj, cv::Point(draw_x, draw_y), 1, CV_RGB(255, 0, 0), 2);

        cv::rectangle(traj, cv::Point(10, 30), cv::Point(580, 60), CV_RGB(0, 0, 0), CV_FILLED);
        cv::imshow("Trajectory", traj);


        slam_log<<Twc.log().transpose()<<endl;

        // show the map and the camera pose
        cv::Affine3d M (
                    cv::Affine3d::Mat3 (
                        Twc.rotation_matrix() ( 0,0 ), Twc.rotation_matrix() ( 0,1 ), Twc.rotation_matrix() ( 0,2 ),
                        Twc.rotation_matrix() ( 1,0 ), Twc.rotation_matrix() ( 1,1 ), Twc.rotation_matrix() ( 1,2 ),
                        Twc.rotation_matrix() ( 2,0 ), Twc.rotation_matrix() ( 2,1 ), Twc.rotation_matrix() ( 2,2 )
                        ),
                    cv::Affine3d::Vec3 (
                        Twc.translation() ( 0,0 ), Twc.translation() ( 1,0 ), Twc.translation() ( 2,0 )
                        )
                    );

        Mat img_show = color.clone();
        for ( auto& pt:vo->map_->map_points_ )
        {
            myslam::MapPoint::Ptr p = pt.second;
            Vector2d pixel = pFrame->camera_->world2pixel ( p->pos_, pFrame->T_c_w_ );
            cv::circle ( img_show, cv::Point2f ( pixel ( 0,0 ),pixel ( 1,0 ) ), 3, cv::Scalar ( 0,255,0 ), 1 );
        }

        cv::imshow ( "image", img_show );
        cv::waitKey ( 1 );
//        vis.setWidgetPose ( "Camera", M );
//        vis.spinOnce ( 1, false );
        cout<<endl;
    }

    return 0;
}