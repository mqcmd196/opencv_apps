// -*- coding:utf-8-unix; mode: c++; indent-tabs-mode: nil; c-basic-offset: 2; -*-
/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2014, Kei Okada.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Kei Okada nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

// https://github.com/Itseez/opencv/blob/2.4/samples/cpp/tutorial_code/ImgTrans/HoughCircle_Demo.cpp
/**
 * @file HoughCircle_Demo.cpp
 * @brief Demo code for Hough Transform
 * @author OpenCV team
 */

#include <ros/ros.h>
#include "opencv_apps/nodelet.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dynamic_reconfigure/server.h>
#include "opencv_apps/HoughCirclesConfig.h"
#include "opencv_apps/Circle.h"
#include "opencv_apps/CircleArray.h"
#include "opencv_apps/CircleArrayStamped.h"

namespace hough_circles {
class HoughCirclesNodelet : public opencv_apps::Nodelet
{
  image_transport::Publisher img_pub_;
  image_transport::Subscriber img_sub_;
  image_transport::CameraSubscriber cam_sub_;
  ros::Publisher msg_pub_;

  boost::shared_ptr<image_transport::ImageTransport> it_;

  hough_circles::HoughCirclesConfig config_;
  dynamic_reconfigure::Server<hough_circles::HoughCirclesConfig> srv;

  bool debug_view_;
  ros::Time prev_stamp_;

  std::string window_name_;
  static bool need_config_update_;

  // initial and max values of the parameters of interests.
  int canny_threshold_initial_value_;
  int accumulator_threshold_initial_value_;
  int max_accumulator_threshold_;
  int max_canny_threshold_;
  double canny_threshold_; int canny_threshold_int; // for trackbar
  double accumulator_threshold_; int accumulator_threshold_int;
  int gaussian_blur_size_;
  double gaussian_sigma_x_; int gaussian_sigma_x_int;
  double gaussian_sigma_y_; int gaussian_sigma_y_int;
  int voting_threshold_;
  double dp_; int dp_int;
  int min_circle_radius_;
  int max_circle_radius_;

  void reconfigureCallback(hough_circles::HoughCirclesConfig &new_config, uint32_t level)
  {
    config_ = new_config;
    canny_threshold_ = config_.canny_threshold;
    accumulator_threshold_ = config_.accumulator_threshold;
    gaussian_blur_size_ = config_.gaussian_blur_size;
    gaussian_sigma_x_ = config_.gaussian_sigma_x;
    gaussian_sigma_y_ = config_.gaussian_sigma_y;
    dp_ = config_.dp;
    min_circle_radius_ = config_.min_circle_radius;
    max_circle_radius_ = config_.max_circle_radius;

    canny_threshold_int = int(canny_threshold_);
    accumulator_threshold_int = int(accumulator_threshold_);
    gaussian_sigma_x_int = int(gaussian_sigma_x_);
    gaussian_sigma_y_int = int(gaussian_sigma_y_);
    dp_int = int(dp_);
  }

  const std::string &frameWithDefault(const std::string &frame, const std::string &image_frame)
  {
    if (frame.empty())
      return image_frame;
    return frame;
  }

  void imageCallbackWithInfo(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& cam_info)
  {
    do_work(msg, cam_info->header.frame_id);
  }

  void imageCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    do_work(msg, msg->header.frame_id);
  }

  static void trackbarCallback( int value, void* userdata)
  {
    need_config_update_ = true;
  }

  void do_work(const sensor_msgs::ImageConstPtr& msg, const std::string input_frame_from_msg)
  {
    // Work on the image.
    try
    {
      // Convert the image into something opencv can handle.
      cv::Mat frame = cv_bridge::toCvShare(msg, msg->encoding)->image;

      // Messages
      opencv_apps::CircleArrayStamped circles_msg;
      circles_msg.header = msg->header;

      // Do the work
      std::vector<cv::Rect> faces;
      cv::Mat src_gray, edges;

      if ( frame.channels() > 1 ) {
        cv::cvtColor( frame, src_gray, cv::COLOR_BGR2GRAY );
      } else {
        src_gray = frame;
      }

      // create the main window, and attach the trackbars
      if( debug_view_) {
        cv::namedWindow( window_name_, cv::WINDOW_AUTOSIZE );

        cv::createTrackbar("Canny Threshold", window_name_, &canny_threshold_int, max_canny_threshold_, trackbarCallback);
        cv::createTrackbar("Accumulator Threshold", window_name_, &accumulator_threshold_int, max_accumulator_threshold_, trackbarCallback);
        cv::createTrackbar("Gaussian Blur Size", window_name_, &gaussian_blur_size_, 30, trackbarCallback);
        cv::createTrackbar("Gaussian Sigam X", window_name_, &gaussian_sigma_x_int, 10, trackbarCallback);
        cv::createTrackbar("Gaussian Sigma Y", window_name_, &gaussian_sigma_y_int, 10, trackbarCallback);
        cv::createTrackbar("Dp", window_name_, &dp_int, 100, trackbarCallback);
        cv::createTrackbar("Min Circle Radius", window_name_, &min_circle_radius_, 500, trackbarCallback);
        cv::createTrackbar("Max Circle Radius", window_name_, &max_circle_radius_, 2000, trackbarCallback);

        if (need_config_update_) {
          config_.canny_threshold = canny_threshold_ = (double)canny_threshold_int;
          config_.accumulator_threshold = accumulator_threshold_ = (double)accumulator_threshold_int;
          config_.gaussian_blur_size = gaussian_blur_size_;
          config_.gaussian_sigma_x = gaussian_sigma_x_ = (double)gaussian_sigma_x_int;
          config_.gaussian_sigma_y = gaussian_sigma_y_ = (double)gaussian_sigma_y_int;
          config_.dp = dp_int;
          config_.min_circle_radius = min_circle_radius_;
          config_.max_circle_radius = max_circle_radius_;
          srv.updateConfig(config_);
          need_config_update_ = false;
        }
      }

      // Reduce the noise so we avoid false circle detection
      // gaussian_blur_size_ must be odd number
      if (gaussian_blur_size_%2 != 1) {
        gaussian_blur_size_ = gaussian_blur_size_ + 1;
      }
      cv::GaussianBlur( src_gray, src_gray, cv::Size(gaussian_blur_size_, gaussian_blur_size_), gaussian_sigma_x_, gaussian_sigma_y_ );

      // those paramaters cannot be =0
      // so we must check here
      canny_threshold_ = std::max(canny_threshold_, 1.0);
      accumulator_threshold_ = std::max(accumulator_threshold_, 1.0);

      //runs the detection, and update the display
      // will hold the results of the detection
      std::vector<cv::Vec3f> circles;
      // runs the actual detection
      cv::HoughCircles( src_gray, circles,
                        CV_HOUGH_GRADIENT,
                        dp_,
                        src_gray.rows/8,
                        canny_threshold_,
                        accumulator_threshold_,
                        min_circle_radius_,
                        max_circle_radius_ );

      // clone the colour, input image for displaying purposes
      for( size_t i = 0; i < circles.size(); i++ )
      {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( frame, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( frame, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );

        opencv_apps::Circle circle_msg;
        circle_msg.center.x = center.x;
        circle_msg.center.y = center.y;
        circle_msg.radius = radius;
        circles_msg.circles.push_back(circle_msg);
      }

      // shows the results
      if( debug_view_) {
        cv::imshow( window_name_, frame );
        int c = cv::waitKey(1);
      }

      // Publish the image.
      sensor_msgs::Image::Ptr out_img = cv_bridge::CvImage(msg->header, msg->encoding,frame).toImageMsg();
      img_pub_.publish(out_img);
      msg_pub_.publish(circles_msg);
    }
    catch (cv::Exception &e)
    {
      NODELET_ERROR("Image processing error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
    }

    prev_stamp_ = msg->header.stamp;
  }

  void subscribe()
  {
    NODELET_DEBUG("Subscribing to image topic.");
    if (config_.use_camera_info)
      cam_sub_ = it_->subscribeCamera("image", 3, &HoughCirclesNodelet::imageCallbackWithInfo, this);
    else
      img_sub_ = it_->subscribe("image", 3, &HoughCirclesNodelet::imageCallback, this);
  }

  void unsubscribe()
  {
    NODELET_DEBUG("Unsubscribing from image topic.");
    img_sub_.shutdown();
    cam_sub_.shutdown();
  }

public:
  virtual void onInit()
  {
    Nodelet::onInit();
    it_ = boost::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(*nh_));

    pnh_->param("debug_view", debug_view_, false);
    if (debug_view_) {
      always_subscribe_ = debug_view_;
    }
    prev_stamp_ = ros::Time(0, 0);

    window_name_ = "Hough Circle Detection Demo";
    canny_threshold_initial_value_ = 200;
    accumulator_threshold_initial_value_ = 50;
    max_accumulator_threshold_ = 200;
    max_canny_threshold_ = 255;

    //declare and initialize both parameters that are subjects to change
    canny_threshold_ = canny_threshold_initial_value_;
    accumulator_threshold_ = accumulator_threshold_initial_value_;
    
    dynamic_reconfigure::Server<hough_circles::HoughCirclesConfig>::CallbackType f =
      boost::bind(&HoughCirclesNodelet::reconfigureCallback, this, _1, _2);
    srv.setCallback(f);

    img_pub_ = advertiseImage(*pnh_, "image", 1);
    msg_pub_ = advertise<opencv_apps::CircleArrayStamped>(*pnh_, "circles", 1);

    onInitPostProcess();
  }
};
bool HoughCirclesNodelet::need_config_update_ = false;
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(hough_circles::HoughCirclesNodelet, nodelet::Nodelet);
