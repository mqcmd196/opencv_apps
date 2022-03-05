// -*- coding:utf-8-unix; mode: c++; indent-tabs-mode: nil; c-basic-offset: 2; -*-
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2022, JSK Lab.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
n *     notice, this list of conditions and the following disclaimer.
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
#include <ros/ros.h>
#include "opencv_apps/nodelet.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include <pluginlib/class_list_macros.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/saliency.hpp>

#include <dynamic_reconfigure/server.h>
#include "opencv_apps/SaliencyConfig.h"

namespace opencv_apps{
  class SaliencyFineGrained;
  template <typename Config>

  class SaliencyNodelet : public opencv_apps::Nodelet{
    protected:
      image_transport::Publisher img_pub_;
      image_transport::Publisher saliency_map_pub;
      image_transport::Subscriber img_sub_;
      image_transport::CameraSubscriber cam_sub_;

      std::shared_ptr<image_transport::ImageTransport> it_;

      typedef dynamic_reconfigure::Server<Config> ReconfigureServer;
      Config config_;
      std::shared_ptr<ReconfigureServer> reconfigure_server_;

      int queue_size_;
      bool debug_view_;

      std::string window_name_;

      boost::mutex mutex_;

      virtual void reconfigureCallback(Config& new_config, uint32_t level) = 0;

      virtual void filter(const cv::Mat& input_image, cv::Mat& output_image) = 0;

      void imageCallbackWithInfo(const sensor_msgs::ImageConstPtr& msg,
                                 const sensor_msgs::CameraInfoConstPtr& cam_info){
        doWork(msg, cam_info->header.frame_id);
      }

      void imageCallback(const sensor_msgs::ImageConstPtr& msg){
        doWork(msg, msg->header.frame_id);
      }

      void doWork(const sensor_msgs::ImageConstPtr& image_msg,
                  const std::string& input_frame_from_msg){
        try{
          cv::Mat frame = cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::BGR8)->image;
          cv::Mat map_frame;
          cv::Mat out_frame;
          filter(frame, out_frame);

          if (debug_view_){
            cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
          }

          std::string new_window_name;

          if (debug_view_){
            if (window_name_ != new_window_name){
              cv::destroyWindow(window_name_);
              window_name_ = new_window_name;
            }
            cv::imshow(window_name_, out_frame);
            int c = cv::waitKey(1);
          }

          sensor_msgs::Image::Ptr out_img =
            cv_bridge::CvImage(image_msg->header, sensor_msgs::image_encodings::MONO8, out_frame).toImageMsg();
          img_pub_.publish(out_img);

        } catch (cv::Exception& e) {
          NODELET_ERROR("Image procesing error: %s %s %s %i", e.err.c_str(), e.func.c_str(), e.file.c_str(), e.line);
        }
      }

      void subscribe(){
        NODELET_DEBUG("Subscribing to image topic\n");
        if (config_.use_camera_info)
          cam_sub_ = it_->subscribeCamera("image", queue_size_, &SaliencyNodelet::imageCallbackWithInfo, this);
        else
          img_sub_ = it_->subscribe("image", queue_size_, &SaliencyNodelet::imageCallback, this);
      }

      void unsubscribe(){
        NODELET_DEBUG("Unsubscribing from image topic\n");
        img_sub_.shutdown();
        cam_sub_.shutdown();
      }

    public:
      virtual void onInit(){
        Nodelet::onInit();
        it_ = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(*nh_));

        pnh_->param("queue_size", queue_size_, 3);
        pnh_->param("debug_view", debug_view_, false);

        if (debug_view_){
          always_subscribe_ = true;
        }

        window_name_ = "Static Saliency Demo";

        img_pub_ = advertiseImage(*pnh_, "image", 1);

        onInitPostProcess();
      }
  };

  class SaliencyFineGrainedNodelet : public SaliencyNodelet<opencv_apps::SaliencyConfig>{
    void filter(const cv::Mat& input_image, cv::Mat& output_image){
      cv::Mat saliency_map;
      std::string saliency_algorithm = "FINE_GRAINED";
      cv::Ptr<cv::saliency::Saliency> algorithm = cv::saliency::Saliency::create(saliency_algorithm);
      algorithm->computeSaliency(input_image, saliency_map);

    }
  };
}

PLUGINLIB_EXPORT_CLASS(opencv_apps::SaliencyFineGrained, nodelet::Nodelet);
