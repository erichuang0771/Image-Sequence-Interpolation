#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/stitching/stitcher.hpp"
#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"
#include "uFind.h"
#include "create_mid_points.h"
#include "stitching_two.h"
#include "cutting.h"

//#define CROP
#define EXTEND -1


/*Plan:
 */

/*	Done:
 * 2) Image blending method? ---->done 10.01.2015
 * 3) New segmentation method?
 * 4) pgm --> ppm --> jpg? ------>done: imwrite(".ppm")31.01.2015
 * 5) download test dateset! ------->peanut2 done:31.01.2015
 * 6) Why I cannot prefectly stitch the bg -----> 15.01.2015
 * 7) Color Image -----> done 23.01.2015
 * 8) Fish eye camera??? --->Done 30.01.2015
 * 9) multiple input
 *
 */


 using namespace cv;
 std::vector<DMatch> good_matches;
 std::vector<Point2f> obj;
 std::vector<Point2f> scene;
 std::vector<KeyPoint> keypoints_object, keypoints_scene, keypoints_object_org,
 keypoints_scene_org;
 Mat img_object;
 Mat img_scene;
 Mat img_seg;
 int y = 500;
 cv::Size sze;
 void readme();
 std::vector<Point2f> middle;



 int loop(string obj_dir, string sec_dir, int m_size, int NNN, int frame_counter, int file_name_counter);
/** @function main */


 void parse_argv(int argc, char** argv) {
  if(argc < 4) {
    printf("Incorrect usage: ImageD image_directory num_of_stes starting_image \n");
    exit(-1);
  }
}
/*
* Read images and call loop functions to stitching them.
*/
int main(int argc, char** argv) {



  parse_argv(argc, argv);
  int num_of_sets = atoi(argv[2]);
  int file_name_counter = atoi(argv[3]);
  for(int set_counter = 0; set_counter < num_of_sets; set_counter++) {
    string dir = argv[1];
    char file_name[10];
    std::sprintf(file_name,"%08d",file_name_counter);
    string obj_dir = dir + file_name + ".pgm";
    file_name_counter = file_name_counter + 4;
    std::sprintf(file_name,"%08d",file_name_counter);
    string sec_dir = dir + file_name + ".pgm";
    std::cout<<obj_dir<<"\n"<<sec_dir<<"\n";
    loop(obj_dir,sec_dir, 100, 4, 0, file_name_counter - 4 );
    std::cout<<"file_name_counter"<<file_name_counter - 4 <<"\n";
        /*  for(int frame_counter = 0; frame_counter < -1; frame_counter++) {
              std::cout<<"panoroma: "<<frame_counter<<"\n\n\n";
              char seed_dir[20];
              std::sprintf(seed_dir,"./results/result: %d.ppm",frame_counter);
              //future frames
              for(int seed_counter = 1; seed_counter <= EXTEND ; seed_counter++) {
                  std::cout<<"\n\n\ninterval: "<<seed_counter<<"\n\n\n";

                  char other_dir[10];
                  std::sprintf(other_dir,"%08d",file_name_counter + seed_counter*5);
                  string other_complete_dir = dir + other_dir + ".pgm";
                  //string seed_complete_dir = dir + seed_dir + ".pgm";
                  std::cout<<seed_dir<<"\n"<<other_complete_dir<<"\n";

                  loop(other_complete_dir, seed_dir,1000, 1, frame_counter, file_name_counter + 5*(set_counter));
              }

              //past frames
              for(int seed_counter = 1; seed_counter <= EXTEND; seed_counter++) {
                  std::cout<<"\n\n\ninterval: "<<seed_counter<<"\n\n\n";

                  char other_dir[10];
                  std::sprintf(other_dir,"%08d",file_name_counter - seed_counter*5);
                  string other_complete_dir = dir + other_dir + ".pgm";
                  //string seed_complete_dir = dir + seed_dir + ".pgm";
                  std::cout<<seed_dir<<"\n"<<other_complete_dir<<"\n";

                  loop(other_complete_dir, seed_dir,1000, 1, frame_counter, file_name_counter + 5*(set_counter));
              }
              std::cout<<"\n\n\n\n\n\n\n\n\n\n";
          }*/
            }
          }

/*** @ main loop function***/
//int main(int argc, char** argv) {
          int loop(string obj_dir, string sec_dir, int m_size, int NNN, int frame_counter, int file_name_counter) {
    /* if (argc != 5) {
         readme();
         return -1;
     }
     */

     std::cout<<"\n************loop begin*****************\n\n";
    Mat img_object_org = imread(obj_dir, CV_LOAD_IMAGE_COLOR); //new
    Mat img_scene_org = imread(sec_dir, CV_LOAD_IMAGE_COLOR);  //previous result
    if (!img_object_org.data || !img_scene_org.data) {
      std::cout << " --(!) Error reading images " << std::endl;
      return -1;
    }


    //-- Step 1: Detect the keypoints using SFT Detector
    //int minHessian = 400; //1400 for m1m2 3&4; 400 for 1&2; 450 for m3 m4

    // SiftFeatureDetector detector;
    Ptr<FeatureDetector> detector = FeatureDetector::create("FAST");

    detector->detect(img_object_org, keypoints_object);
    detector->detect(img_scene_org, keypoints_scene);

    //--STEP 1.5   create the boarder
    if(img_object_org.rows == img_scene_org.rows && img_object_org.cols == img_scene_org.cols) {
      for (int i = 0; i < keypoints_scene.size(); i++) {
        {
          keypoints_scene[i].pt.x = keypoints_scene[i].pt.x
          + img_object_org.cols * (0.5);
          keypoints_scene[i].pt.y = keypoints_scene[i].pt.y
          + img_object_org.rows * (0.5);
        }
      }
    } else {
        /*   Scene does nothing about feature points
         *  Feature points are already in enlarged corredinates
         */

       }

       for (int i = 0; i < keypoints_object.size(); i++) {
        {
          keypoints_object[i].pt.x = keypoints_object[i].pt.x
          + img_object_org.cols * (0.5);
          keypoints_object[i].pt.y = keypoints_object[i].pt.y
          + img_object_org.rows * (0.5);
        }
      }

      Size sz;
      img_object = Mat::zeros(img_object_org.rows * 2, img_object_org.cols * 2,
        CV_8UC3);

      if(img_object_org.rows == img_scene_org.rows && img_object_org.cols == img_scene_org.cols) {

        img_scene = Mat::zeros(img_object_org.rows * 2, img_object_org.cols * 2, CV_8UC3);

      } else {

        img_scene = Mat::zeros(img_scene_org.rows, img_scene_org.cols, CV_8UC3);

      }


      Mat ROI_obj = img_object(
        cv::Rect(img_object_org.cols * (0.5), img_object_org.rows * (0.5),
         img_object_org.cols, img_object_org.rows));

      Mat ROI_scene = img_scene(
        cv::Rect(img_object_org.cols * (0.5), img_object_org.rows * (0.5),
         img_object_org.cols, img_object_org.rows));

      img_object_org.copyTo(ROI_obj);
      if(img_object_org.rows == img_scene_org.rows && img_object_org.cols == img_scene_org.cols) {
        img_scene_org.copyTo(ROI_scene);
      } else {

        img_scene = img_scene_org;

      }
      ROI_obj.release();
      ROI_scene.release();

    //-- Step 2: Calculate descriptors (feature vectors)
      SiftDescriptorExtractor extractor;

      Mat descriptors_object, descriptors_scene;

      extractor.compute(img_object, keypoints_object, descriptors_object);
      extractor.compute(img_scene, keypoints_scene, descriptors_scene);

    //-- Step 3: Matching descriptor vectors using  matcher
      BFMatcher matcher(NORM_L2,false);
      std::vector<DMatch> matches;
      matcher.match(descriptors_object, descriptors_scene, matches);
      std::cout<<"SIFT key points number:"<<matches.size()<<"\n";


      double max_dist = 0;
      double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
      for (int i = 0; i < descriptors_object.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist)
          min_dist = dist;
        if (dist > max_dist)
          max_dist = dist;
      }

      printf("-- Max dist : %f \n", max_dist);
      printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

      for (int i = 0; i < descriptors_object.rows; i++) {
        if ((matches[i].distance < 8 * min_dist)) {
          good_matches.push_back(matches[i]);
        }
      }
      std::cout<<"initial macthing points: "<<good_matches.size()<<"\n\n###########";
    //show_match(img_object, img_scene, keypoints_object,  keypoints_scene, good_matches);
      descriptors_object.release();
      descriptors_scene.release();
    /*
        K-means and Dual-Homography
    */

        Scalar colorTab[] =     
        {
          Scalar(0, 0, 255),
          Scalar(0,255,0),
        };
        int K = 2; Mat points(good_matches.size(), 1, CV_32FC3), labels;
        Mat centers(K, 1, points.type());
        for (int i = 0; i < good_matches.size(); ++i)
        {
          float dx = keypoints_scene.at(good_matches[i].trainIdx).pt.x - keypoints_object.at(good_matches[i].queryIdx).pt.x;
          float dy = keypoints_scene.at(good_matches[i].trainIdx).pt.y - keypoints_object.at(good_matches[i].queryIdx).pt.y;
          points.at<Vec3f>(i,0)[0] = keypoints_scene.at(good_matches[i].trainIdx).pt.x;//(keypoints_scene_org.at(good_matches[i].trainIdx).pt.x - cc[0])*d[0]/(fc[0]);
          points.at<Vec3f>(i,0)[1] = keypoints_scene.at(good_matches[i].trainIdx).pt.y;//(keypoints_scene_org.at(good_matches[i].trainIdx).pt.y - cc[1])*d[0]/(fc[1]);
          points.at<Vec3f>(i,0)[2] = sqrt(dx*dx + dy*dy);
        }
        std::vector<DMatch> part_A_matches, part_B_matches;
        kmeans(points, K, labels,
          TermCriteria(  TermCriteria::EPS+TermCriteria::COUNT, 10, 3.0),
          5, KMEANS_RANDOM_CENTERS, centers); 
        Mat show;
        img_scene.copyTo(show);
        for( int i = 0; i < good_matches.size(); i++ )
        {
          int clusterIdx = labels.at<int>(i);
          Point2f ipt;
            // float depth = points.at<Vec3f>(i)[2];
            ipt.x =  points.at<Vec3f>(i)[0]; //points.at<Vec3f>(i)[0]*(fc[0]/depth) + cc[0] + img_object_org.cols * (0.5);
            ipt.y =  points.at<Vec3f>(i)[1]; //points.at<Vec3f>(i)[1]*(fc[1]/depth) + cc[1] + img_object_org.rows * (0.5);
            circle( show, ipt, 2, colorTab[clusterIdx], CV_FILLED, CV_AA );
            if (clusterIdx == 0)
            {
               /* code */
              part_A_matches.push_back(good_matches[i]);
            } else{
              part_B_matches.push_back(good_matches[i]);
            }
          }
          std::cout<<"k-means done!\n part_A size:"<<part_A_matches.size()<<"\n part_B size: "<<part_B_matches.size()<<"\n";

          for (int i = 0; i < 2; ++i)
          {
    /* code */
            Point2f ipt;
            ipt.x = centers.at<Vec3f>(i)[0]; //points.at<Vec3f>(i)[0]*(fc[0]/depth) + cc[0] + img_object_org.cols * (0.5);
            ipt.y = centers.at<Vec3f>(i)[1]; //points.at<Vec3f>(i)[1]*(fc[1]/depth) + cc[1] + img_object_org.rows * (0.5);
            circle( show, ipt, 5, Scalar(255,0,0), CV_FILLED, CV_AA );
          }
          // imshow("keypoints distrubtion",show);
    /*
     * Create Segments!
     * Double check what these parameters mean?
     */
    /*Next step:
     *
     * seg_A ppm --> Mat
     * seg_B ppm --> Mat
     *
     * making mask
     *
     * stitching background
     *
     * stitching object
     */
     int cnt = 0;
     float ratio = 1;
     int num = NNN;
     match_seg mid, part_A, part_B;
     mid = find_mid_point(good_matches, ratio, keypoints_object, keypoints_scene);
     part_A = find_mid_point(part_A_matches, ratio, keypoints_object, keypoints_scene);
     part_B = find_mid_point(part_B_matches, ratio, keypoints_object, keypoints_scene);
    // Mat Hl = findHomography(mid.obj, mid.mid, CV_RANSAC);
    // Mat Hr = findHomography(mid.scene, mid.mid, CV_RANSAC);
     // assert(good_matches.size() > 4);
     if (good_matches.size()<4)
     {
       /* code */
      return 0;
     }
     
     Mat total[3];
     try{
       total[0] = findHomography(mid.obj, mid.scene, CV_RANSAC);
       total[1] = findHomography(part_A.obj, part_A.scene, CV_RANSAC);
       total[2] = findHomography(part_B.obj, part_B.scene, CV_RANSAC);
     }catch (int e){
      std::cout<<"missing a frame due to lack of feature points < 4\n";
      good_matches.clear();
      keypoints_object.clear();
      keypoints_object_org.clear();
      keypoints_scene.clear();
      keypoints_scene_org.clear();
      img_object.release();
      img_scene.release();
      return 0;
    }

    Mat sqrtm[3];
    for (int i = 0; i < 3; ++i)
    {
  /* code */
      Eigen::MatrixXd X = Eigen::MatrixXd(total[i].rows,total[i].cols);
    //OpenCV -> Eigen 
      try{
        cv2eigen(total[i],X);
        Eigen::MatrixXd C = X.sqrt();
        Eigen::MatrixXd C2 = C.sqrt();
    //Eigen -> OpenCV
        eigen2cv(C2, sqrtm[i]);
      }catch (int i){
       std::cout<<"cannot solve square root of Homography matrix\n";

       good_matches.clear();
       keypoints_object.clear();
       keypoints_object_org.clear();
       keypoints_scene.clear();
       keypoints_scene_org.clear();
       img_object.release();
       img_scene.release();

       return 0;
     }
   }

   while(cnt < num) {
     Mat result;
        //show_match(result_A, result_B, keypoints_object,  keypoints_scene, good_matches);
        //result = Mat::zeros(result_A.rows,result_A.cols,CV_8UC3);
     if (part_A_matches.size() < 4 || part_B_matches.size() < 4 )
     {
           /* code */
      result = stitching_bg( mid, img_object,  img_scene, img_object, file_name_counter + cnt, sqrtm[0] , cnt, good_matches);

    }else{

      result = dual_stitching_bg(mid, part_A, part_B, img_object,  img_scene, img_object, file_name_counter + cnt, sqrtm[1], sqrtm[2] , cnt, good_matches, centers);
    }
        /*
         * start stitching!--> Object!
         */
         std::cout<<"start stitching!--> Object!\n";
         int seg_index = 0;
         char s[10];
         char d[50];
         Mat cropped_result;
         if(img_object_org.rows == img_scene_org.rows && img_object_org.cols == img_scene_org.cols) {
          std::sprintf(s,"./results/result: %d.ppm",cnt);
          std::sprintf(d,"./results/render: %d.ppm",file_name_counter  + cnt);
          cropped_result = result(cv::Rect(img_object_org.cols * (0.5), img_object_org.rows * (0.5),img_object_org.cols, img_object_org.rows));
           imwrite(d, cropped_result);//        waitKey(300);


         } else {
          std::sprintf(s,"./results/result: %d.ppm",frame_counter);
          imwrite(s, result);
#ifdef CROP
          std::sprintf(d,"./results/cropped_result: %d.ppm",frame_counter);
          cropped_result = result(cv::Rect(img_object_org.cols * (0.5), img_object_org.rows * (0.5),img_object_org.cols, img_object_org.rows));
#endif
        }
#ifdef CROP
        imwrite(d, cropped_result);//        waitKey(300);
#endif
        std::cout<<"--------->"<<file_name_counter<<"ratio: "<<ratio<<"\n\n";
        ratio = ratio - 1.0/(num-1);
        cnt++;
      }


    //exit(0);
    //free memeory:
      good_matches.clear();
      keypoints_object.clear();
      keypoints_object_org.clear();
      keypoints_scene.clear();
      keypoints_scene_org.clear();
      img_object.release();
      img_scene.release();

      return 0;

    }

/** @function readme */
    void readme() {

      std::cout << " Usage: ./ImageD <img1>(large) <img2> num_of_frames min_size_of_seg" << std::endl;
    }
