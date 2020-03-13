/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once
#ifndef LDSO_UNDISORT_H_
#define LDSO_UNDISORT_H_

#include <Eigen/Core>
#include "frontend/ImageAndExposure.h"
#include "NumTypes.h"
#include "MinimalImage.h"
#include "Settings.h"

// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>

namespace ldso {
    class PhotometricUndistorter {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PhotometricUndistorter(std::string file, std::string noiseImage, std::string vignetteImage, int w_, int h_);

        ~PhotometricUndistorter();

        // removes readout noise, and converts to irradiance.
        // affine normalizes values to 0 <= I < 256.
        // raw irradiance = a*I + b.
        // output will be written in [output].
        template<typename T>
        void processFrame(T *image_in, float exposure_time, float factor = 1);

        void unMapFloatImage(float *image);

        ImageAndExposure *output;

        float *getG() { if (!valid) return 0; else return G; };
    private:
        float G[256 * 256];
        int GDepth;
        float *vignetteMap;
        float *vignetteMapInv;
        int w, h;
        bool valid;
    };

    // 基类 多鱼眼组合全景
    class UndistortMultiFisheye {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual ~UndistortMultiFisheye();

        virtual void setParasPyramids() {};

        // virtual void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const = 0;

        virtual const char* getCameraModelType() const = 0;

        inline const Eigen::Vector2i getSize() const { return Eigen::Vector2i(w, h); };

        inline const Eigen::Vector2i getOriginalSize() const { return Eigen::Vector2i(wOrg, hOrg); };

        inline const int getcamNums() const { return camNums; }

        inline bool isValid() { return valid; };

        template<typename T>
        ImageAndExposure *
        undistortFisheye(const MinimalImage<T> *image_raw, float exposure = 0, double timestamp = 0, float factor = 1) ;

        static UndistortMultiFisheye *
        getUndistorterForFile(std::string configFilename, int camNum, std::string gammaFilename, std::string vignetteFilename);

        void loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage);

        void CalculateRotationMatFromEulerAngle(double Rx, double Ry, double Rz, double * R);

        void CalculateRotationMatFromCayleyParameter(double c1, double c2, double c3, double * R);


        bool setwh(MinimalImageB *image_raw , std::string gammaFile, std::string vignetteFile); 

        float getBGradOnly(float color);

        float getBInvGradOnly(float color);

        // for multi-fisheye

        //将鱼眼图像上的像素点投射到纠正后图像上
        virtual bool LadybugRectifyImage(int CameraNum, double Pixalx, double Pixaly,
                double* RectifiedPixalx, double* RectifiedPixaly, int level = 0) {};

        //将纠正后图像上的像素点投射到鱼眼图像上
        virtual bool LadybugUnRectifyImage(int CameraNum, double Pixalx, double Pixaly,
                                   double* DistortedPixalx, double* DistortedPixaly, int level = 0) {};

        //将鱼眼图像的像素坐标投影到全景球面上，Radius是自定义的球半径(单位：m), 默认为20m
        bool LadybugProjectFishEyePtToSphere(int CameraNum, 
                double FishEyePixalx, double FishEyePixaly, 
                double* SphereX, double* SphereY, double* SphereZ, int level = 0);

        //反投影，将球面上的坐标投影到鱼眼相机上，CamerNum为输出值，程序自动判断这个点落在哪个相机上
        bool LadybugReprojectSpherePtToFishEyeImg(double SphereX, double SphereY, double SphereZ,
                        int* CameraNum, double* FishEyePixalx, double* FishEyePixaly, int level = 0);

        //将纠正后图像的像素坐标投影到全景球面上，Radius是自定义的球半径(单位：m), 默认为20m
        bool LadybugProjectRectifyPtToSphere(int CameraNum, 
                double RectifiedPixalx, double RectifiedPixaly, 
                double* SphereX, double* SphereY, double* SphereZ, int level = 0);

        //反投影，将球面上的坐标投影到纠正后图像上，CamerNum为输出值，程序自动判断这个点落在哪个相机上
        bool LadybugReprojectSpherePtToRectify(double SphereX, double SphereY, double SphereZ,
                        int* CameraNum, double* RectifiedPixalx, double* RectifiedPixaly, int level = 0);

        //反投影，将球面上的坐标投影到纠正后图像上，选择所要投影的相机编号，主要用与极线计算
        bool LadybugReprojectSpherePtToRectifyfixNum(double SphereX, double SphereY, double SphereZ,
                    int CameraNum, double* RectifiedPixalx, double* RectifiedPixaly, int level);
        inline Eigen::Matrix<float, 2, 3> computedStoR(float SphereX, float SphereY, float SphereZ, int n, int level = 0)
        {
            Eigen::Matrix2f dIL;
            dIL << 0,-1,1,0;

            Eigen::Matrix<float, 2, 3> duXs;

            Eigen::Matrix<float, 3, 1> RaySphere;
            RaySphere(0, 0) = SphereX;
            RaySphere(1, 0) = SphereY;
            RaySphere(2, 0) = SphereZ;

            Eigen::Matrix<float, 3, 3> RotationMat = (mvExParas[n].block<3, 3>(0, 0)).cast<float>();
            Eigen::Matrix<float, 3, 1> CamPos = mvExParas[n].block<3, 1>(0, 3).cast<float>();
            Eigen::Matrix<float, 3, 1> RayCoor = RaySphere - CamPos;
            Eigen::Matrix<float, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
            float x = RectifiedImgPtCoor(0, 0);
            float y = RectifiedImgPtCoor(1, 0);
            float z = RectifiedImgPtCoor(2, 0);

            float x0r, y0r;
            float f = mvInnerParasPR[level][n](0, 13);
            x0r = mvInnerParasPR[level][n](0, 14);
            y0r = mvInnerParasPR[level][n](0, 15);

            duXs << f/z, 0, -f/(z*z)*x,
                    0, f/z, -f/(z*z)*y;
            
            Eigen::Matrix<float, 2, 3> dSR;
            dSR = dIL * duXs * RotationMat.inverse();

            return dSR;
        }

        inline void computedXs(float rho1, Mat33f R, Vec3f t, float X, float Y, float Z, 
                    Eigen::Matrix<float, 3, 6> * dXsdpose, Eigen::Matrix<float, 3, 1> * dXsdrho1)
        {
            Vec3f Xc;
            Xc = R * Vec3f(X/Z, Y/Z, 1) * rho1 + t;
            float x,y,z;
            x = Xc(0);
            y = Xc(1);
            z = Xc(2);
            float l2 = x * x + y * y + z * z;
            float l23 = l2 * sqrt(l2);

            Eigen::Matrix<float, 3, 3> dXsdXc;
            dXsdXc << -mSphereRadius  *(y * y + z * z) / l23, x * y * mSphereRadius / l23, x * z * mSphereRadius / l23,
                    x * y * mSphereRadius / l23, -mSphereRadius * (x * x + z * z) / l23, y * z * mSphereRadius / l23,
                    x * z * mSphereRadius / l23, y * z * mSphereRadius / l23, -mSphereRadius * (x * x + y * y) / l23;

            Eigen::Matrix<float, 3, 6> dXcdpose;
            dXcdpose << 1, 0, 0, 0, -z, y,
                        0, 1, 0, z, 0, -x,
                        0, 0, 1, -y, x, 0;
            
            Eigen::Matrix<float, 3, 1> dXcdroh1;
            dXcdroh1 = - R * Vec3f(X/Z, Y/Z,1) / (rho1 * rho1);

            *dXsdpose = dXsdXc * dXcdpose;
            *dXsdrho1 = dXsdXc * dXcdroh1;
        }

        //输入图像的大小
        inline void InputImgSize(int width, int height);

        //输入全景球半径
        void InputSphereRadius(int Radius);

        //输出纠正后图像的像素中心
        virtual void GetRectifiedImgCenter(int CameraNum, double* x0, double *y0, int level = 0) {};

        //输出纠正后图像的焦距
        virtual double GetRectifiedImgFocalLength(int CameraNum, int level = 0) {};

        //输出指定鱼眼相机的正算内参
        //cv::Mat GetUnitCameraInnPara(int CameraNum);

        //输出全景球半径
        float GetSphereRadius() const;

        //输出图像的顶端投射到全景球面后的Z坐标
        float GetUpperLevelofSphere();

        //设置当前使用的相机号
        void SetCurrentCamNum(int CamNUm);

        //输出当前使用的相机号
        int GetCurrentCamNmu();



        PhotometricUndistorter *photometricUndist;

        // multi-fisheye
        int camNums;    // 相机总数
        int CameraNum;  // 当前使用相机号
        std::vector<Eigen::MatrixXd> mvInnerParas;
        std::vector<Eigen::MatrixXd> mvInvInnerParas;  
        std::vector<Eigen::MatrixXd> mvExParas;

        //  内参金字塔 pyramids
        std::vector<Eigen::MatrixXd> mvInnerParasPR[PYR_LEVELS];
        std::vector<Eigen::MatrixXd> mvInvInnerParasPR[PYR_LEVELS];

        int wPR[PYR_LEVELS],hPR[PYR_LEVELS]; 

        float wM3G;     // ??
        float hM3G;

    protected:
        int w, h, wOrg, hOrg, wUp, hUp;
        int upsampleUndistFactor;
        bool valid;
        bool passthrough;

        float *remapX;
        float *remapY;

        // gamma function, by default from 0 to 255
        float B[256];
        float Binv[256]; 

        void applyBlurNoise(float *img) const;

        float mSphereRadius=20.0;


        //Ladybug使用其单独的像素坐标系
        //其像素坐标系为我们习惯的图像右上角为原点，向下为X轴，向坐为Y轴(相当于我们习惯的图像逆时针转90°)
        //以下两个函数用于我们习惯的图像像素坐标和Ladybug的像素坐标之间的相互转换
        Eigen::Vector2d ImgPixCoor2LadybugPixCoor(Eigen::Vector2d pt, int ImgWidth, int ImgHeight);
        Eigen::Vector2d LadybugPixCoor2ImgPixCoor(Eigen::Vector2d pt, int ImgWidth, int Height);
    };

    // 等距模型
    class UndistortMFEquidistant : public UndistortMultiFisheye 
    {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        UndistortMFEquidistant(std::string configFileName, int camNum, bool noprefix);

        ~UndistortMFEquidistant() {};

        inline const char* getCameraModelType() const override { return "Equidistant"; }

        void setParasPyramids() override;


        //将鱼眼图像上的像素点投射到纠正后图像上
        bool LadybugRectifyImage(int CameraNum, double Pixalx, double Pixaly,
                double* RectifiedPixalx, double* RectifiedPixaly, int level =0) override;

        //将纠正后图像上的像素点投射到鱼眼图像上
        bool LadybugUnRectifyImage(int CameraNum, double Pixalx, double Pixaly,
                                   double* DistortedPixalx, double* DistortedPixaly, int level =0) override;

         //输出纠正后图像的像素中心
        void GetRectifiedImgCenter(int CameraNum, double* x0, double *y0, int level = 0) override;

        //输出纠正后图像的焦距
        double GetRectifiedImgFocalLength(int CameraNum, int level = 0) override;

    };
    // 多项式模型
    class UndistortMFPoly : public UndistortMultiFisheye 
    {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        UndistortMFPoly(std::string configFileName, int camNum, bool noprefix);

        ~UndistortMFPoly() {};

        inline const char* getCameraModelType() const override { return "Poly"; }

    };
}

#endif // LDSO_UNDISORT_H_
