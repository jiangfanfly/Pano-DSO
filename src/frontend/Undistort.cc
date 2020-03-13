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




#include <sstream>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <iterator>

#include "Settings.h"
#include "internal/GlobalFuncs.h"
#include "frontend/Undistort.h"
#include "frontend/ImageRW.h"

using namespace ldso::internal;

namespace ldso {

// ===================PhotometricUndistorter==========================

    PhotometricUndistorter::PhotometricUndistorter(
            std::string file,
            std::string noiseImage,
            std::string vignetteImage,
            int w_, int h_) {
        valid = false;
        vignetteMap = 0;
        vignetteMapInv = 0;
        w = w_;
        h = h_;
        output = new ImageAndExposure(w, h);
        if (file == "" || vignetteImage == "") {
            printf("NO PHOTOMETRIC Calibration!\n");
        }


        // read G.
        std::ifstream f(file.c_str());
        printf("Reading Photometric Calibration from file %s\n", file.c_str());
        if (!f.good()) {
            printf("PhotometricUndistorter: Could not open file!\n");
            return;
        }


        {
            std::string line;
            std::getline(f, line);
            std::istringstream l1i(line);
            std::vector<float> Gvec = std::vector<float>(std::istream_iterator<float>(l1i),
                                                         std::istream_iterator<float>());


            GDepth = Gvec.size();

            if (GDepth < 256) {
                printf("PhotometricUndistorter: invalid format! got %d entries in first line, expected at least 256!\n",
                       (int) Gvec.size());
                return;
            }


            for (int i = 0; i < GDepth; i++) G[i] = Gvec[i];

            for (int i = 0; i < GDepth - 1; i++) {
                if (G[i + 1] <= G[i]) {
                    printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
                    return;
                }
            }

            float min = G[0];
            float max = G[GDepth - 1];
            for (int i = 0; i < GDepth; i++)
                G[i] = 255.0 * (G[i] - min) / (max - min);            // make it to 0..255 => 0..255.
        }

        if (setting_photometricCalibration == 0) {
            for (int i = 0; i < GDepth; i++) G[i] = 255.0f * i / (float) (GDepth - 1);
        }


        printf("Reading Vignette Image from %s\n", vignetteImage.c_str());
        MinimalImage<unsigned short> *vm16 = IOWrap::readImageBW_16U(vignetteImage.c_str());
        MinimalImageB *vm8 = IOWrap::readImageBW_8U(vignetteImage.c_str());
        vignetteMap = new float[w * h];
        vignetteMapInv = new float[w * h];

        if (vm16 != 0) {
            if (vm16->w != w || vm16->h != h) {
                printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n",
                       vm16->w, vm16->h, w, h);
                if (vm16 != 0) delete vm16;
                if (vm8 != 0) delete vm8;
                return;
            }

            float maxV = 0;
            for (int i = 0; i < w * h; i++)
                if (vm16->at(i) > maxV) maxV = vm16->at(i);

            for (int i = 0; i < w * h; i++)
                vignetteMap[i] = vm16->at(i) / maxV;
        } else if (vm8 != 0) {
            if (vm8->w != w || vm8->h != h) {
                printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n",
                       vm8->w, vm8->h, w, h);
                if (vm16 != 0) delete vm16;
                if (vm8 != 0) delete vm8;
                return;
            }

            float maxV = 0;
            for (int i = 0; i < w * h; i++)
                if (vm8->at(i) > maxV) maxV = vm8->at(i);

            for (int i = 0; i < w * h; i++)
                vignetteMap[i] = vm8->at(i) / maxV;
        } else {
            printf("PhotometricUndistorter: Invalid vignette image\n");
            if (vm16 != 0) delete vm16;
            if (vm8 != 0) delete vm8;
            return;
        }

        if (vm16 != 0) delete vm16;
        if (vm8 != 0) delete vm8;


        for (int i = 0; i < w * h; i++)
            vignetteMapInv[i] = 1.0f / vignetteMap[i];


        printf("Successfully read photometric calibration!\n");
        valid = true;
    }

    PhotometricUndistorter::~PhotometricUndistorter() {
        if (vignetteMap != 0) delete[] vignetteMap;
        if (vignetteMapInv != 0) delete[] vignetteMapInv;
        delete output;
    }


    void PhotometricUndistorter::unMapFloatImage(float *image) {
        int wh = w * h;
        for (int i = 0; i < wh; i++) {
            float BinvC;
            float color = image[i];

            if (color < 1e-3)
                BinvC = 0.0f;
            else if (color > GDepth - 1.01f)
                BinvC = GDepth - 1.1;
            else {
                int c = color;
                float a = color - c;
                BinvC = G[c] * (1 - a) + G[c + 1] * a;
            }

            float val = BinvC;
            if (val < 0) val = 0;
            image[i] = val;
        }
    }

    template<typename T>
    void PhotometricUndistorter::processFrame(T *image_in, float exposure_time, float factor) {
        int wh = w * h;
        float *data = output->image;
        assert(output->w == w && output->h == h);
        assert(data != 0);


        if (!valid || exposure_time <= 0 ||
            setting_photometricCalibration == 0) // disable full photometric calibration.
        {
            for (int i = 0; i < wh; i++) {
                data[i] = factor * image_in[i];
            }
            output->exposure_time = exposure_time;
            output->timestamp = 0;
        } else {
            for (int i = 0; i < wh; i++) {
                data[i] = G[image_in[i]];
            }

            if (setting_photometricCalibration == 2) {
                for (int i = 0; i < wh; i++) {
                    if (!std::isinf(vignetteMapInv[i])) {
                        data[i] *= vignetteMapInv[i];
                    } else {
                        data[i] *= vignetteMapInv[i];
                    }
                }
            }

            output->exposure_time = exposure_time;
            output->timestamp = 0;
        }

        if (!setting_useExposure)
            output->exposure_time = 1;

    }

    template void
    PhotometricUndistorter::processFrame<unsigned char>(unsigned char *image_in, float exposure_time, float factor);

    template void
    PhotometricUndistorter::processFrame<unsigned short>(unsigned short *image_in, float exposure_time, float factor);


// =========================UndistortMultiFisheye=================================16
    UndistortMultiFisheye::~UndistortMultiFisheye()
    {
        if (remapX != 0) delete[] remapX;
        if (remapY != 0) delete[] remapY;
    }

    UndistortMultiFisheye *UndistortMultiFisheye::getUndistorterForFile(std::string configFilename, int camNum, std::string gammaFilename, std::string vignetteFilename)
    {

        UndistortMultiFisheye *u;

        //Use Equidistant model while the number of files is 3.
        if (true) {
            printf("found Equidistant fisheye model, building rectifier.\n");
            u = new UndistortMFEquidistant(configFilename, camNum, true);
            if (!u->isValid()) {
                delete u;
                return 0;
            }
        }
        // Use Poly model
        else 
        {
            printf("found Poly fisheye model, building rectifier.\n");
            u = new UndistortMFPoly(configFilename, camNum, true);
            if (!u->isValid()) {
                delete u;
                return 0;
            }
        }
        return u;

    }

    void UndistortMultiFisheye::loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage) {
        photometricUndist = new PhotometricUndistorter(file, noiseImage, vignetteImage, getOriginalSize()[0],
                                                       getOriginalSize()[1]);
    }

    template<typename T>
    ImageAndExposure *
    UndistortMultiFisheye::undistortFisheye(const MinimalImage<T> *image_raw, float exposure, double timestamp, float factor)  
    {
        if (image_raw->w != wOrg || image_raw->h != hOrg) {
            printf("Undistort::undistort: wrong image size (%d %d instead of %d %d) \n", image_raw->w, image_raw->h, w,
                   h);
            exit(1);
        }

        photometricUndist->processFrame<T>(image_raw->data, exposure, factor);
        ImageAndExposure *result = new ImageAndExposure(w, h, timestamp);
        photometricUndist->output->copyMetaTo(*result);

        if (!passthrough) {
            float *out_data = result->image;
            float *in_data = photometricUndist->output->image;

            float *noiseMapX = 0;
            float *noiseMapY = 0;
            if (benchmark_varNoise > 0) {
                int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
                noiseMapX = new float[numnoise];
                noiseMapY = new float[numnoise];
                memset(noiseMapX, 0, sizeof(float) * numnoise);
                memset(noiseMapY, 0, sizeof(float) * numnoise);

                for (int i = 0; i < numnoise; i++) {
                    noiseMapX[i] = 2 * benchmark_varNoise * (rand() / (float) RAND_MAX - 0.5f);
                    noiseMapY[i] = 2 * benchmark_varNoise * (rand() / (float) RAND_MAX - 0.5f);
                }
            }


            for (int idx = w * h - 1; idx >= 0; idx--) {
                // get interp. values
                float xx = remapX[idx];
                float yy = remapY[idx];

                if (benchmark_varNoise > 0) {
                    float deltax = getInterpolatedElement11BiCub(noiseMapX,
                                                                 4 + (xx / (float) wOrg) * benchmark_noiseGridsize,
                                                                 4 + (yy / (float) hOrg) * benchmark_noiseGridsize,
                                                                 benchmark_noiseGridsize + 8);
                    float deltay = getInterpolatedElement11BiCub(noiseMapY,
                                                                 4 + (xx / (float) wOrg) * benchmark_noiseGridsize,
                                                                 4 + (yy / (float) hOrg) * benchmark_noiseGridsize,
                                                                 benchmark_noiseGridsize + 8);
                    float x = idx % w + deltax;
                    float y = idx / w + deltay;
                    if (x < 0.01) x = 0.01;
                    if (y < 0.01) y = 0.01;
                    if (x > w - 1.01) x = w - 1.01;
                    if (y > h - 1.01) y = h - 1.01;

                    xx = getInterpolatedElement(remapX, x, y, w);
                    yy = getInterpolatedElement(remapY, x, y, w);
                }


                if (xx < 0)
                    out_data[idx] = 0;
                else {
                    // get integer and rational parts
                    int xxi = xx;
                    int yyi = yy;
                    xx -= xxi;
                    yy -= yyi;
                    float xxyy = xx * yy;

                    // get array base pointer
                    const float *src = in_data + xxi + yyi * wOrg;

                    // interpolate (bilinear)
                    out_data[idx] = xxyy * src[1 + wOrg]
                                    + (yy - xxyy) * src[wOrg]
                                    + (xx - xxyy) * src[1]
                                    + (1 - xx - yy + xxyy) * src[0];
                }
            }

            if (benchmark_varNoise > 0) {
                delete[] noiseMapX;
                delete[] noiseMapY;
            }

        } else {
            memcpy(result->image, photometricUndist->output->image, sizeof(float) * w * h);
        }

        applyBlurNoise(result->image);

        return result;
    }

    template ImageAndExposure *
    UndistortMultiFisheye::undistortFisheye<unsigned char>(const MinimalImage<unsigned char> *image_raw, float exposure, double timestamp,
                                        float factor);

    template ImageAndExposure *
    UndistortMultiFisheye::undistortFisheye<unsigned short>(const MinimalImage<unsigned short> *image_raw, float exposure,
                                         double timestamp, float factor);

    void UndistortMultiFisheye::CalculateRotationMatFromEulerAngle(double Rx, double Ry, double Rz, double * R)
    {
        double cRx, cRy, cRz, sRx, sRy, sRz;
        cRx = cos(Rx); cRy = cos(Ry); cRz = cos(Rz);
        sRx = sin(Rx); sRy = sin(Ry); sRz = sin(Rz);
        R[0] = cRz * cRy;
        R[1] = cRz * sRy * sRx - sRz * cRx;
        R[2] = cRz * sRy * cRx + sRz * sRx;
        R[3] = sRz * cRy;
        R[4] = sRz * sRy* sRx + cRz * cRx;
        R[5] = sRz * sRy* cRx - cRz * sRx;
        R[6] = -sRy;
        R[7] = cRy * sRx;
        R[8] = cRy * cRx;
    }

    void UndistortMultiFisheye::CalculateRotationMatFromCayleyParameter(double c1, double c2, double c3, double * R)
    {
        double c1sqr = c1 * c1;
        double c2sqr = c2 * c2;
        double c3sqr = c3 * c3;

        double scale = 1 + c1sqr + c2sqr + c3sqr;
        double invScale = 1.0 / scale;

        double r1 = 1 + c1sqr - c2sqr - c3sqr;
        double r2 = 2 * (c1*c2 - c3);
        double r3 = 2 * (c1*c3 + c2);
        double r4 = 2 * (c1*c2 + c3);
        double r5 = 1 - c1sqr + c2sqr - c3sqr;
        double r6 = 2 * (c2*c3 - c1);
        double r7 = 2 * (c1*c3 - c2);
        double r8 = 2 * (c2*c3 + c1);
        double r9 = 1 - c1sqr - c2sqr + c3sqr;

        R[0] = invScale * r1;
        R[1] = invScale * r2;
        R[2] = invScale * r3;
        R[3] = invScale * r4;
        R[4] = invScale * r5;
        R[5] = invScale * r6;
        R[6] = invScale * r7;
        R[7] = invScale * r8;
        R[8] = invScale * r9;


    }

    void UndistortMultiFisheye::applyBlurNoise(float *img) const {
        if (benchmark_varBlurNoise == 0) return;

        int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
        float *noiseMapX = new float[numnoise];
        float *noiseMapY = new float[numnoise];
        float *blutTmp = new float[w * h];

        if (benchmark_varBlurNoise > 0) {
            for (int i = 0; i < numnoise; i++) {
                noiseMapX[i] = benchmark_varBlurNoise * (rand() / (float) RAND_MAX);
                noiseMapY[i] = benchmark_varBlurNoise * (rand() / (float) RAND_MAX);
            }
        }


        float gaussMap[1000];
        for (int i = 0; i < 1000; i++)
            gaussMap[i] = expf((float) (-i * i / (100.0 * 100.0)));

        // x-blur.
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                float xBlur = getInterpolatedElement11BiCub(noiseMapX,
                                                            4 + (x / (float) w) * benchmark_noiseGridsize,
                                                            4 + (y / (float) h) * benchmark_noiseGridsize,
                                                            benchmark_noiseGridsize + 8);

                if (xBlur < 0.01) xBlur = 0.01;


                int kernelSize = 1 + (int) (1.0f + xBlur * 1.5);
                float sumW = 0;
                float sumCW = 0;
                for (int dx = 0; dx <= kernelSize; dx++) {
                    int gmid = 100.0f * dx / xBlur + 0.5f;
                    if (gmid > 900) gmid = 900;
                    float gw = gaussMap[gmid];

                    if (x + dx > 0 && x + dx < w) {
                        sumW += gw;
                        sumCW += gw * img[x + dx + y * this->w];
                    }

                    if (x - dx > 0 && x - dx < w && dx != 0) {
                        sumW += gw;
                        sumCW += gw * img[x - dx + y * this->w];
                    }
                }

                blutTmp[x + y * this->w] = sumCW / sumW;
            }

        // y-blur.
        for (int x = 0; x < w; x++)
            for (int y = 0; y < h; y++) {
                float yBlur = getInterpolatedElement11BiCub(noiseMapY,
                                                            4 + (x / (float) w) * benchmark_noiseGridsize,
                                                            4 + (y / (float) h) * benchmark_noiseGridsize,
                                                            benchmark_noiseGridsize + 8);

                if (yBlur < 0.01) yBlur = 0.01;

                int kernelSize = 1 + (int) (0.9f + yBlur * 2.5);
                float sumW = 0;
                float sumCW = 0;
                for (int dy = 0; dy <= kernelSize; dy++) {
                    int gmid = 100.0f * dy / yBlur + 0.5f;
                    if (gmid > 900) gmid = 900;
                    float gw = gaussMap[gmid];

                    if (y + dy > 0 && y + dy < h) {
                        sumW += gw;
                        sumCW += gw * blutTmp[x + (y + dy) * this->w];
                    }

                    if (y - dy > 0 && y - dy < h && dy != 0) {
                        sumW += gw;
                        sumCW += gw * blutTmp[x + (y - dy) * this->w];
                    }
                }
                img[x + y * this->w] = sumCW / sumW;
            }


        delete[] noiseMapX;
        delete[] noiseMapY;
    }

    float UndistortMultiFisheye::getBGradOnly(float color) {
                int c = color + 0.5f;
                if (c < 5) {
                    c = 5;
                }
                if (c > 250) {
                    c = 250;
                }
                return B[c + 1] - B[c];
            }

    float UndistortMultiFisheye::getBInvGradOnly(float color) {
                int c = color + 0.5f;
                if (c < 5) {
                    c = 5;
                }
                if (c > 250) {
                    c = 250;
                }
                return Binv[c + 1] - Binv[c];
            }


    bool UndistortMultiFisheye::setwh(MinimalImageB *image_raw , std::string gammaFile, std::string vignetteFile) 
    { 
        w = wOrg = image_raw->w; 
        h = hOrg = image_raw->h;

        if(w == image_raw->w)
        {
            loadPhotometricCalibration(gammaFile, "", vignetteFile);
            return true;
        }
        else
            return false;
            
    }

    Eigen::Vector2d UndistortMultiFisheye::ImgPixCoor2LadybugPixCoor(Eigen::Vector2d pt, int ImgWidth, int ImgHeight)
    {
        double xImg, yImg, xLB, yLB;
        xImg = pt(0);
        yImg = pt(1);
        xLB = yImg;
        yLB = ImgWidth - xImg - 1;
        return Eigen::Vector2d(xLB, yLB);
    }

    Eigen::Vector2d UndistortMultiFisheye::LadybugPixCoor2ImgPixCoor(Eigen::Vector2d pt, int ImgWidth, int Height)
    {
        double xImg, yImg, xLB, yLB;
        xLB = pt(0);
        yLB = pt(1);
        xImg = ImgWidth - yLB - 1;
        yImg = xLB;
        return Eigen::Vector2d(xImg, yImg);
    }

    float UndistortMultiFisheye::GetSphereRadius() const
    {
        return mSphereRadius;
    }

    //将鱼眼图像的像素坐标投影到全景球面上，Radius是自定义的球半径(单位：m)
    bool UndistortMultiFisheye::LadybugProjectFishEyePtToSphere(int CameraNum, 
                double FishEyePixalx, double FishEyePixaly, 
                double* SphereX, double* SphereY, double* SphereZ, int level)
    {
        *SphereX = 0;
        *SphereX = 0;
        *SphereX = 0;

        double RectifiedPixalx, RectifiedPixaly;
        //从鱼眼图像像素坐标转换到纠正像素坐标
        LadybugRectifyImage(CameraNum, FishEyePixalx, FishEyePixaly, &RectifiedPixalx, &RectifiedPixaly, level);

        double Xs, Ys, Zs;
        LadybugProjectRectifyPtToSphere(CameraNum, RectifiedPixalx, RectifiedPixaly, &Xs, &Ys, &Zs, level);
        *SphereX = Xs;
        *SphereY = Ys;
        *SphereZ = Zs;
       
    }

    //反投影，将球面上的坐标投影到鱼眼相机上，CamerNum为输出值，程序自动判断这个点落在哪个相机上
    bool UndistortMultiFisheye::LadybugReprojectSpherePtToFishEyeImg(double SphereX, double SphereY, double SphereZ,
                        int* CameraNum, double* FishEyePixalx, double* FishEyePixaly, int level)
    {
        double RectPixalx, RectPixaly;
        int Cam;
        LadybugReprojectSpherePtToRectify(SphereX, SphereY, SphereZ, &Cam, &RectPixalx, &RectPixaly, level);
        
        double DistortedPixalx, DistortedPixaly;
        LadybugUnRectifyImage(Cam, RectPixalx, RectPixaly, &DistortedPixalx, &DistortedPixaly, level);
        *CameraNum = Cam;
        *FishEyePixalx = DistortedPixalx;
        *FishEyePixaly = DistortedPixaly;

        return true;
    }

    //将纠正后图像的像素坐标投影到全景球面上，Radius是自定义的球半径(单位：m), 默认为20m
    bool UndistortMultiFisheye::LadybugProjectRectifyPtToSphere(int CameraNum, 
            double RectifiedPixalx, double RectifiedPixaly, 
            double* SphereX, double* SphereY, double* SphereZ, int level)
    {
        //纠正像素坐标换算到Ladybug的图像坐标系下
        Eigen::Vector2d RectifiedPixalCoor;
        RectifiedPixalCoor=ImgPixCoor2LadybugPixCoor(Eigen::Vector2d(RectifiedPixalx, RectifiedPixaly), wPR[level] ,hPR[level]);
        RectifiedPixalx=RectifiedPixalCoor(0);
        RectifiedPixaly=RectifiedPixalCoor(1);

        //获取当前鱼眼相机的内外方位元素
        Eigen::MatrixXd CurrentTransMat = mvExParas[CameraNum];
        double FocalLength = GetRectifiedImgFocalLength(CameraNum, level);
        double x0rectified, y0rectified;
        GetRectifiedImgCenter(CameraNum, &x0rectified, &y0rectified, level);
        // //得到的x0，y0是我们习惯的竖着的图像，在Ladybug框架下计算需要换算成Laydbug框架下的像素中心
        // Eigen::Vector2d ImgCenter = ImgPixCoor2LadybugPixCoor(Eigen::Vector2d(x0rectified,y0rectified), wPR[level], hPR[level]);
        // x0rectified=ImgCenter(0);
        // y0rectified=ImgCenter(1);


        //组成此点在此相机坐标系下的坐标
        Eigen::Matrix<double, 3, 1> RectifiedImgPtCoor;
        RectifiedImgPtCoor(0, 0) = RectifiedPixalx - x0rectified;
        RectifiedImgPtCoor(1, 0) = RectifiedPixaly - y0rectified;
        RectifiedImgPtCoor(2, 0) = FocalLength;

        //获得当前相机的旋转与平移量
        Eigen::Matrix<double, 3, 3> RotationMat = CurrentTransMat.block<3, 3>(0, 0);
        double Tx = CurrentTransMat(0, 3);
        double Ty = CurrentTransMat(1, 3);
        double Tz = CurrentTransMat(2, 3);

        //此点乘以旋转矩阵以计算出其在Ladybug坐标系下的坐标，即在球面坐标系下的光线向量
        Eigen::Matrix<double, 3, 1> RayCoor = RotationMat * RectifiedImgPtCoor;
        double CX = RayCoor(0, 0);
        double CY = RayCoor(1, 0);
        double CZ = RayCoor(2, 0);

        //光线与球面交会，球面半径为输入的Radius   (kCx)^2 + (kCy)^2 + (kCz)^2 - 20 = 0   一元二次方程 求根公式 
        double a = CX * CX + CY * CY + CZ * CZ;
        double b = 2 * (CX * Tx + CY * Ty + CZ * Tz);
        double c = Tx * Tx + Ty * Ty + Tz * Tz - mSphereRadius * mSphereRadius;
        double b4ac = sqrt(b*b - 4 * a*c);
        double k = (-b + b4ac) / (2 * a);

        //其在球面坐标系的坐标=相机投影中心的位置向量+光线向量
        *SphereX = k * CX + Tx;
        *SphereY = k * CY + Ty;
        *SphereZ = k * CZ + Tz;

        if (SphereX != 0 || SphereY != 0 || SphereY != 0)
        {
            return true;
        }
        else
        {
            std::cout << "鱼眼图像转球面坐标时出问题了" << std::endl;
            return false;
        }
    }

    //反投影，将球面上的坐标投影到纠正后图像上，CamerNum为输出值，程序自动判断这个点落在哪个相机上
    bool UndistortMultiFisheye::LadybugReprojectSpherePtToRectify(double SphereX, double SphereY, double SphereZ,
                    int* CameraNum, double* RectifiedPixalx, double* RectifiedPixaly, int level)
    {
        int Cam;
        Eigen::Matrix<double, 4, 4> T;
        Eigen::Matrix<double, 3, 1> RaySphere;
        RaySphere(0, 0) = SphereX;
        RaySphere(1, 0) = SphereY;
        RaySphere(2, 0) = SphereZ;
        double minCosAngle = 0.0;
        for (int i = 0; i < camNums; i++)
        {
            Eigen::Matrix<double, 3, 1> CamPose=mvExParas[i].block<3, 1>(0, 3);
            double cosAngle = double(RaySphere.dot(CamPose)) / (RaySphere.norm() * CamPose.norm());
            if (cosAngle>0 && cosAngle <=1 && cosAngle>minCosAngle)
            {
                minCosAngle = cosAngle;
                T = mvExParas[i];
                Cam = i;
            }
        }
        // if (T.value())
        //     return false;

        Eigen::Matrix<double, 3, 3> RotationMat = T.block<3, 3>(0, 0);
        Eigen::Matrix<double, 3, 1> CamPos = T.block<3, 1>(0, 3);
        Eigen::Matrix<double, 3, 1> RayCoor = RaySphere - CamPos;
        Eigen::Matrix<double, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
        double x = RectifiedImgPtCoor(0, 0);
        double y = RectifiedImgPtCoor(1, 0);
        double z = RectifiedImgPtCoor(2, 0);

        double FocalLength = GetRectifiedImgFocalLength(Cam, level);
        double x0rectified, y0rectified;
        GetRectifiedImgCenter(Cam, &x0rectified, &y0rectified, level);

        double RectPixalx = FocalLength * (x / z) + x0rectified;
        double RectPixaly = FocalLength * (y / z) + y0rectified;

        Eigen::Vector2d RectPixalCoor = LadybugPixCoor2ImgPixCoor(Eigen::Vector2d(RectPixalx, RectPixaly), wPR[level], hPR[level]);
        *RectifiedPixalx = RectPixalCoor(0);
        *RectifiedPixaly = RectPixalCoor(1);
        *CameraNum = Cam;

        return true;
    }

    //反投影，将球面上的坐标投影到纠正后图像上，选择所要投影的相机编号，主要用与极线计算
    bool UndistortMultiFisheye::LadybugReprojectSpherePtToRectifyfixNum(double SphereX, double SphereY, double SphereZ,
                    int CameraNum, double* RectifiedPixalx, double* RectifiedPixaly, int level)
    {
        Eigen::Matrix<double, 4, 4> T;
        Eigen::Matrix<double, 3, 1> RaySphere;
        RaySphere(0, 0) = SphereX;
        RaySphere(1, 0) = SphereY;
        RaySphere(2, 0) = SphereZ;

        T = mvExParas[CameraNum];


        Eigen::Matrix<double, 3, 3> RotationMat = T.block<3, 3>(0, 0);
        Eigen::Matrix<double, 3, 1> CamPos = T.block<3, 1>(0, 3);
        Eigen::Matrix<double, 3, 1> RayCoor = RaySphere - CamPos;
        Eigen::Matrix<double, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
        double x = RectifiedImgPtCoor(0, 0);
        double y = RectifiedImgPtCoor(1, 0);
        double z = RectifiedImgPtCoor(2, 0);

        double FocalLength = GetRectifiedImgFocalLength(CameraNum, level);
        double x0rectified, y0rectified;
        GetRectifiedImgCenter(CameraNum, &x0rectified, &y0rectified, level);

        double RectPixalx = FocalLength * (x / z) + x0rectified;
        double RectPixaly = FocalLength * (y / z) + y0rectified;

        Eigen::Vector2d RectPixalCoor = LadybugPixCoor2ImgPixCoor(Eigen::Vector2d(RectPixalx, RectPixaly), wPR[level], hPR[level]);
        *RectifiedPixalx = RectPixalCoor(0);
        *RectifiedPixaly = RectPixalCoor(1);

        return true;
    }


    // Eigen::Matrix<float, 2, 3> UndistortMultiFisheye::computedStoR(float SphereX, float SphereY, float SphereZ, int n, int level)
    // {
    //     Eigen::Matrix2f dIL;
    //     dIL << 0,-1,1,0;

    //     Eigen::Matrix<float, 2, 3> duXs;

    //     Eigen::Matrix<float, 3, 1> RaySphere;
    //     RaySphere(0, 0) = SphereX;
    //     RaySphere(1, 0) = SphereY;
    //     RaySphere(2, 0) = SphereZ;

    //     Eigen::Matrix<float, 3, 3> RotationMat = (mvExParas[n].block<3, 3>(0, 0)).cast<float>();
    //     Eigen::Matrix<float, 3, 1> CamPos = mvExParas[n].block<3, 1>(0, 3).cast<float>();
    //     Eigen::Matrix<float, 3, 1> RayCoor = RaySphere - CamPos;
    //     Eigen::Matrix<float, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
    //     float x = RectifiedImgPtCoor(0, 0);
    //     float y = RectifiedImgPtCoor(1, 0);
    //     float z = RectifiedImgPtCoor(2, 0);

    //     float x0r, y0r;
    //     float f = mvInnerParasPR[level][n](0, 13);
    //     x0r = mvInnerParasPR[level][n](0, 14);
    //     y0r = mvInnerParasPR[level][n](0, 15);

    //     duXs << f/z, 0, -f/(z*z)*x,
    //             0, f/z, -f/(z*z)*y;
        
    //     Eigen::Matrix<float, 2, 3> dSR;
    //     dSR = dIL * duXs * RotationMat.inverse();

    //     return dSR;
    // }

    // void UndistortMultiFisheye::computedXs(float rho1, Mat33f R, Vec3f t, float X, float Y, float Z, 
    //                 Eigen::Matrix<float, 3, 6> * dXsdpose, Eigen::Matrix<float, 3, 1> * dXsdrho1)
    // {
    //     Vec3f Xc;
    //     Xc = R * Vec3f(X/Z, Y/Z, 1) * rho1 + t;
    //     float x,y,z;
    //     x = Xc(0);
    //     y = Xc(1);
    //     z = Xc(2);
    //     float l2 = x * x + y * y + z * z;
    //     float l23 = l2 * sqrt(l2);

    //     Eigen::Matrix<float, 3, 3> dXsdXc;
    //     dXsdXc << -mSphereRadius  *(y * y + z * z) / l23, x * y * mSphereRadius / l23, x * z * mSphereRadius / l23,
    //               x * y * mSphereRadius / l23, -mSphereRadius * (x * x + z * z) / l23, y * z * mSphereRadius / l23,
    //               x * z * mSphereRadius / l23, y * z * mSphereRadius / l23, -mSphereRadius * (x * x + y * y) / l23;

    //     Eigen::Matrix<float, 3, 6> dXcdpose;
    //     dXcdpose << 1, 0, 0, 0, -z, y,
    //                 0, 1, 0, z, 0, -x,
    //                 0, 0, 1, -y, x, 0;
        
    //     Eigen::Matrix<float, 3, 1> dXcdroh1;
    //     dXcdroh1 = - R * Vec3f(X/Z, Y/Z,1) / (rho1 * rho1);

    //     *dXsdpose = dXsdXc * dXcdpose;
    //     *dXsdrho1 = dXsdXc * dXcdroh1;
    // }


// ====================Equidistant================
    UndistortMFEquidistant::UndistortMFEquidistant(std::string configFileName, int camNum, bool noprefix)
    {
        printf("Creating Equidistant undistorter\n");

        camNums = camNum;  // 得到组合相机数量

        // read calib files,including internal, inv internal and external parameters.
        std::ifstream infile, invinfile, exfile;
        infile.open((configFileName + "/InnPara.txt").c_str());
        invinfile.open((configFileName + "/InvInnPara.txt").c_str());
        exfile.open((configFileName + "/ExPara.txt").c_str());

        Eigen::Matrix<double, 1, 16> temp;
        //int num;
        while (infile >> temp(0,0) >> temp(0,1) >> temp(0,2) >> temp(0,3) >> temp(0,4) >> temp(0,5)
                >> temp(0,6) >> temp(0,7) >> temp(0,8) >> temp(0,9) >> temp(0,10) >> temp(0,11) 
                >> temp(0,12) >> temp(0,13) >> temp(0,14) >> temp(0,15) )
        {
            //cout<< temp <<endl;
            mvInnerParas.emplace_back(temp);
        }

        while (invinfile >> temp(0,0) >> temp(0,1) >> temp(0,2) >> temp(0,3) >> temp(0,4) >> temp(0,5)
                >> temp(0,6) >> temp(0,7) >> temp(0,8) >> temp(0,9) >> temp(0,10) >> temp(0,11) 
                >> temp(0,12) >> temp(0,13) >> temp(0,14) >> temp(0,15) )
        {
            mvInvInnerParas.emplace_back(temp);
        }

        Eigen::Matrix<double, 4, 4> tempT;
        double Rx, Ry, Rz, Tx, Ty, Tz;
        double R[9];
        while (exfile >> Rx >> Ry >> Rz >> Tx >> Ty >> Tz)
        {
            CalculateRotationMatFromEulerAngle(Rx, Ry, Rz, R);
            tempT << R[0], R[1], R[2], Tx,
                     R[3], R[4], R[5], Ty,
                     R[6], R[7], R[8], Tz,
                     0,     0,     0,   1;
            // cout<< tempT << endl;
            mvExParas.emplace_back(tempT);
        }

        valid = true;
        passthrough = true;

        for (int i = 0; i < 256; i++) 
        {
            Binv[i] = B[i] = i;    // set gamma function to identity
        }

    }

    void UndistortMFEquidistant::setParasPyramids()
    {
        int wlvl = w;
        int hlvl = h;
        pyrLevelsUsed = 1;
        // 计算可建金字塔层数
        while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS) {
                wlvl /= 2;
                hlvl /= 2;
                pyrLevelsUsed++;
        }
        printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
                   pyrLevelsUsed - 1, wlvl, hlvl);
        if (wlvl > 100 && hlvl > 100) {
            printf("\n\n===============WARNING!===================\n "
                            "using not enough pyramid levels.\n"
                            "Consider scaling to a resolution that is a multiple of a power of 2.\n");
        }
        if (pyrLevelsUsed < 3) {
            printf("\n\n===============WARNING!===================\n "
                            "I need higher resolution.\n"
                            "I will probably segfault.\n");
        }

        wM3G = w - 3;
        hM3G = h - 3;

        wPR[0] = w;
        hPR[0] = h;
        mvInnerParasPR[0] = mvInnerParas;
        mvInvInnerParasPR[0] = mvInvInnerParas;

        for(int level = 1; level < pyrLevelsUsed; ++level)
        {
            wPR[level] = w >> level;
            hPR[level] = h >> level;


            mvInnerParasPR[level].resize(camNums);
            mvInvInnerParasPR[level].resize(camNums);
            for(int n = 0; n < camNums; n++)
            {
                
                int n1 = 1 << level;  // * 2 ^level
                long n2 = n1*n1;
	            long n4 = n2*n2;
	            long n6 = n4*n2;
	            long n8 = n6*n2;

                mvInnerParasPR[level][n].resize(1, 16);
                mvInvInnerParasPR[level][n].resize(1, 16);

                mvInnerParasPR[level][n](0,0) = mvInnerParas[n](0,0);
                mvInnerParasPR[level][n](0,1) = mvInnerParas[n](0,1) * n2;                  // k1
                mvInnerParasPR[level][n](0,2) = mvInnerParas[n](0,2) * n4;                  // k2
                mvInnerParasPR[level][n](0,3) = mvInnerParas[n](0,3) * n6;                  // k3
                mvInnerParasPR[level][n](0,4) = mvInnerParas[n](0,4) * n8;                  // k4
                mvInnerParasPR[level][n](0,5) = mvInnerParas[n](0,5) * n1;                   // p1
                mvInnerParasPR[level][n](0,6) = mvInnerParas[n](0,6) * n1;                   // p2
                mvInnerParasPR[level][n](0,7) = mvInnerParas[n](0,7);                       // c1
                mvInnerParasPR[level][n](0,8) = mvInnerParas[n](0,8);                       // c2
                mvInnerParasPR[level][n](0,9) = mvInnerParas[n](0,9);                       // lamda
                mvInnerParasPR[level][n](0,10) = mvInnerParas[n](0,10) / n1;                 // fdistorted
                mvInnerParasPR[level][n](0,11) = (mvInnerParas[n](0,11) + 0.5 ) / n1 -0.5;   // x0distorted
                mvInnerParasPR[level][n](0,12) = (mvInnerParas[n](0,12) + 0.5 ) / n1 -0.5;   // y0distorted
                mvInnerParasPR[level][n](0,13) = mvInnerParas[n](0,13) / n1;                 // frectified
                mvInnerParasPR[level][n](0,14) = (mvInnerParas[n](0,14) + 0.5 ) / n1 -0.5;   // x0rectified
                mvInnerParasPR[level][n](0,15) = (mvInnerParas[n](0,15) + 0.5 ) / n1 -0.5;   // y0rectified

                mvInvInnerParasPR[level][n](0,0) = mvInvInnerParas[n](0,0);
                mvInvInnerParasPR[level][n](0,1) = mvInvInnerParas[n](0,1) * n2;                    // k1
                mvInvInnerParasPR[level][n](0,2) = mvInvInnerParas[n](0,2) * n4;                    // k2
                mvInvInnerParasPR[level][n](0,3) = mvInvInnerParas[n](0,3) * n6;                    // k3
                mvInvInnerParasPR[level][n](0,4) = mvInvInnerParas[n](0,4) * n8;                    // k4
                mvInvInnerParasPR[level][n](0,5) = mvInvInnerParas[n](0,5) * n1;                     // p1
                mvInvInnerParasPR[level][n](0,6) = mvInvInnerParas[n](0,6) * n1;                     // p2
                mvInvInnerParasPR[level][n](0,7) = mvInvInnerParas[n](0,7);                         // c1
                mvInvInnerParasPR[level][n](0,8) = mvInvInnerParas[n](0,8);                         // c2
                mvInvInnerParasPR[level][n](0,9) = mvInvInnerParas[n](0,9);                         // lamda
                mvInvInnerParasPR[level][n](0,10) = mvInvInnerParas[n](0,10) / n1;                   // fdistorted
                mvInvInnerParasPR[level][n](0,11) = (mvInvInnerParas[n](0,11) + 0.5 ) / n1 -0.5;     // x0distorted
                mvInvInnerParasPR[level][n](0,12) = (mvInvInnerParas[n](0,12) + 0.5 ) / n1 -0.5;     // y0distorted
                mvInvInnerParasPR[level][n](0,13) = mvInvInnerParas[n](0,13) / n1;                   // frectified
                mvInvInnerParasPR[level][n](0,14) = (mvInvInnerParas[n](0,14) + 0.5 ) / n1 -0.5;     // x0rectified
                mvInvInnerParasPR[level][n](0,15) = (mvInvInnerParas[n](0,15) + 0.5 ) / n1 -0.5;     // y0rectified

            }
        }

    }

    //将鱼眼图像上的像素点投射到纠正后图像上
    bool UndistortMFEquidistant::LadybugRectifyImage(int CameraNum, double Pixalx, double Pixaly,
                double* RectifiedPixalx, double* RectifiedPixaly, int level)
    {
        //畸变参数k1,k2,k3,k4,p1,p2,c1,c2
        double k1, k2, k3, k4;
        double p1, p2;
        double c1, c2;
        //尺度参数lamda
        double lamda;
        //鱼眼图像内定向参数(fdistorted, x0distorted, y0distorted)
        //矫正图像内定向参数(frectified，x0rectified, y0rectified)
        double fdistorted, frectified;
        double x0distorted, y0distorted;
        double x0rectified, y0rectified;

        Eigen::Matrix<double, 1, 16> mCurrentInnerPara = mvInnerParasPR[level][CameraNum];
        if (CameraNum != mCurrentInnerPara(0,0))
        {
            for (size_t i = 0; i < mvInnerParasPR[level].size(); i++)
            {
                if (CameraNum = int(mvInnerParasPR[level][i](0, 0)))
                {
                    mCurrentInnerPara = mvInnerParasPR[level][i];
                }
            }
        }

        //赋值得到每个内参参数
        k1 = mCurrentInnerPara(0, 1);
        k2 = mCurrentInnerPara(0, 2);
        k3 = mCurrentInnerPara(0, 3);
        k4 = mCurrentInnerPara(0, 4);
        p1 = mCurrentInnerPara(0, 5);
        p2 = mCurrentInnerPara(0, 6);
        c1 = mCurrentInnerPara(0, 7);
        c2 = mCurrentInnerPara(0, 8);
        lamda = mCurrentInnerPara(0, 9);
        fdistorted = mCurrentInnerPara(0, 10);
        x0distorted = mCurrentInnerPara(0, 11);
        y0distorted = mCurrentInnerPara(0, 12);
        frectified = mCurrentInnerPara(0, 13);
        x0rectified = mCurrentInnerPara(0, 14);
        y0rectified = mCurrentInnerPara(0, 15);

        //开始纠正
        //首先转换为LadyBug坐标系(相当于图像坐标系逆时针旋转90度)下的平面坐标，此处使用的是鱼眼相机的x0,y0;
        Eigen::Vector2d LadybugPixalCoor = ImgPixCoor2LadybugPixCoor(Eigen::Vector2d(Pixalx, Pixaly), wPR[level], hPR[level]);
        double xt = LadybugPixalCoor(0)- x0distorted;
        double yt = LadybugPixalCoor(1) - y0distorted;
        //使用等距投影模型进行投影将鱼眼图像投影到纠正后图像
        double sqrtr = sqrt(xt*xt + yt * yt);
        double Rectifiedx = ((fdistorted*xt*tan(sqrtr / fdistorted)) / sqrtr);
        double Rectifiedy = ((fdistorted*yt*tan(sqrtr / fdistorted)) / sqrtr);
        //按照缩放比例缩放至鱼眼相机同等大小的尺度上
        double x_ = lamda * Rectifiedx;
        double y_ = lamda * Rectifiedy;
        //使用图像畸变模型，计算畸变改正量，使用的畸变参数有k1,k2,k3,k4,p1,p2,c1,c2
        double r2 = x_ * x_ + y_ * y_;
        double dx = x_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + 2 * p1*x_*y_ + p2 * (r2 + 2 * x_*x_) + c1 * x_ + c2 * y_;
        double dy = y_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + p1 * (r2 + 2 * y_*y_) + 2 * p2*x_*y_ + c1 * y_ + c2 * x_;
        //进行畸变改正并转换为像素坐标,注意此处要使用纠正后影像的x0,y0;
        double xRectTemp= (x_ - dx) + x0rectified;
        double yRectTemp= (y_ - dy) + y0rectified;
        //从Ladybug的像素坐标转换为正常图像的像素坐标(相当于顺时针旋转90度)
        Eigen::Vector2d RectifiedPixalCoor = LadybugPixCoor2ImgPixCoor(Eigen::Vector2d(xRectTemp, yRectTemp), wPR[level], hPR[level]);
        *RectifiedPixalx = RectifiedPixalCoor(0);
        *RectifiedPixaly = RectifiedPixalCoor(1);

        return true;
    }

    //将纠正后图像上的像素点投射到鱼眼图像上
    bool UndistortMFEquidistant::LadybugUnRectifyImage(int CameraNum, double Pixalx, double Pixaly,
                                   double* DistortedPixalx, double* DistortedPixaly, int level)
    {
        //畸变参数k1,k2,k3,k4,p1,p2,c1,c2
        double k1, k2, k3, k4;
        double p1, p2;
        double c1, c2;
        //尺度参数lamda
        double lamda;
        //鱼眼图像内定向参数(fdistorted, x0distorted, y0distorted)
        //矫正图像内定向参数(frectified，x0rectified, y0rectified)
        double fdistorted, frectified;
        double x0distorted, y0distorted;
        double x0rectified, y0rectified;

        Eigen::Matrix<double, 1, 16> mCurrentInvInnerPara = mvInvInnerParasPR[level][CameraNum];
        if (CameraNum != int(mCurrentInvInnerPara(0, 0)))
        {
            for (size_t i = 0; i < mvInvInnerParasPR[level].size(); i++)
            {
                if (CameraNum = int(mvInvInnerParasPR[level][i](0, 0)))
                {
                    mCurrentInvInnerPara = mvInvInnerParasPR[level][i];
                }
            }
        }

        //赋值得到每个内参参数
        k1 = mCurrentInvInnerPara(0, 1);
        k2 = mCurrentInvInnerPara(0, 2);
        k3 = mCurrentInvInnerPara(0, 3);
        k4 = mCurrentInvInnerPara(0, 4);
        p1 = mCurrentInvInnerPara(0, 5);
        p2 = mCurrentInvInnerPara(0, 6);
        c1 = mCurrentInvInnerPara(0, 7);
        c2 = mCurrentInvInnerPara(0, 8);
        lamda = mCurrentInvInnerPara(0, 9);
        fdistorted = mCurrentInvInnerPara(0, 10);
        x0distorted = mCurrentInvInnerPara(0, 11);
        y0distorted = mCurrentInvInnerPara(0, 12);
        frectified = mCurrentInvInnerPara(0, 13);
        x0rectified = mCurrentInvInnerPara(0, 14);
        y0rectified = mCurrentInvInnerPara(0, 15);

        Eigen::Vector2d LadybugPixalCoor = ImgPixCoor2LadybugPixCoor(Eigen::Vector2d(Pixalx, Pixaly), wPR[level], hPR[level]);
        double x_ = LadybugPixalCoor(0) - x0rectified;
        double y_ = LadybugPixalCoor(1) - y0rectified;
        double r2 = x_ * x_ + y_ * y_;
        double dx = x_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + 2 * p1*x_*y_ + p2 * (r2 + 2 * x_*x_) + c1 * x_ + c2 * y_;
        double dy = y_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + p1 * (r2 + 2 * y_*y_) + 2 * p2*x_*y_ + c1 * y_ + c2 * x_;
        double xt = x_ + dx;
        double yt = y_ + dy;
        double sqrtr = sqrt(xt*xt + yt * yt);
        double Distortedx = ((frectified*xt*atan(sqrtr / frectified)) / sqrtr);
        double Distortedy = ((frectified*yt*atan(sqrtr / frectified)) / sqrtr);
        double xd = lamda*Distortedx + x0distorted;
        double yd = lamda*Distortedy + y0distorted;

        Eigen::Vector2d DistortedPixalCoor = LadybugPixCoor2ImgPixCoor(Eigen::Vector2d(xd, yd), wPR[level], hPR[level]);
        *DistortedPixalx = DistortedPixalCoor(0);
        *DistortedPixaly = DistortedPixalCoor(1);

        return true;
    }

    void UndistortMFEquidistant::GetRectifiedImgCenter(int CameraNum, double* x0, double *y0, int level)
    {
        Eigen::Matrix<double, 1, 16> CurrentInnerPara = mvInnerParasPR[level][CameraNum];

        // Eigen::Vector2d ImgCenter = LadybugPixCoor2ImgPixCoor(Eigen::Vector2d(CurrentInnerPara(0, 14),
        //                                               CurrentInnerPara(0, 15)), wPR[level], wPR[level]);
        // *x0 = ImgCenter(0);
        // *y0 = ImgCenter(1);
        *x0 = CurrentInnerPara(0, 14);
        *y0 = CurrentInnerPara(0, 15);
    }

    //输出纠正后图像的焦距
    double UndistortMFEquidistant::GetRectifiedImgFocalLength(int CameraNum, int level)
    {
        Eigen::Matrix<double, 1, 16> CurrentInnerPara = mvInnerParasPR[level][CameraNum];
        return CurrentInnerPara(0, 13);
    }

}
