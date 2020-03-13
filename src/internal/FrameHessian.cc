#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"

#include <iostream>
#include <opencv2/opencv.hpp>


namespace ldso {

    namespace internal {

        void FrameHessian::setStateZero(const Vec10 &state_zero) {

            assert(state_zero.head<6>().squaredNorm() < 1e-20);

            this->state_zero = state_zero;

            for (int i = 0; i < 6; i++) {
                Vec6 eps;
                eps.setZero();
                eps[i] = 1e-3;
                SE3 EepsP = SE3::exp(eps);
                SE3 EepsM = SE3::exp(-eps);
                SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
                SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
                nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
            }

            // scale change
            SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
            w2c_leftEps_P_x0.translation() *= 1.00001;
            w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
            SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
            w2c_leftEps_M_x0.translation() /= 1.00001;
            w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
            nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

            nullspaces_affine.setZero();
            nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
            assert(ab_exposure > 0);
            nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
        }

        // for multi-fisheye
        void FrameHessian::setStateZeroforMF(const Vec16 &state_zeroforMF) {

            assert(state_zeroforMF.head<6>().squaredNorm() < 1e-20);

            this->state_zeroforMF = state_zeroforMF;

            // pose nullspace
            for (int i = 0; i < 6; i++) {
                Vec6 eps;
                eps.setZero();
                eps[i] = 1e-3;
                SE3 EepsP = SE3::exp(eps);
                SE3 EepsM = SE3::exp(-eps);
                SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
                SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
                nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
            }

            // scale change
            SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
            w2c_leftEps_P_x0.translation() *= 1.00001;
            w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
            SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
            w2c_leftEps_M_x0.translation() /= 1.00001;
            w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
            nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

            // nullspaces_affine.setZero();
            // nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
            // assert(ab_exposure > 0);
            // nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
        }

        void FrameHessian::makeImages(float *color, const shared_ptr<CalibHessian> &HCalib) {

            for (int i = 0; i < pyrLevelsUsed; i++) {
                dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];
                absSquaredGrad[i] = new float[wG[i] * hG[i]];
                memset(absSquaredGrad[i], 0, wG[i] * hG[i]);  // xy 方向梯度值的平方和
                memset(dIp[i], 0, 3 * wG[i] * hG[i]);  // 初始化函数。作用是将某一块内存中的内容全部设置为指定的值
            }
            dI = dIp[0];   // 原始尺度图像信息

            // make d0
            int w = wG[0];
            int h = hG[0];
            for (int i = 0; i < w * h; i++) {
                dI[i][0] = color[i];
            }

            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
                int wl = wG[lvl], hl = hG[lvl];
                Eigen::Vector3f *dI_l = dIp[lvl];

                float *dabs_l = absSquaredGrad[lvl];
                if (lvl > 0) {
                    int lvlm1 = lvl - 1;
                    int wlm1 = wG[lvlm1];
                    Eigen::Vector3f *dI_lm = dIp[lvlm1];

                    // lvl层由lvl-1层插值得到
                    for (int y = 0; y < hl; y++)
                        for (int x = 0; x < wl; x++) {
                            dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                                                           dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                           dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                           dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
                        }
                }

                for (int idx = wl; idx < wl * (hl - 1); idx++) {
                    float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
                    float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

                    if (std::isnan(dx) || std::fabs(dx) > 255.0) dx = 0;
                    if (std::isnan(dy) || std::fabs(dy) > 255.0) dy = 0;

                    dI_l[idx][1] = dx;
                    dI_l[idx][2] = dy;

                    dabs_l[idx] = dx * dx + dy * dy;

                    if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0) {
                        float gw = HCalib->getBGradOnly((float) (dI_l[idx][0]));
                        dabs_l[idx] *=
                                gw * gw;    // convert to gradient of original color space (before removing response).
                        // if (std::isnan(dabs_l[idx])) dabs_l[idx] = 0;
                    }
                }
            }

            // === debug stuffs === //
            if (setting_enableLoopClosing && setting_showLoopClosing) {
                frame->imgDisplay = cv::Mat(hG[0], wG[0], CV_8UC3);
                uchar *data = frame->imgDisplay.data;
                for (int i = 0; i < w * h; i++) {
                    for (int c = 0; c < 3; c++) {
                        *data = color[i] > 255 ? 255 : uchar(color[i]);
                        data++;
                    }
                }
            }
        }

//
        // // multi-fisheye image
        // void FrameHessian::makeImages(vector<float *> color, UndistortMultiFisheye * UMF) 
        // {
        //     camNums = color.size();
        //     int* wG = UMF->wPR;
        //     int* hG = UMF->hPR;
        //     for (int i = 0; i < pyrLevelsUsed; i++) 
        //     {
        //         vdIp[i].resize(camNums);
        //         vabsSquaredGrad[i].resize(camNums);
        //         for(int n = 0; n < camNums; n++)
        //         {
        //             vdIp[i][n] = new Eigen::Vector3f[wG[i] * hG[i]];
        //             vabsSquaredGrad[i][n] = new float[wG[i] * hG[i]];
        //             memset(vabsSquaredGrad[i][n], 0, wG[i] * hG[i]);  // xy 方向梯度值的平方和
        //             memset(vdIp[i][n], 0, 3 * wG[i] * hG[i]);  // 初始化函数。作用是将某一块内存中的内容全部设置为指定的值
        //         }
        //     }
        //     vdI.resize(camNums);
        //     vdI = vdIp[0];

        //     int w = wG[0];
        //     int h = hG[0];
        //     for(int n = 0; n < camNums; n++)
        //     {
        //         for(int i = 0; i < w * h; i++)
        //         {
        //             vdI[n][i][0] = color[n][i];     // 灰度信息赋值 [0], x,y梯度为[1][2]
        //         }
        //     }

        //     // 计算 x,y 方向梯度
        //     for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        //     {
        //         for(int n = 0; n < camNums; n++)
        //         {
        //             int wl = wG[lvl], hl = hG[lvl];
        //             vector<Eigen::Vector3f *> dI_l = vdIp[lvl];

        //             vector<float *> dabs_l = vabsSquaredGrad[lvl];
        //             if (lvl > 0) 
        //             {
        //                 int lvlm1 = lvl - 1;
        //                 int wlm1 = wG[lvlm1];
        //                 vector<Eigen::Vector3f *> dI_lm = vdIp[lvlm1];  // 上一层 lvl-1

        //                 // lvl层由lvl-1层插值得到
        //                 for (int y = 0; y < hl; y++)
        //                     for (int x = 0; x < wl; x++) 
        //                     {
        //                         dI_l[n][x + y * wl][0] = 0.25f * (dI_lm[n][2 * x + 2 * y * wlm1][0] +
        //                                                     dI_lm[n][2 * x + 1 + 2 * y * wlm1][0] +
        //                                                     dI_lm[n][2 * x + 2 * y * wlm1 + wlm1][0] +
        //                                                     dI_lm[n][2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        //                     }
        //             }

        //             for (int idx = wl; idx < wl * (hl - 1); idx++) 
        //             {
        //                 float dx = 0.5f * (dI_l[n][idx + 1][0] - dI_l[n][idx - 1][0]);
        //                 float dy = 0.5f * (dI_l[n][idx + wl][0] - dI_l[n][idx - wl][0]);

        //                 if (std::isnan(dx) || std::fabs(dx) > 255.0) dx = 0;
        //                 if (std::isnan(dy) || std::fabs(dy) > 255.0) dy = 0;

        //                 dI_l[n][idx][1] = dx;
        //                 dI_l[n][idx][2] = dy;

        //                 dabs_l[n][idx] = dx * dx + dy * dy;

        //                 UMF->getSize();

        //                 if (setting_gammaWeightsPixelSelect == 1 && UMF != 0) 
        //                 {
        //                     float gw = UMF->getBGradOnly(dI_l[n][idx][0]);
        //                     dabs_l[n][idx] *= gw * gw;    // convert to gradient of original color space (before removing response).
        //                     // if (std::isnan(dabs_l[idx])) dabs_l[idx] = 0;
        //                 }
        //             }
        //         }
        //     }

        //     // === debug stuffs === //
        //     if (setting_enableLoopClosing && setting_showLoopClosing) 
        //     {
        //         frame->vimgDisplay.resize(camNums);
        //         for(int n = 0; n < camNums; n++)
        //         {
        //             frame->vimgDisplay[n] = cv::Mat(hG[0], wG[0], CV_8UC3);
        //             uchar *data = frame->vimgDisplay[n].data;
        //             for (int i = 0; i < w * h; i++) 
        //             {
        //                 for (int c = 0; c < 3; c++) 
        //                 {
        //                     *data = color[n][i] > 255 ? 255 : uchar(color[n][i]);
        //                     data++;
        //                 }
        //             }
        //         }
                
        //     }
            
        // }
//

        // multi-fisheye image  在纠正后的平面图像上求梯度
        void FrameHessian::makeImages(vector<float *> color, UndistortMultiFisheye * UMF) 
        {
            camNums = color.size();
            int* wG = UMF->wPR;
            int* hG = UMF->hPR;


            vdfisheyeI.resize(camNums);
            for(int n = 0; n < camNums; n++)
            {
                vdfisheyeI[n] = new float [wG[0] * hG[0]];
                memset(vdfisheyeI[n], 0, wG[0] * hG[0]);
            }
            for (int i = 0; i < pyrLevelsUsed; i++) 
            {
                vdIp[i].resize(camNums);
                vabsSquaredGrad[i].resize(camNums);
                vmask[i].resize(camNums);
                for(int n = 0; n < camNums; n++)
                {
                    vdIp[i][n] = new Eigen::Vector3f[wG[i] * hG[i]];
                    vabsSquaredGrad[i][n] = new float[wG[i] * hG[i]];
                    vmask[i][n] = new float [wG[i] * hG[i]];
                    memset(vabsSquaredGrad[i][n], 0, wG[i] * hG[i]);  // xy 方向梯度值的平方和
                    memset(vdIp[i][n], 0, 3 * wG[i] * hG[i]);  // 初始化函数。作用是将某一块内存中的内容全部设置为指定的值
                    memset(vmask[i][n], 0,  wG[i] * hG[i]);
                }
            }
            vdI.resize(camNums);
            vdI = vdIp[0];

            int w = wG[0];
            int h = hG[0];
            for(int n = 0; n < camNums; n++)
            {
                for(int i = 0; i < w * h; i++)
                {
                    vdfisheyeI[n][i] = color[n][i];     // 灰度信息赋值 [0], x,y梯度为[1][2]
                }

                for (int y = 0; y < hG[0]; y++)
                {
                    for (int x = 0; x < wG[0]; x++)
                    {
                        double xDistorted, yDistorted;
                        int xRect = x;
                        int yRect = y;
                        UMF->LadybugUnRectifyImage(0, xRect, yRect, &xDistorted, &yDistorted, 0);
                        int i =  x + y * wG[0];
                        int j = (int)xDistorted + (int)yDistorted * wG[0];
                        if (xDistorted >= 0 && xDistorted < wG[0] &&
                            yDistorted >= 0 && yDistorted < hG[0])
                        {
                            // resizeRectifiedImg.at<cv::Vec3b>(yRect, xRect) = LBG.BilinearInterpolation(resizeFishEyeImg, xDistorted, yDistorted);
                            vdI[n][i][0] = color[n][j];
                            vmask[0][n][i] = 255;

                        }
                    }
                }
            }

            // 计算 x,y 方向梯度
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
            {
                for(int n = 0; n < camNums; n++)
                {
                    //memset(vabsSquaredGrad[lvl][n], 0, wG[lvl] * hG[lvl]);
                    int wl = wG[lvl], hl = hG[lvl];
                    Eigen::Vector3f* dI_l = vdIp[lvl][n];
                    float* vmask_l = vmask[lvl][n];

                    float* dabs_l = vabsSquaredGrad[lvl][n];
                    if (lvl > 0) 
                    {
                        int lvlm1 = lvl - 1;
                        int wlm1 = wG[lvlm1];
                        Eigen::Vector3f* dI_lm = vdIp[lvlm1][n];  // 上一层 lvl-1
                        float* vmask_lm = vmask[lvlm1][n];

                        // lvl层由lvl-1层插值得到
                        for (int y = 0; y < hl; y++)
                            for (int x = 0; x < wl; x++) 
                            {
                                dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
                                                            dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                            dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                            dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

                                vmask_l[x + y * wl] = 0.25f * (vmask_lm[2 * x + 2 * y * wlm1] +
                                                            vmask_lm[2 * x + 1 + 2 * y * wlm1] +
                                                            vmask_lm[2 * x + 2 * y * wlm1 + wlm1] +
                                                            vmask_lm[2 * x + 1 + 2 * y * wlm1 + wlm1]);
                            }
                    }
                    
                    // for (int idx = 0; idx < wl; idx++)
                    //     for (int idy = 1; idy < (hl-1); idy++) 
                    // for (int idx = wl; idx < wl * (hl - 1); idx++) 
                    for (int idx = wl*4; idx < wl * (hl - 4); idx++) 
                    {
                        //  if(lvl == 0 && n == 1 && idx == 215040)
                        //     {
                        //         cout << "level:" << lvl << " CamNum:" << n << " idx:" << idx <<  " vabsSquaredGrad[lvl][n][idx]:" << vabsSquaredGrad[lvl][n][idx] <<endl;
                        //     }
                        // double rx,ry;
                        // UMF->LadybugRectifyImage(n, idx, idy, &rx, &ry, lvl);
                        // double dx1, dx2, dx3, dx4, dy1, dy2, dy3, dy4;
                        // UMF->LadybugUnRectifyImage(n, rx+1, ry, &dx1, &dy1, lvl);
                        // UMF->LadybugUnRectifyImage(n, rx-1, ry, &dx2, &dy2, lvl);
                        // UMF->LadybugUnRectifyImage(n, rx, ry+1, &dx3, &dy3, lvl);
                        // UMF->LadybugUnRectifyImage(n, rx, ry-1, &dx4, &dy4, lvl);

                        // if(dx1 < 0 || dx2 < 0 || dx3 < 0 || dx4 < 0 ||
                        //     dy1 < 0 || dy2 < 0 || dy3 < 0 || dy4 < 0 )
                        //     continue;
                        // float dx = 0.5f * (dI_l[(int)dx1 + (int)(dy1*wl)][0] - dI_l[(int)dx2 + (int)(dy2*wl)][0]);
                        // float dy = 0.5f * (dI_l[(int)dx3 + (int)(dy3*wl)][0] - dI_l[(int)dx4 + (int)(dy4*wl)][0]);
                        // int id = idx + idy * wl;
                        if(vmask_l[idx + 1] == 0 || vmask_l[idx - 1] == 0 || vmask_l[idx + wl] == 0 || vmask_l[idx - wl] == 0)
                        {
                            vabsSquaredGrad[lvl][n][idx] = 0;
                            continue;
                        }
                        if(vmask_l[idx + 2] == 0 || vmask_l[idx - 2] == 0 || vmask_l[idx + 2*wl] == 0 || vmask_l[idx - 2*wl] == 0)
                        {
                            vabsSquaredGrad[lvl][n][idx] = 0;
                            continue;
                        }
                        if(vmask_l[idx + 3] == 0 || vmask_l[idx - 3] == 0 || vmask_l[idx + 3*wl] == 0 || vmask_l[idx - 3*wl] == 0)
                        {
                            vabsSquaredGrad[lvl][n][idx] = 0;
                            continue;
                        }
                        if(vmask_l[idx + 4] == 0 || vmask_l[idx - 4] == 0 || vmask_l[idx + 4*wl] == 0 || vmask_l[idx - 4*wl] == 0)
                        {
                            vabsSquaredGrad[lvl][n][idx] = 0;
                            continue;
                        }

                        float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
                        float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

                        if (std::isnan(dx) || std::fabs(dx) > 255.0) dx = 0;
                        if (std::isnan(dy) || std::fabs(dy) > 255.0) dy = 0;


                        dI_l[idx][1] = dx;
                        dI_l[idx][2] = dy;

                        dabs_l[idx] = dx * dx + dy * dy;

                        if (setting_gammaWeightsPixelSelect == 1 && UMF != 0) 
                        {
                            float gw = UMF->getBGradOnly(dI_l[idx][0]);
                            dabs_l[idx] *= gw * gw;    // convert to gradient of original color space (before removing response).
                            // if (std::isnan(dabs_l[idx])) dabs_l[idx] = 0;
                        }
                        
                    }

                }
            }

            for(int n = 0; n < UMF->camNums; n++)
            {
                float *mapmax0 = vabsSquaredGrad[0][n];

                // ofstream outim;
                // string path = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/vabsSquaredGrad_makeImages_" + to_string(n) + ".txt";
                // outim.open(path.c_str());

                // for (int iw = 0; iw < w;  iw++)
                //     for (int ih = 0; ih < h;  ih++)
                // {
                //     int idx = iw + ih * h;
                //     if(vabsSquaredGrad[0][n][idx] < 0)
                //     {
                //         cout << "level:" << 0 << " CamNum:" << n << " idx:" << idx <<  " vabsSquaredGrad[lvl][n][idx]:" << vabsSquaredGrad[0][n][idx] <<endl;
                //     }

                //     if(mapmax0[idx] < 0)
                //     {
                //         int aa = 0;
                //     }
                    
                //     outim << iw << " " << ih << " " << mapmax0[idx] << " " << vabsSquaredGrad[0][n][idx] <<endl;
                // }
                // outim.close();
            }

            // === debug stuffs === //
            if (setting_enableLoopClosing && setting_showLoopClosing) 
            {
                frame->vimgDisplay.resize(camNums);
                for(int n = 0; n < camNums; n++)
                {
                    frame->vimgDisplay[n] = cv::Mat(hG[0], wG[0], CV_8UC3);
                    uchar *data = frame->vimgDisplay[n].data;
                    for (int i = 0; i < w * h; i++) 
                    {
                        for (int c = 0; c < 3; c++) 
                        {
                            *data = color[n][i] > 255 ? 255 : uchar(color[n][i]);
                            data++;
                        }
                    }
                }
                
            }

            // string outpath = "/home/jiangfan/桌面/pan_dso_calib/select_pixel/";
            // // output image
            // for(int lvl = 0; lvl < pyrLevelsUsed; lvl++)
            // {
            //     for(int n = 0; n < camNums; n++)
            //     {
            //         int wl = wG[lvl], hl = hG[lvl];
            //         cv::Mat imga(hl,wl,CV_8UC1);
            //         for(int y=0; y<hl;y++)
            //             for(int x=0;x<wl;x++)
            //             {
            //                 int i=x+y*wl;
            //                 imga.at<uchar>(y,x) = vdIp[lvl][n][i](0);
            //             }
            //         string outimg = outpath + to_string(frameID) + "_" + to_string(lvl) + "_" + to_string(n) + "_" + "img.jpg";
            //         cv::imwrite(outimg, imga);
            //     }
            // }
            
        }

        void FrameHessian::takeData() {
            prior = getPrior().head<8>();
            delta = get_state_minus_stateZero().head<8>();
            delta_prior = (get_state() - getPriorZero()).head<8>();
        }

        void FrameHessian::takeDataforMF() {
            priorforMF = getPriorforMF();
            deltaforMF = get_state_minus_stateZeroforMF();
            delta_priorforMF = get_state_minus_PriorstateZeroforMF();
        }
    }

}
