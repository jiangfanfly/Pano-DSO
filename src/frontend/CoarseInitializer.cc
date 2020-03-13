#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"
#include "internal/GlobalFuncs.h"
#include "frontend/PixelSelector2.h"
#include "frontend/CoarseInitializer.h"
#include "frontend/nanoflann.h"

#include <iostream>


namespace ldso {

    CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3()) {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            points[lvl] = 0;
            numPoints[lvl] = 0;
        }

        JbBuffer = new Vec10f[ww * hh];
        JbBuffer_new = new Vec10f[ww * hh];

        fixAffine = true;
        printDebug = false;

        wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;     // 尺度
        wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
        wM.diagonal()[6] = SCALE_A;
        wM.diagonal()[7] = SCALE_B;
    }

    // for multi-fisheye
    CoarseInitializer::CoarseInitializer(UndistortMultiFisheye* uMF) : UMF(uMF)
    {
        camNums = UMF->getcamNums();
        int wh =UMF->getSize()(0)*UMF->getSize()(1);

        vJbBuffer.resize(camNums);
        vJbBuffer_new.resize(camNums);

        vvthisToNext_aff.resize(camNums);

        for(int i = 0; i<camNums; i++)
        {
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
            {
                vpoints[lvl].resize(camNums);
                vnumPoints[lvl].resize(camNums);
                vpoints[lvl][i] = 0;
                vnumPoints[lvl][i] = 0;
            }

            vvthisToNext_aff[i].resize(camNums);
            for(int n = 0; n <camNums; n++)
                vvthisToNext_aff[i][n] = AffLight(0, 0);

            vJbBuffer[i] = new Vec10f[wh];
            vJbBuffer_new[i] = new Vec10f[wh];
        }

        thisToNext = SE3();

        fixAffine = true;
        printDebug = false;

        wMF.diagonal()[0] = wMF.diagonal()[1] = wMF.diagonal()[2] = SCALE_XI_ROT;     // 尺度
        wMF.diagonal()[3] = wMF.diagonal()[4] = wMF.diagonal()[5] = SCALE_XI_TRANS;
       
       for (int i = 0; i < 25; i++)
       {
           wMF.diagonal()[6+i*2] = SCALE_A;
           wMF.diagonal()[6+i*2 + 1] = SCALE_A;
       }
        
    }

    CoarseInitializer::~CoarseInitializer() {
        // for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
        //     if (points[lvl] != 0) delete[] points[lvl];
        // }

        // delete[] JbBuffer;
        // delete[] JbBuffer_new;

        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        {
            for(int n = 0; n<camNums; n++)
            {
                if (vpoints[lvl][n] != 0) delete[] vpoints[lvl][n];
            }
            vpoints[lvl].clear();
        }
        

        for(int i = 0; i<camNums; i++)
        {

            delete[] vJbBuffer[i];
            delete[] vJbBuffer_new[i];
        }
        vJbBuffer.clear();
        vJbBuffer_new.clear();
    }

    bool CoarseInitializer::trackFrame(shared_ptr<FrameHessian> newFrameHessian) {

        newFrame = newFrameHessian;
        int maxIterations[] = {5, 5, 10, 30, 50};

        alphaK = 2.5 * 2.5;
        alphaW = 150 * 150;
        regWeight = 0.8;
        couplingWeight = 1;
        // 第二帧
        if (!snapped) {
            thisToNext.translation().setZero();
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
                int npts = numPoints[lvl];
                Pnt *ptsl = points[lvl];
                for (int i = 0; i < npts; i++) {
                    ptsl[i].iR = 1;
                    ptsl[i].idepth_new = 1;
                    ptsl[i].lastHessian = 0;
                }
            }
        }

        SE3 refToNew_current = thisToNext;
        AffLight refToNew_aff_current = thisToNext_aff;

        if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
            refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure),
                                            0); // coarse approximation.


        Vec3f latestRes = Vec3f::Zero();
        for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--) {


            if (lvl < pyrLevelsUsed - 1)
                propagateDown(lvl + 1);   // 将当前层所有点的逆深度设置为的它们 parent （上一层）的逆深度,加速优化

            Mat88f H, Hsc;
            Vec8f b, bsc;
            resetPoints(lvl);
            Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);  // 计算Hessian矩阵等信息 ,注释中a 为所有点的逆深度，b为位姿se3和放射变化a,b
            applyStep(lvl);

            float lambda = 0.1;
            float eps = 1e-4;
            int fails = 0;

            int iteration = 0;
            while (true) {
                Mat88f Hl = H;
                for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);   // L-M lamda H(1+lamda)x = b
                Hl -= Hsc * (1 / (1 + lambda));         // 舒尔布消元后 HH = Hbb - Hsc
                Vec8f bl = b - bsc * (1 / (1 + lambda));   // bb = b - bsc           xb = HH^-1 * bb

                Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
                bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));


                Vec8f inc;  // 更新量
                if (fixAffine) {
                    inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                      (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                    inc.tail<2>().setZero();
                } else
                    inc = -(wM * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.


                SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
                AffLight refToNew_aff_new = refToNew_aff_current;
                refToNew_aff_new.a += inc[6];
                refToNew_aff_new.b += inc[7];
                doStep(lvl, lambda, inc);   // 更新点的逆深度信息


                Mat88f H_new, Hsc_new;
                Vec8f b_new, bsc_new;
                Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);  // 计算Hessian矩阵等信息
                Vec3f regEnergy = calcEC(lvl);   // AccumulatorX累计残差。

                float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
                float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);


                bool accept = eTotalOld > eTotalNew;   // 如果能量函数值减小则接收此次优化

                if (accept) {

                    if (resNew[1] == alphaK * numPoints[lvl])
                        snapped = true;
                    H = H_new;
                    b = b_new;
                    Hsc = Hsc_new;
                    bsc = bsc_new;
                    resOld = resNew;
                    refToNew_aff_current = refToNew_aff_new;
                    refToNew_current = refToNew_new;
                    applyStep(lvl);   //生效前面保存的优化结果
                    optReg(lvl);    // 所有点的iR设置为 neighbor的中位数
                    lambda *= 0.5;
                    fails = 0;
                    if (lambda < 0.0001) lambda = 0.0001;
                } else {
                    fails++;
                    lambda *= 4;
                    if (lambda > 10000) lambda = 10000;
                }

                bool quitOpt = false;

                if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
                    Mat88f H, Hsc;
                    Vec8f b, bsc;

                    quitOpt = true;
                }


                if (quitOpt) break;
                iteration++;
            }
            latestRes = resOld;

        }

        thisToNext = refToNew_current;
        thisToNext_aff = refToNew_aff_current;

        for (int i = 0; i < pyrLevelsUsed - 1; i++)
            propagateUp(i);    // 使用低一层点的逆深度更新其高一层点 parent 的逆深度

        frameID++;
        if (!snapped) snappedAt = 0;

        if (snapped && snappedAt == 0)
            snappedAt = frameID;

        return snapped && frameID > snappedAt + 5;
    }

    bool CoarseInitializer::trackFrameforMF(shared_ptr<FrameHessian> newFrameHessian, shared_ptr<PangolinDSOViewer> viewer)
    {
        newFrame = newFrameHessian;

        if(viewer)
            viewer->pushLiveFrameforMF(newFrame);

        int maxIterations[] = {5, 5, 10, 30, 50};

        alphaK = 2.5 * 2.5;
        alphaW = 150 * 150;
        regWeight = 0.8;
        couplingWeight = 1;

        // 第二帧 进入 初始化每个点逆深度为1, 初始化光度参数, 位姿SE3
        // 初始化 点 参数 snapped应该指的是位移足够大了，不够大就重新优化
        if (!snapped) 
        {
            thisToNext.translation().setZero();
            for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
            {
                for(int n = 0; n < camNums; n++)
                {
                    int npts = vnumPoints[lvl][n];
                    Pnt *ptsl = vpoints[lvl][n];
                    for (int i = 0; i < npts; i++) 
                    {
                        ptsl[i].iR = 1;
                        ptsl[i].idepth_new = 1;
                        ptsl[i].lastHessian = 0;
                    }
                }
            }
        }

        SE3 refToNew_current = thisToNext;
        vector<vector<AffLight>>  vvrefToNew_aff_current(vvthisToNext_aff);

        if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
        {
            for(int n = 0; n < camNums; n++)
                for(int i = 0; i < camNums; i++)
                vvrefToNew_aff_current[n][i] = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0); // coarse approximation.  a21 = t2*e^a2 / t1*e^a1   b21 = b2 - a21*b1 
        }

        Vec3f latestRes = Vec3f::Zero();

        for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)    // 优化从金字塔最高层开始，也就是分辨率最低开始，逐步精化结果
        {
            
            if (lvl < pyrLevelsUsed - 1)
                propagateDownforMF(lvl + 1);    // 将当前层所有点的逆深度设置为的它们 parent （上一层）的逆深度,加速优化, 顶层未初始化到, reset来完成

            Eigen::Matrix<float, 56, 56> H, Hsc;
            Eigen::Matrix<float, 56, 1> b, bsc;
            resetPointsforMF(lvl);  // 只在最高层有用,进行初始化  点深度设为邻近的均值
            
            //LOG(INFO) << "calcResAndGSforMF" <<endl;
            Vec3f resOld = calcResAndGSforMF(lvl, H, b, Hsc, bsc, refToNew_current, vvrefToNew_aff_current, false);  // 计算Hessian矩阵等信息 
            

            applyStepforMF(lvl);

            float lambda = 0.1;
            float eps = 1e-4;
            int fails = 0;

            int iteration = 0;
            while(true)
            {
                Eigen::Matrix<float, 56, 56> Hl = H;
                for(int i = 0; i < 56; i++)     //// L-M lamda H(1+lamda)x = b
                    Hl(i, i) *= (1 + lambda);
                Hl -= Hsc * (1 / (1 + lambda));  // 舒尔补消元后 HH = Hbb - Hsc

                Eigen::Matrix<float, 56, 1> bl = b - bsc * (1 / (1 + lambda));     // bb = b - bsc           xb = HH^-1 * bb

                Hl = wMF * Hl * wMF * (0.01f / (UMF->wPR[lvl] * UMF->hPR[lvl]));
                bl = wMF * bl * (0.01f / (UMF->wPR[lvl] * UMF->hPR[lvl]) ); 

                Eigen::Matrix<float, 56, 1> inc;  // 更新量
                if (fixAffine) {
                    inc.head<6>() = -(wMF.toDenseMatrix().topLeftCorner<6, 6>() *
                                      (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                    inc.tail<50>().setZero();
                } else
                    inc = -(wMF * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.

                SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
                vector<vector<AffLight>> vvrefToNew_aff_new = vvrefToNew_aff_current;
                for(int n1 = 0; n1 < camNums; n1++)
                    for(int n2 = 0; n2 < camNums; n2++)
                    {
                        int abidx = 2 * (n1 + camNums*n2) + 6;
                        vvrefToNew_aff_new[n1][n2].a += inc[abidx];
                        vvrefToNew_aff_new[n1][n2].b += inc[abidx + 1];
                    }

                doStepforMF(lvl, lambda, inc); // 更新点的逆深度信息

                Mat5656f H_new, Hsc_new;
                Vec56f b_new, bsc_new;
                Vec3f resNew = calcResAndGSforMF(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, vvrefToNew_aff_new, false);  // 计算Hessian矩阵等信息
                Vec3f regEnergy = calcECforMF(lvl);   // AccumulatorX累计残差。

                // 点光度残差和 + 平移量 + 点深度idepth与iR差 之和
                float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
                float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);


                bool accept = eTotalOld > eTotalNew;   // 如果能量函数值减小则接收此次优化
                
                if (accept) 
                {
                    // 应该是位移足够大，才开始优化IR
                    if (resNew[1] == alphaK * sumPoints[lvl])   // 当 alphaEnergy > alphaK*npts
                        snapped = true;
                    H = H_new;
                    b = b_new;
                    Hsc = Hsc_new;
                    bsc = bsc_new;
                    resOld = resNew;
                    vvrefToNew_aff_current = vvrefToNew_aff_new;
                    refToNew_current = refToNew_new;
                    applyStepforMF(lvl);   //生效前面保存的优化结果
                    optRegforMF(lvl);    // 所有点的iR设置为 neighbor的中位数
                    lambda *= 0.5;
                    fails = 0;
                    if (lambda < 0.0001) lambda = 0.0001;
                } else {
                    fails++;
                    lambda *= 4;
                    if (lambda > 10000) lambda = 10000;
                }

                bool quitOpt = false;

                if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
                    Mat5656f H, Hsc;
                    Vec56f b, bsc;

                    quitOpt = true;
                }


                if (quitOpt) break;
                iteration++;
            }
            latestRes = resOld;

            LOG(INFO) << "calcResAndGSforMF is over." << " level: " << lvl <<".  iterations:" << iteration << ". Res: " << latestRes(0)/latestRes(2) << endl;
        
        }   
        // 优化后赋值位姿, 从底层计算上层点的深度
        thisToNext = refToNew_current;
        vvthisToNext_aff = vvrefToNew_aff_current;

        LOG(INFO) << "T New_currentToref(wc) :" << endl << refToNew_current.matrix().inverse() <<endl;
                

        for (int i = 0; i < pyrLevelsUsed - 1; i++)
            propagateUpforMF(i);    // 使用低一层点的逆深度更新其高一层点 parent 的逆深度

        frameID++;
        if (!snapped) snappedAt = 0;

        if (snapped && snappedAt == 0)
            snappedAt = frameID;

        debugPlotforMF(0, viewer);

        // // output points
        // string path = "/home/jiangfan/桌面/pan_dso_calib/pano_inital/pc.txt";
        // ofstream outp;
        // outp.open(path.c_str());
        // int id = 0;
        // // for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        // //     {
        //         for(int n = 0; n < camNums; n++)
        //         {
        //             int npts = vnumPoints[0][n];
        //             Pnt *ptsl = vpoints[0][n];
        //             for (int i = 0; i < npts; i++) 
        //             {
        //                 outp << id << " " <<  ptsl[i].xs << " " <<  ptsl[i].ys << " " <<  ptsl[i].zs << " " <<  ptsl[i].idepth << endl;
        //                 id ++ ;
        //             }
                    
        //         }
        //     // }
        // outp.close();
        

        LOG(INFO) << "snapped:" << snapped <<".  snappedAt:" << snappedAt << endl;
        return snapped && frameID > snappedAt + 5;
        // return snapped && frameID > snappedAt;
    }

    void CoarseInitializer::debugPlotforMF(int lvl, shared_ptr<PangolinDSOViewer> viewer)
    {
        bool needCall = false;
        if(viewer)
            needCall = needCall || viewer->needPushDepthImage();
        if(!needCall) return;

        int wl = firstFrame->UMF->wPR[lvl], hl = firstFrame->UMF->hPR[lvl];
        vector<MinimalImageB3* >iRImg;
        iRImg.resize(camNums);
        for(int n = 0; n < camNums; n++)
        {
            Eigen::Vector3f* colorRef = firstFrame->vdIp[lvl][n];
            // float* colorRef = firstFrame->vdfisheyeI[n];
            iRImg[n] = new MinimalImageB3(wl, hl);
            for(int i=0; i< wl*hl; i++)
                iRImg[n]->at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);
                // iRImg[n]->at(i) = Vec3b(colorRef[i],colorRef[i],colorRef[i]);

            int npts = vnumPoints[lvl][n];
            float nid = 0, sid=0;
            for(int i=0;i<npts;i++)
            {
                Pnt* point = vpoints[lvl][n]+i;
                if(point->isGood)
                {
                    nid++;
                    sid += point->iR;
                }
            }
            float fac = nid / sid;

            for(int i=0;i<npts;i++)
            {
                Pnt* point = vpoints[lvl][n]+i;

                if(!point->isGood)
                {
                    // double dx, dy;
                    // UMF->LadybugUnRectifyImage(n, point->u, point->v, &dx, &dy, 0);
                    // if(dx > wl-2 || dy > hl-2 || dx < 0 || dy < 0)
                    //     continue;
                    // int idx = ceil(dx);
                    // int idy = ceil(dy);
                    // iRImg[n]->setPixel9(idx, idy, Vec3b(0,0,0));
                    iRImg[n]->setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));
                }
                else
                {
                    // double dx, dy;
                    // UMF->LadybugUnRectifyImage(n, point->u, point->v, &dx, &dy, 0);
                    // if(dx > wl-2 || dy > hl-2 || dx < 0 || dy < 0)
                    //     continue;
                    // int idx = ceil(dx);
                    // int idy = ceil(dy);
                    // iRImg[n]->setPixel9(idx, idy, makeRainbow3B(point->iR*fac));

                    iRImg[n]->setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
                }
                    
            }
            
        }
       
        if(viewer)
            viewer->pushDepthImageforMF(iRImg);

        for(int n = 0; n < camNums; n++)
        {
            cv::Mat imga(hl,wl,CV_8UC3);
            for(int y=0; y<hl;y++)
                for(int x=0;x<wl;x++)
            {
                //float v = (!std::isfinite(fh->vdI[i][camNum][0]) || v>255) ? 255 : fh->vdI[i][camNum][0];
                //imga.at<cv::Vec3b>(i) = cv::Vec3b(v, fh->vdI[i][camNum][1], fh->vdI[i][camNum][2]);
                int i=x+y*wl;
                imga.at<cv::Vec3b>(y,x) = cv::Vec3b(iRImg[n]->data[i][0], iRImg[n]->data[i][1], iRImg[n]->data[i][2]);
            }
            string outpath = "/home/jiangfan/桌面/pan_dso_calib/pano_inital/";
            string outimg = outpath +  to_string(n) + "imgin.jpg";
            cv::imwrite(outimg, imga);
        }
    }

    // calculates residual, Hessian and Hessian-block neede for re-substituting depth.
    Vec3f CoarseInitializer::calcResAndGS(
            int lvl, Mat88f &H_out, Vec8f &b_out,
            Mat88f &H_out_sc, Vec8f &b_out_sc,
            const SE3 &refToNew, AffLight refToNew_aff,
            bool plot) {
        int wl = w[lvl], hl = h[lvl];
        Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];
        Eigen::Vector3f *colorNew = newFrame->dIp[lvl];

        Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);    

        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];

        Accumulator11 E;
        acc9.initialize();
        E.initialize();


        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        for (int i = 0; i < npts; i++) {

            Pnt *point = ptsl + i;

            point->maxstep = 1e10;
            // setFisrtFrame 中把所有点设为isgood = true  根据判断当前点是否是一个足够好的点选择是否对逆深度求导
            if (!point->isGood) {
                E.updateSingle((float) (point->energy[0]));
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }

            VecNRf dp0;    // dp0, dp1, dp2, dp3, dp4, dp5 是光度误差对 se(3) 六个量的导数
            VecNRf dp1;
            VecNRf dp2;
            VecNRf dp3;
            VecNRf dp4;
            VecNRf dp5;
            VecNRf dp6;    // dp6, dp7 是光度误差对辐射仿射变换的参数的导数
            VecNRf dp7;
            VecNRf dd;     // dd 是光度误差对逆深度的导数
            VecNRf r;
            JbBuffer_new[i].setZero();

            // sum over all residuals.
            bool isGood = true;
            float energy = 0;
            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];


                Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;  // R21 * K^-1 * [u,v,1] + t21 * d1  得到在2帧相机坐标系下坐标
                float u = pt[0] / pt[2];
                float v = pt[1] / pt[2];   // [u,v] 第2帧相机归一化坐标系下坐标
                float Ku = fxl * u + cxl;
                float Kv = fyl * v + cyl;
                float new_idepth = point->idepth_new / pt[2];   // d2

                if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0)) {
                    isGood = false;
                    break;
                }

                Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
                //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

                //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
                float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

                if (!std::isfinite(rlR) || !std::isfinite((float) hitColor[0])) {
                    isGood = false;   
                    break;
                }


                float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];  // 计算光度误差 r = I2 - a21*I1 -b21 (I2 = a21*I1 + b21)
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);    // abs(r)<sigam :r^2    abs(r)>= sigma : sigam*(2abs(r)-sigam)
 

                float dxdd = (t[0] - t[2] * u) / pt[2];     // d2(t21_x-u2'*t21_z)     用于dr21/dd1  (dd)
                float dydd = (t[1] - t[2] * v) / pt[2];     // d2(t21_y-v2'*t21_z)

                if (hw < 1) hw = sqrtf(hw);        // 残差函数 f(x) =sqrt(hw) *r
                float dxInterp = hw * hitColor[1] * fxl;    // hw(gx*fx)
                float dyInterp = hw * hitColor[2] * fyl;    // hw(gy*fy)
                dp0[idx] = new_idepth * dxInterp;           // d2 * hw(gx*fx) = hw(gx*fx*d2)
                dp1[idx] = new_idepth * dyInterp;           // d2 * hw(gy*fy) = hw(gx*fx*d2)
                dp2[idx] = new_idepth * (u * dxInterp + v * dyInterp);    // -hw(gx*fx*d2*u + gy*fy*d2*v)
                dp3[idx] = - u * v * dxInterp - (1 + v * v) * dyInterp;
                dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
                dp5[idx] = -v * dxInterp + u * dyInterp;
                dp6[idx] = -hw * r2new_aff[0] * rlR;           // ？？优化对象是loga21=refToNew_aff_current.a,然后r2new_aff[0]=exp(log21)   所以r21 = wh(I2 - exp(loga21)*I1-b21),   dr21/dlog21 = -wh*exp(log21)*I1
                dp7[idx] = -hw * 1;
                dd[idx] = dxInterp * dxdd + dyInterp * dydd;   // hw(gx*fx)*dxdd 
                r[idx] = hw * residual;

                float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
                if (maxstep < point->maxstep) point->maxstep = maxstep;

                // immediately compute dp*dd' and dd*dd' in JbBuffer1.   用与舒尔布中消去所有点的逆深度参数，留下位姿和a,b后的雅克比
                JbBuffer_new[i][0] += dp0[idx] * dd[idx];
                JbBuffer_new[i][1] += dp1[idx] * dd[idx];
                JbBuffer_new[i][2] += dp2[idx] * dd[idx];
                JbBuffer_new[i][3] += dp3[idx] * dd[idx];
                JbBuffer_new[i][4] += dp4[idx] * dd[idx];
                JbBuffer_new[i][5] += dp5[idx] * dd[idx];
                JbBuffer_new[i][6] += dp6[idx] * dd[idx];
                JbBuffer_new[i][7] += dp7[idx] * dd[idx];
                JbBuffer_new[i][8] += r[idx] * dd[idx];
                JbBuffer_new[i][9] += dd[idx] * dd[idx];
            }

            if (!isGood || energy > point->outlierTH * 20) {   // 当点位于图像外，或者pattern 残差和大于阈值 则判定此点 isGood = false
                E.updateSingle((float) (point->energy[0]));
                point->isGood_new = false;
                point->energy_new = point->energy;
                continue;
            }


            // add into energy. good的点的雅克比才会被归入到总的雅克比中参与增量方程的求解；
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new[0] = energy;

            // update Hessian matrix.  
            // 计算前面4的整数倍 patternNum
            for (int i = 0; i + 3 < patternNum; i += 4)
                acc9.updateSSE(
                        _mm_load_ps(((float *) (&dp0)) + i),
                        _mm_load_ps(((float *) (&dp1)) + i),
                        _mm_load_ps(((float *) (&dp2)) + i),
                        _mm_load_ps(((float *) (&dp3)) + i),
                        _mm_load_ps(((float *) (&dp4)) + i),
                        _mm_load_ps(((float *) (&dp5)) + i),
                        _mm_load_ps(((float *) (&dp6)) + i),
                        _mm_load_ps(((float *) (&dp7)) + i),
                        _mm_load_ps(((float *) (&r)) + i));

            // 计算多出4的整倍数的最后 patternNum
            for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
                acc9.updateSingle(
                        (float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i],
                        (float) dp4[i], (float) dp5[i], (float) dp6[i], (float) dp7[i],
                        (float) r[i]);


        }

        E.finish();
        acc9.finish();

        // calculate alpha energy, and decide if we cap it.   ？？ 加入和深度1平面的约束
        Accumulator11 EAlpha;
        EAlpha.initialize();
        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood_new) {
                E.updateSingle((float) (point->energy[1]));
            } else {
                point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
                E.updateSingle((float) (point->energy_new[1]));
            }
        }
        EAlpha.finish();
        float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

        // compute alpha opt.
        float alphaOpt;
        if (alphaEnergy > alphaK * npts) {
            alphaOpt = 0;
            alphaEnergy = alphaK * npts;
        } else {
            alphaOpt = alphaW;
        }

        acc9SC.initialize();
        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood_new)
                continue;

            point->lastHessian_new = JbBuffer_new[i][9];

            JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
            JbBuffer_new[i][9] += alphaOpt;

            if (alphaOpt == 0) {
                JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
                JbBuffer_new[i][9] += couplingWeight;
            }

            JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);   // ?? 相当于求逆？便于后面求 xa ?  分母里多了一个1，猜测是为了防止JbBuffer_new[i][9]太小造成系统不稳定。
            acc9SC.updateSingleWeighted(
                    (float) JbBuffer_new[i][0], (float) JbBuffer_new[i][1], (float) JbBuffer_new[i][2],
                    (float) JbBuffer_new[i][3],
                    (float) JbBuffer_new[i][4], (float) JbBuffer_new[i][5], (float) JbBuffer_new[i][6],
                    (float) JbBuffer_new[i][7],
                    (float) JbBuffer_new[i][8], (float) JbBuffer_new[i][9]);
        }
        acc9SC.finish();

        H_out = acc9.H.topLeftCorner<8, 8>();// / acc9.num;H = JbWJb
        b_out = acc9.H.topRightCorner<8, 1>();// / acc9.num;b = JbWr
        H_out_sc = acc9SC.H.topLeftCorner<8, 8>();// / acc9.num;  H = JnewWJnew    Jnew = 1/Ja * (Ja^T*Jb)
        b_out_sc = acc9SC.H.topRightCorner<8, 1>();// / acc9.num; b = JnewWr

        H_out(0, 0) += alphaOpt * npts;
        H_out(1, 1) += alphaOpt * npts;
        H_out(2, 2) += alphaOpt * npts;

        Vec3f tlog = refToNew.log().head<3>().cast<float>();
        b_out[0] += tlog[0] * alphaOpt * npts;
        b_out[1] += tlog[1] * alphaOpt * npts;
        b_out[2] += tlog[2] * alphaOpt * npts;


        return Vec3f(E.A, alphaEnergy, E.num);
    }

    // for multi-fisheye
    Vec3f CoarseInitializer::calcResAndGSforMF(
                int lvl,
                Eigen::Matrix<float, 56, 56> &H_out, Eigen::Matrix<float, 56, 1> &b_out,
                Eigen::Matrix<float, 56, 56> &H_out_sc, Eigen::Matrix<float, 56, 1> &b_out_sc,
                const SE3 &refToNew, vector<vector<AffLight>> vvrefToNew_aff,
                bool plot)
    {
        int wl = UMF->wPR[lvl], hl = UMF->hPR[lvl];

        vector<Eigen::Vector3f* > vcolorRef(firstFrame->vdIp[lvl]);
        vector<Eigen::Vector3f* > vcolorNew(newFrame->vdIp[lvl]);

        Mat33f R = refToNew.rotationMatrix().cast<float>();
        Vec3f t = refToNew.translation().cast<float>();
        
        vector<vector<Eigen::Vector2f>> vvr2new_aff;
        vvr2new_aff.resize(camNums);
        for (int n1 = 0; n1 < camNums; n1++)
        {
            vvr2new_aff[n1].resize(camNums);
            for (int n2 = 0; n2 < camNums; n2++)
            {
                vvr2new_aff[n1][n2]= Eigen::Vector2f(exp(vvrefToNew_aff[n1][n2].a), vvrefToNew_aff[n1][n2].b);
            }        
        }

        Accumulator11 E;
        acc57.initialize();  // Accumulator57
        E.initialize();
        //LOG(INFO) << "calc pt" << endl ;
        for (int n = 0; n < camNums; n++)
        {
            int npts = vnumPoints[lvl][n];
            Pnt *ptsl = vpoints[lvl][n];
            for (int i = 0; i < npts; i++)
            {
                // LOG(INFO) << "calc pt start." << endl ;
                
                Pnt*point = ptsl + i;

                point->maxstep = 1e10;
                // setFisrtFrame 中把所有点设为isgood = true  根据判断当前点是否是一个足够好的点选择是否对逆深度求导
                if(!point->isGood)
                {
                    E.updateSingle((float) (point->energy[0]));
                    point->energy_new = point->energy;
                    point->isGood_new = false;
                    continue;
                }

                VecNRf dp0;    // dp0, dp1, dp2, dp3, dp4, dp5 是光度误差对 se(3) 六个量的导数
                VecNRf dp1;
                VecNRf dp2;
                VecNRf dp3;
                VecNRf dp4;
                VecNRf dp5;
                VecNRf dp6;    // dp6, dp7 是光度误差对辐射仿射变换的参数的导数
                VecNRf dp7;
                VecNRf dd;     // dd 是光度误差对逆深度的导数
                VecNRf r;
                vJbBuffer_new[n][i].setZero();

                // sum over all residuals.
                bool isGood = true;
                float energy = 0;
                for (int idx = 0; idx < patternNum; idx++) 
                {
                    int dx = patternP[idx][0];
                    int dy = patternP[idx][1];

                    if( (firstFrame->vmask[lvl][n][(int)(point->u + dx) + (int)(point->v + dy)*wl]) == 0 )   
                    {
                        isGood = false;
                        break;
                    }

                    double X,Y,Z;
                    //LOG(INFO) << "LadybugProjectFishEyePtToSphere" << endl ;
                    //UMF->LadybugProjectFishEyePtToSphere(n, point->u + dx, point->v + dy, &X, &Y, &Z, lvl);   // [u,v] -> sphereX 鱼眼像素坐标转换到球面坐标(球面半径20)
                    UMF->LadybugProjectRectifyPtToSphere(n, point->u + dx, point->v + dy, &X, &Y, &Z, lvl);

                    float SphereRadius = UMF->GetSphereRadius();

                    //Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) + t * point->idepth_new;   // pt = R21 * X球面归一 +  t21 * rho1
                    Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius)/ point->idepth_new + t ;   // pt = R21 * X球面归一 / rho1 +  t21

                    if(!isfinite(pt(0)))
                    {
                        isGood = false;
                        break;
                    }

                    float S_norm_pt = SphereRadius/pt.norm();

                    float xs = S_norm_pt*pt(0);
                    float ys = S_norm_pt*pt(1);
                    float zs = S_norm_pt*pt(2);

                    double Ku, Kv;
                    int tocamnum;

                    //UMF->LadybugReprojectSpherePtToFishEyeImg(xs, ys, zs, &tocamnum, &Ku, &Kv, lvl);
                    // UMF->LadybugReprojectSpherePtToRectify(xs, ys, zs, &tocamnum, &Ku, &Kv, lvl);
                    UMF->LadybugReprojectSpherePtToRectifyfixNum(xs, ys, zs, n, &Ku, &Kv, lvl);
                    tocamnum = n;
                    //LOG(INFO) << "LadybugReprojectSpherePtToFishEyeImg" << endl ;

                    // dpi/pz' 
			        float new_idepth = point->idepth_new/pt.norm(); // 新一帧target上的逆深度  rho2 = rho1 / pt.norm()

                    if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
                    //if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 )) 
                    {
                        isGood = false;
                        break;
                    }
                    if( (newFrame->vmask[lvl][tocamnum][(int)Ku + (int)Kv*wl]) == 0 )  
                    {
                        isGood = false;
                        break;
                    } 

                    Vec3f hitColor = getInterpolatedElement33(vcolorNew[tocamnum], Ku, Kv, wl);    // 在当前第二帧 中的 灰度值
                    // Vec3f hitColor = getInterpolatedElement33BiCub(vcolorNew[tocamnum], Ku, Kv, wl);
                    // Vec3f hitColor1 = vcolorNew[tocamnum][(int)Ku + (int)Kv * wl];

                    //float rlR1 = vcolorRef[n][int(point->u +dx) + int(point->v+dy) * wl][0];
                    float rlR = getInterpolatedElement31(vcolorRef[n], point->u + dx, point->v + dy, wl);  // 在 第一帧 中的 灰度值

                    if (!std::isfinite(rlR) || !std::isfinite((float) hitColor[0]) || std::isnan((float) hitColor[1])) {
                        isGood = false;   
                        break;
                    }

                    float residual = hitColor[0] - vvr2new_aff[n][tocamnum][0] * rlR - vvr2new_aff[n][tocamnum][1];  // 计算光度误差 r = I2 - a21*I1 -b21 (I2 = a21*I1 + b21)   n 第一帧像素所在相机号 ， tocamnum 投影到当前帧 像素所在相机号
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);   // huber 核 限制误差较大时不会影响其他项 2次范数变为1次
                    energy += hw * residual * residual * (2 - hw);    // abs(r)<sigam :hw = 1 r^2    abs(r)>= sigma : hw = sigma/r  sigma/r*(2abs(r)-sigam/r) *r^2
    
                    // 求导
                    Eigen::Matrix<float, 2, 3> dxrdXs;
                    dxrdXs = computedStoR(xs, ys, zs, tocamnum, lvl);

// // NumericDiff 数值求导 -----test dxrdXs
                    // cout << "dxrdXs:"<<endl << dxrdXs <<endl;
                    // float xs1, ys1, zs1;
                    // float delta = 1e-6;
                    // Eigen::Matrix<float, 2, 3> dxrdXstest;

                    // xs1 = xs + delta;
                    // double Ku1, Kv1;
                    // int nnn;
                    // UMF->LadybugReprojectSpherePtToRectifyfixNum(xs1, ys, zs, n, &Ku1, &Kv1, lvl);
                    // dxrdXstest(0, 0) = (Ku1 - Ku)/delta;
                    // dxrdXstest(1, 0) = (Kv1 - Kv)/delta;

                    // ys1 = ys + delta;
                    // UMF->LadybugReprojectSpherePtToRectifyfixNum(xs, ys1, zs, n, &Ku1, &Kv1, lvl);
                    // dxrdXstest(0, 1) = (Ku1 - Ku)/delta;
                    // dxrdXstest(1, 1) = (Kv1 - Kv)/delta;

                    // zs1 = zs + delta;
                    // UMF->LadybugReprojectSpherePtToRectifyfixNum(xs, ys, zs1, n, &Ku1, &Kv1, lvl);
                    // dxrdXstest(0, 2) = (Ku1 - Ku)/delta;
                    // dxrdXstest(1, 2) = (Kv1 - Kv)/delta;

                    // cout << "dxrdXstest:"<<endl << dxrdXstest <<endl;

// //test end


                    Eigen::Matrix<float, 3, 6> dXsdpose;
                    Eigen::Matrix<float, 3, 1> dXsdrho1;
                    computedXs(point->idepth, R, t, pt, &dXsdpose, &dXsdrho1, X, Y, Z);  // 应该用 idpeth吧

// // NumericDiff 数值求导 -----test dXsdpose dXsdrho1
                    // cout << "dXsdpose:"<<endl << dXsdpose <<endl;
                    // // test
                    // Eigen::Matrix<float, 3, 6> dXsdposetest;
                    // Vec3f tnew1(t);
                    // tnew1(0) = tnew1(0) + 1e-6;
                    // Vec3f xstest1 = testJaccobian2(X, Y, Z, R, tnew1, point->idepth_new );
                    // dXsdposetest(0, 0) = (xstest1(0) - xs) / 1e-6;
                    // dXsdposetest(1, 0) = (xstest1(1) - ys) / 1e-6;
                    // dXsdposetest(2, 0) = (xstest1(2) - zs) / 1e-6;

                    // Vec3f tnew2(t);
                    // tnew2(1) = tnew2(1) + 1e-6;
                    // Vec3f xstest2 = testJaccobian2(X, Y, Z, R, tnew2, point->idepth_new );
                    // dXsdposetest(0, 1) = (xstest2(0) - xs) / 1e-6;
                    // dXsdposetest(1, 1) = (xstest2(1) - ys) / 1e-6;
                    // dXsdposetest(2, 1) = (xstest2(2) - zs) / 1e-6;

                    // Vec3f tnew3(t);
                    // tnew3(2) = tnew3(2) + 1e-6;
                    // Vec3f xstest3 = testJaccobian2(X, Y, Z, R, tnew3, point->idepth_new );
                    // dXsdposetest(0, 2) = (xstest3(0) - xs) / 1e-6;
                    // dXsdposetest(1, 2) = (xstest3(1) - ys) / 1e-6;
                    // dXsdposetest(2, 2) = (xstest3(2) - zs) / 1e-6;

                    // SO3 Rnew1(R.cast<double>());
                    // Eigen::Vector3d so31 = Rnew1.log();
                    // so31(0) = so31(0) + 1e-6;
                    // Vec3f xstest4 = testJaccobian2(X, Y, Z, Sophus::SO3::exp(so31).matrix().cast<float>(), t, point->idepth_new );
                    // dXsdposetest(0, 3) = (xstest4(0) - xs) / 1e-6;
                    // dXsdposetest(1, 3) = (xstest4(1) - ys) / 1e-6;
                    // dXsdposetest(2, 3) = (xstest4(2) - zs) / 1e-6;

                    // SO3 Rnew2(R.cast<double>());
                    // Eigen::Vector3d so32 = Rnew2.log();
                    // so32(1) = so32(1) + 1e-6;
                    // Vec3f xstest5 = testJaccobian2(X, Y, Z, Sophus::SO3::exp(so32).matrix().cast<float>(), t, point->idepth_new );
                    // dXsdposetest(0, 4) = (xstest5(0) - xs) / 1e-6;
                    // dXsdposetest(1, 4) = (xstest5(1) - ys) / 1e-6;
                    // dXsdposetest(2, 4) = (xstest5(2) - zs) / 1e-6;

                    // SO3 Rnew3(R.cast<double>());
                    // Eigen::Vector3d so33 = Rnew3.log();
                    // so33(2) = so33(2) + 1e-6;
                    // Vec3f xstest6 = testJaccobian2(X, Y, Z, Sophus::SO3::exp(so33).matrix().cast<float>(), t, point->idepth_new );
                    // dXsdposetest(0, 5) = (xstest6(0) - xs) / 1e-6;
                    // dXsdposetest(1, 5) = (xstest6(1) - ys) / 1e-6;
                    // dXsdposetest(2, 5) = (xstest6(2) - zs) / 1e-6;
                    // cout << "dXsdposetest:"<<endl << dXsdposetest <<endl;


                    // cout << "dXsdrho1:"<<endl << dXsdrho1 <<endl;
                    // Eigen::Matrix<float, 3, 1> dXsdrho1test;
                    // float idepthnew1= point->idepth_new;
                    // idepthnew1 = idepthnew1 + 1e-6;
                    // Vec3f xstest7 = testJaccobian2(X, Y, Z, R, t, idepthnew1 );
                    // dXsdrho1test(0, 0) = (xstest7(0) - xs) / 1e-6;
                    // dXsdrho1test(1, 0) = (xstest7(1) - ys) / 1e-6;
                    // dXsdrho1test(2, 0) = (xstest7(2) - zs) / 1e-6;

                    // cout << "dXsdrho1test:"<<endl << dXsdrho1test <<endl;


// //test end


                    if (hw < 1) hw = sqrtf(hw);        // 残差函数 f(x) =sqrt(hw) *r
                    float dxInterp = hw * hitColor[1];    // hw*gx
                    float dyInterp = hw * hitColor[2];    // hw*gy

                    Eigen::Matrix<float, 1, 6> dIdpose;
                    Eigen::Matrix<float, 1, 1> dIdrho1;
                    dIdpose = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdXs * dXsdpose;    // dI/dpose = dI/dxr * dxr/dXs * dXs/dpose
                    dIdrho1 = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdXs * dXsdrho1;    // dI/drho1 = dI/dxr * dxr/dXs * dXs/drho1


                    dp0[idx] = dIdpose(0,0);                // dp0, dp1, dp2, dp3, dp4, dp5 是光度误差对 se(3) 六个量的导数
                    dp1[idx] = dIdpose(0,1);          
                    dp2[idx] = dIdpose(0,2);   
                    dp3[idx] = dIdpose(0,3);
                    dp4[idx] = dIdpose(0,4);
                    dp5[idx] = dIdpose(0,5);
                    dp6[idx] = -hw * vvr2new_aff[n][tocamnum][0] * rlR;           // 优化对象是loga21=vvrefToNew_aff_current.a,然后vvr2new_aff[0]=exp(log21)   所以r21 = wh(I2 - exp(log21)*I1-b21),   dr21/dlog21 = -wh*exp(log21)*I1
                    dp7[idx] = -hw * 1;
                    dd[idx] = dIdrho1(0,0);   // h
                    r[idx] = hw * residual;   // 用于计算b

// NumericDiff 数值求导
                    // vector<float> dp, dpp;
                    // dp.resize(7);
                    // dpp.resize(7);
                    // // Eigen::Matrix<float, 1, 6> dIdposetest;
                    // // Eigen::Matrix<float, 1, 1> dIdrho1test;
                    // // dIdposetest = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdXstest * dXsdposetest;    // dI/dpose = dI/dxr * dxr/dXs * dXs/dpose
                    // // dIdrho1test = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdXstest * dXsdrho1test;
                    // dpp = NumericDiff(n, point->u + dx, point->v + dy, lvl, R, t, point->idepth_new, vvr2new_aff);

                    // cout << "dIdpose:      " << dp0[idx] << " " << dp1[idx] << " " << dp2[idx] << " " << dp3[idx] << " " << dp4[idx] << " " << dp5[idx] << " "<< dd[idx] << " " <<endl;
                    // // cout << "dIdposetest:  " << dIdposetest(0) << " " << dIdposetest(1) << " " << dIdposetest(2) << " " << dIdposetest(3) << " " << dIdposetest(4) << " " << dIdposetest(5) << " " << dIdrho1test(0) << " " <<endl;
                    // cout << "dIdposetest2: " << dpp[0] << " " << dpp[1] << " " << dpp[2] << " " << dpp[3] << " " << dpp[4] << " " << dpp[5] << " " << dpp[6] << " " <<endl;

                    // // // dp0[idx] = dp[0];                // dp0, dp1, dp2, dp3, dp4, dp5 是光度误差对 se(3) 六个量的导数
                    // // // dp1[idx] = dp[1];          
                    // // // dp2[idx] = dp[2];   
                    // // // dp3[idx] = dp[3];
                    // // // dp4[idx] = dp[4];
                    // // // dp5[idx] = dp[5];
                    //  dd[idx] = dpp[6];   // h

                    float idepthnew1 = point->idepth_new;
                    float idepthnew2 = point->idepth_new;
                    float da =1e-6; 
                    idepthnew1 = idepthnew1 + da;
                    idepthnew2 = idepthnew2 - da;
                    Vec5f r1, r2;
                    r1 = testJaccobian(n, point->u + dx, point->v + dy, lvl, R, t,idepthnew1, vvr2new_aff);
                    r2 = testJaccobian(n, point->u + dx, point->v + dy, lvl, R, t,idepthnew2, vvr2new_aff);
                    //float di 
                    // dp[6] = (r1- r2)/(2*da);
                    dd[idx] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);
                   

//test end

                    float maxstep = 1.0f / (dxrdXs * dXsdrho1).norm();
                    if (maxstep < point->maxstep) point->maxstep = maxstep;

                    // // immediately compute dp*dd' and dd*dd' in JbBuffer1.   用于舒尔布中消去所有点的逆深度参数的更新 Hab
                    vJbBuffer_new[n][i][0] += dp0[idx] * dd[idx];
                    vJbBuffer_new[n][i][1] += dp1[idx] * dd[idx];
                    vJbBuffer_new[n][i][2] += dp2[idx] * dd[idx];
                    vJbBuffer_new[n][i][3] += dp3[idx] * dd[idx];
                    vJbBuffer_new[n][i][4] += dp4[idx] * dd[idx];
                    vJbBuffer_new[n][i][5] += dp5[idx] * dd[idx];
                    vJbBuffer_new[n][i][6] += dp6[idx] * dd[idx];
                    vJbBuffer_new[n][i][7] += dp7[idx] * dd[idx];
                    vJbBuffer_new[n][i][8] += r[idx] * dd[idx];
                    vJbBuffer_new[n][i][9] += dd[idx] * dd[idx];

                    // 若 pattern tocamnum不同则说明在纠正有效影像边界 isGood设为false
                    if(idx == 0)
                        tocamnpa = tocamnum;
                    else
                    {
                        if(tocamnpa != tocamnum)
                        {    isGood = false;
                             break;
                        }
                    }

                    point->tocamnum = tocamnum;

                }
                
                if (!isGood || energy > point->outlierTH * 20) // 当点isGood == false，或者pattern 残差和大于阈值 则判定此点 isGood = false, 后续则不加入jacbobian更新
                {   
                    E.updateSingle((float) (point->energy[0]));
                    point->isGood_new = false;
                    point->energy_new = point->energy;
                    continue;
                }

                E.updateSingle(energy);
                point->isGood_new = true;
                point->energy_new[0] = energy;

                // LOG(INFO) << "updateSingle" << endl ;

                // update Hessian matrix.  
                // 计算前面4的整数倍 patternNum
                for (int i = 0; i + 3 < patternNum; i += 4)
                    acc57.updateSSE(
                            _mm_load_ps(((float *) (&dp0)) + i),
                            _mm_load_ps(((float *) (&dp1)) + i),
                            _mm_load_ps(((float *) (&dp2)) + i),
                            _mm_load_ps(((float *) (&dp3)) + i),
                            _mm_load_ps(((float *) (&dp4)) + i),
                            _mm_load_ps(((float *) (&dp5)) + i),
                            _mm_load_ps(((float *) (&dp6)) + i),
                            _mm_load_ps(((float *) (&dp7)) + i),
                            _mm_load_ps(((float *) (&r)) + i),
                            n, tocamnpa);

                // 计算多出4的整倍数的最后 patternNum
                for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
                    acc57.updateSingle(
                            (float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i],
                            (float) dp4[i], (float) dp5[i], (float) dp6[i], (float) dp7[i],
                            (float) r[i], n, tocamnpa);

                // LOG(INFO) << "updateSingle is over" << endl ;
            }

        }

        E.finish();
        acc57.finish();
        //LOG(INFO) << "calc pt is over" << endl ;
        Accumulator11 EAlpha;
        EAlpha.initialize();
        int sumnpts =  sumPoints[lvl];
        for(int n = 0; n < camNums; n++)
        {
            int npts = vnumPoints[lvl][n];
            Pnt *ptsl = vpoints[lvl][n];

            for (int i = 0; i < npts; i++) 
            {
                Pnt *point = ptsl + i;
                if (!point->isGood_new) {
                    EAlpha.updateSingle((float) (point->energy[1]));    // 点不好,用之前的
                } else {
                    point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
                    EAlpha.updateSingle((float) (point->energy_new[1]));
                }
            }
        }

        EAlpha.finish();
        float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * sumnpts);   // 平移越大, 越容易初始化成功?
        
        // compute alpha opt.   为了使尺度收敛增加两个约束项
        float alphaOpt;
        if (alphaEnergy > alphaK * sumnpts) 
        {
            alphaOpt = 0;                   // snapped 为 ture ，位移足够大   Epj = Epj + (dpi- dpiR)^2
            alphaEnergy = alphaK * sumnpts;
        } else 
        {
            alphaOpt = alphaW;              // snapped 为 false ，位移不足够大  Epj = Epj + alphaw*(dpi- 1)^2 + t* N
        }

        acc57SC.initialize();
        for(int n = 0; n < camNums; n++)
        {
            int npts = vnumPoints[lvl][n];
            Pnt *ptsl = vpoints[lvl][n];
            for (int i = 0; i < npts; i++) 
            {
                Pnt *point = ptsl + i;
                if (!point->isGood_new)
                    continue;

                point->lastHessian_new = vJbBuffer_new[n][i][9];

                vJbBuffer_new[n][i][8] += alphaOpt * (point->idepth_new - 1);
                vJbBuffer_new[n][i][9] += alphaOpt;

                if (alphaOpt == 0) {
                    vJbBuffer_new[n][i][8] += couplingWeight * (point->idepth_new - point->iR);
                    vJbBuffer_new[n][i][9] += couplingWeight;
                }

                vJbBuffer_new[n][i][9] = 1 / (1 + vJbBuffer_new[n][i][9]);   // 舒尔布部分  分母里多了一个1，猜测是为了防止JbBuffer_new[i][9]太小造成系统不稳定。
                //! dp*dd*(dd^2)^-1*dd*dp
                acc57SC.updateSingleWeighted(
                        (float) vJbBuffer_new[n][i][0], (float) vJbBuffer_new[n][i][1], (float) vJbBuffer_new[n][i][2],
                        (float) vJbBuffer_new[n][i][3],
                        (float) vJbBuffer_new[n][i][4], (float) vJbBuffer_new[n][i][5], (float) vJbBuffer_new[n][i][6],
                        (float) vJbBuffer_new[n][i][7],
                        (float) vJbBuffer_new[n][i][8], (float) vJbBuffer_new[n][i][9],
                        n, point->tocamnum);
            }
        }
        acc57SC.finish();

        H_out = acc57.H.topLeftCorner<56, 56>().cast<float>();// / acc57.num;H = JbWJb
        b_out = acc57.H.topRightCorner<56, 1>().cast<float>();// / acc57.num;b = JbWr
        H_out_sc = acc57SC.H.topLeftCorner<56, 56>().cast<float>();// / acc57.num;  H = JnewWJnew    Jnew = 1/Ja * (Ja^T*Jb)
        b_out_sc = acc57SC.H.topRightCorner<56, 1>().cast<float>();// / acc57.num; b = JnewWr

        H_out(0, 0) += alphaOpt * sumnpts;
        H_out(1, 1) += alphaOpt * sumnpts;
        H_out(2, 2) += alphaOpt * sumnpts;

        Vec3f tlog = refToNew.log().head<3>().cast<float>();    // 李代数, 平移部分 (上一次的位姿值)
        b_out[0] += tlog[0] * alphaOpt * sumnpts;
        b_out[1] += tlog[1] * alphaOpt * sumnpts;
        b_out[2] += tlog[2] * alphaOpt * sumnpts;

        return Vec3f(E.A, alphaEnergy, E.num);
    }

// for multi-fisheye Diff test drdT drdrho
    vector<float> CoarseInitializer::NumericDiff(int n, double x, double y,int lvl, Mat33f R, Vec3f t, float idepth, vector<vector<Eigen::Vector2f>> vvr2new_aff)
    {
// 中值求导
        // vector<float> dp;
        // dp.resize(7);
        // float da =1e-6; 
        // Vec5f r1, r2;
        // Vec3f tnew11(t),tnew12(t);
        // tnew11(0) = t(0) + da;
        // tnew12(0) = t(0) - da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew11,idepth, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, R, tnew12,idepth, vvr2new_aff);
        // //float dt0 
        // // dp[0] = (r1- r2)/(2*da);
        // dp[0] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // Vec3f tnew21(t),tnew22(t);
        // tnew21(1) = t(1) + da;
        // tnew22(1) = t(1) - da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew21,idepth, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, R, tnew22,idepth, vvr2new_aff);
        // //float dt1 
        // // dp[1] = (r1- r2)/(2*da);
        // dp[1] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // Vec3f tnew31(t),tnew32(t);
        // tnew31(2) = t(2) + da;
        // tnew32(2) = t(2) - da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew31,idepth, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, R, tnew32,idepth, vvr2new_aff);
        // //float dt2 
        // // dp[2] = (r1- r2)/(2*da);
        // dp[2] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // SO3 Rnew11(R.cast<double>()),Rnew12(R.cast<double>());
        // Eigen::Vector3d so311= Rnew11.log();
        // Eigen::Vector3d so312= Rnew12.log();
        // so311(0) = so311(0) + da;
        // so312(0) = so312(0) - da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so311).matrix().cast<float>(), t,idepth, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so312).matrix().cast<float>(), t,idepth, vvr2new_aff);
        // //float dr1 
        // // dp[3] = (r1- r2)/(2*da);
        // dp[3] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // SO3 Rnew21(R.cast<double>()),Rnew22(R.cast<double>());
        // Eigen::Vector3d so321= Rnew21.log();
        // Eigen::Vector3d so322= Rnew22.log();
        // so321(1) = so321(1) + da;
        // so322(1) = so322(1) - da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so321).matrix().cast<float>(), t,idepth, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so322).matrix().cast<float>(), t,idepth, vvr2new_aff);
        // //float dr2 
        // // dp[4] = (r1- r2)/(2*da);
        // dp[4] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // SO3 Rnew31(R.cast<double>()),Rnew32(R.cast<double>());
        // Eigen::Vector3d so331= Rnew31.log();
        // Eigen::Vector3d so332= Rnew32.log();
        // so331(2) = so331(2) + da;
        // so332(2) = so332(2) - da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so331).matrix().cast<float>(), t,idepth, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so332).matrix().cast<float>(), t,idepth, vvr2new_aff);
        // //float dr3 
        // // dp[5] = (r1- r2)/(2*da);
        // dp[5] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // float idepthnew1=idepth;
        // float idepthnew2=idepth;
        // idepthnew1 = idepthnew1 + da;
        // idepthnew2 = idepthnew2 - da;
        // r1 = testJaccobian(n,x, y, lvl, R, t,idepthnew1, vvr2new_aff);
        // r2 = testJaccobian(n,x, y, lvl, R, t,idepthnew2, vvr2new_aff);
        // //float di 
        // // dp[6] = (r1- r2)/(2*da);
        // dp[6] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

// // 前向求导
        Vec5f r = testJaccobian(n,x, y, lvl, R, t,idepth, vvr2new_aff);
        vector<float> dp;
        dp.resize(7);
        float da =1e-6; 
        Vec5f r1, r2;
        Vec3f tnew11(t);
        tnew11(0) = t(0) + da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew11,idepth, vvr2new_aff);
        //float dt0 
        dp[0] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        Vec3f tnew21(t);
        tnew21(1) = t(1) + da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew21,idepth, vvr2new_aff);
        //float dt1 
        //dp[1] = (r1- r)/da;
        dp[1] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        Vec3f tnew31(t);
        tnew31(2) = t(2) + da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew31,idepth, vvr2new_aff);
        //float dt2 
        // dp[2] = (r1- r)/da;
        dp[2] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        SO3 Rnew11(R.cast<double>());
        Eigen::Vector3d so311= Rnew11.log();
        so311(0) = so311(0) + da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so311).matrix().cast<float>(), t,idepth, vvr2new_aff);
        //float dr1 
        //dp[3] = (r1- r)/da;
        dp[3] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        SO3 Rnew21(R.cast<double>());
        Eigen::Vector3d so321= Rnew21.log();
        so321(1) = so321(1) + da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so321).matrix().cast<float>(), t,idepth, vvr2new_aff);
        //float dr2 
        // dp[4] = (r1- r)/da;
        dp[4] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        SO3 Rnew31(R.cast<double>());
        Eigen::Vector3d so331= Rnew31.log();
        so331(2) = so331(2) + da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so331).matrix().cast<float>(), t,idepth, vvr2new_aff);
        //float dr3 
        // dp[5] = (r1- r)/da;
        dp[5] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;


        float idepthnew1=idepth;
        idepthnew1 = idepthnew1 + da;
        r1 = testJaccobian(n,x, y, lvl, R, t,idepthnew1, vvr2new_aff);
        //float di 
        // dp[6] = (r1- r)/da;
        dp[6] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;
//
        return dp;
    }

    Vec5f CoarseInitializer::testJaccobian(int n, double RectifiedPixalx, double RectifiedPixaly, int lvl, Mat33f R, Vec3f t, float idepth, vector<vector<Eigen::Vector2f>> vvr2new_aff)
    {
        bool isGood = true;
        int wl = UMF->wPR[lvl], hl = UMF->hPR[lvl];

        vector<Eigen::Vector3f* > vcolorRef(firstFrame->vdIp[lvl]);
        vector<Eigen::Vector3f* > vcolorNew(newFrame->vdIp[lvl]);
        double X,Y,Z;
        //LOG(INFO) << "LadybugProjectFishEyePtToSphere" << endl ;
        //UMF->LadybugProjectFishEyePtToSphere(n, point->u + dx, point->v + dy, &X, &Y, &Z, lvl);   // [u,v] -> sphereX 鱼眼像素坐标转换到球面坐标(球面半径20)
        UMF->LadybugProjectRectifyPtToSphere(n, RectifiedPixalx, RectifiedPixaly, &X, &Y, &Z, lvl);

        float SphereRadius = UMF->GetSphereRadius();

        //Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) + t * idepth;  // R21( rho1^-1 * X归一 )+  t21  = rho1^-1 * (R21 * X归一 +  t21 * rho1)   这里取 R21 * X归一 +  t21 * rho1
        
        Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius)/idepth + t;  // R21( rho1^-1 * X归一 )+  t21
        
        float S_norm_pt = SphereRadius/pt.norm();

        float xs = S_norm_pt*pt(0);
        float ys = S_norm_pt*pt(1);
        float zs = S_norm_pt*pt(2);

        double Ku, Kv;
        int tocamnum;

        //UMF->LadybugReprojectSpherePtToFishEyeImg(xs, ys, zs, &tocamnum, &Ku, &Kv, lvl);
        UMF->LadybugReprojectSpherePtToRectifyfixNum(xs, ys, zs, n, &Ku, &Kv, lvl);
        tocamnum = n;
        //LOG(INFO) << "LadybugReprojectSpherePtToFishEyeImg" << endl ;

        //float new_idepth = point->idepth_new / pt[2];   // d1 * d2


        Vec3f hitColor = getInterpolatedElement33(vcolorNew[tocamnum], Ku, Kv, wl);    // 在当前第二帧 中的 灰度值
        // Vec3f hitColor = getInterpolatedElement33BiCub(vcolorNew[tocamnum], Ku, Kv, wl);

        //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
        float rlR = getInterpolatedElement31(vcolorRef[n], RectifiedPixalx, RectifiedPixaly, wl);  // 在 第一帧 中的 灰度值


        float residual = hitColor[0] - vvr2new_aff[n][tocamnum][0] * rlR - vvr2new_aff[n][tocamnum][1];  // 计算光度误差 r = I2 - a21*I1 -b21 (I2 = a21*I1 + b21)   n 第一帧像素所在相机号 ， tocamnum 投影到当前帧 像素所在相机号
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
        if (hw < 1) hw = sqrtf(hw);
        float r = hw * residual;
        float dxInterp = hw * hitColor[1];    // hw*gx
        float dyInterp = hw * hitColor[2];    // hw*gy
        float u = Ku;
        float v = Kv;
        Vec5f a;
        a << u, v, r, dxInterp, dyInterp;
        return a;
    }

    Vec3f CoarseInitializer::testJaccobian2(float X, float Y, float Z, Mat33f R, Vec3f t, float idepth)
    {
        float ms = UMF->GetSphereRadius();
        Vec3f pt = R * Vec3f(X/ms, Y/ms, Z/ms)/idepth + t;  // R21( rho1^-1 * X归一 )+  t21
        
        float S_norm_pt = 20/pt.norm();

        float xs = S_norm_pt*pt(0);
        float ys = S_norm_pt*pt(1);
        float zs = S_norm_pt*pt(2);

        return Vec3f(xs, ys, zs);
    }

    float CoarseInitializer::rescale() {
        float factor = 20 * thisToNext.translation().norm();
        return factor;
    }

    // AccumulatorX累计残差。
    Vec3f CoarseInitializer::calcEC(int lvl) {
        if (!snapped) return Vec3f(0, 0, numPoints[lvl]);
        AccumulatorX<2> E;
        E.initialize();
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            Pnt *point = points[lvl] + i;
            if (!point->isGood_new) continue;
            float rOld = (point->idepth - point->iR);
            float rNew = (point->idepth_new - point->iR);
            E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

        }
        E.finish();

        return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
    }

    // for multi-fisheye  计算旧的和新的逆深度与iR的差值, 返回旧的差, 新的差, 数目   iR是逆深度的均值，尺度收敛到IR
    Vec3f CoarseInitializer::calcECforMF(int lvl)
    {
    
         if (!snapped) return Vec3f(0, 0, sumPoints[lvl]);
        AccumulatorX<2> E;
        E.initialize();

        for(int n = 0; n < camNums; n++)
        {
            int npts = vnumPoints[lvl][n];
            for (int i = 0; i < npts; i++) {
                Pnt *point = vpoints[lvl][n] + i;
                if (!point->isGood_new) continue;
                float rOld = (point->idepth - point->iR);
                float rNew = (point->idepth_new - point->iR);
                E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

            }
        }
        E.finish();

        return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
    }

    void CoarseInitializer::optReg(int lvl) {
        int npts = numPoints[lvl];
        Pnt *ptsl = points[lvl];
        if (!snapped) {
            for (int i = 0; i < npts; i++)
                ptsl[i].iR = 1;
            return;
        }

        for (int i = 0; i < npts; i++) {
            Pnt *point = ptsl + i;
            if (!point->isGood) continue;

            float idnn[10];
            int nnn = 0;
            for (int j = 0; j < 10; j++) {
                if (point->neighbours[j] == -1) continue;
                Pnt *other = ptsl + point->neighbours[j];
                if (!other->isGood) continue;
                idnn[nnn] = other->iR;
                nnn++;
            }

            if (nnn > 2) {
                std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
                point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
            }
        }

    }

    // for multi-fisheye 所有点的 iR 设置为其 neighbour 逆深度的中位数 ,smooth 相当于进行逆深度的平滑
    void CoarseInitializer::optRegforMF(int lvl)
    {
        for(int n = 0; n < camNums; n++)
        {
            int npts = vnumPoints[lvl][n];
            Pnt *ptsl = vpoints[lvl][n];
            //* 位移不足够则设置iR是1, 位移不足没有初始化成功
            if (!snapped) 
            {
                for (int i = 0; i < npts; i++)
                    ptsl[i].iR = 1;
                return;
            }

            for (int i = 0; i < npts; i++) 
            {
                Pnt *point = ptsl + i;
                if (!point->isGood) continue;

                float idnn[10];
                int nnn = 0;
                // 获得当前点周围最近10个点, 质量好的点的iR
                for (int j = 0; j < 10; j++) 
                {
                    if (point->neighbours[j] == -1) continue;
                    Pnt *other = ptsl + point->neighbours[j];
                    if (!other->isGood) continue;
                    idnn[nnn] = other->iR;
                    nnn++;
                }
                // 与最近点中位数进行加权获得新的iR
                if (nnn > 2) 
                {
                    std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); // 获取中位数
                    point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];    // reweigt = 0.8
                }
            }

        }
        
    }


    void CoarseInitializer::propagateUp(int srcLvl) {
        assert(srcLvl + 1 < pyrLevelsUsed);
        // set idepth of target

        int nptss = numPoints[srcLvl];
        int nptst = numPoints[srcLvl + 1];
        Pnt *ptss = points[srcLvl];
        Pnt *ptst = points[srcLvl + 1];

        // set to zero.
        for (int i = 0; i < nptst; i++) {
            Pnt *parent = ptst + i;
            parent->iR = 0;
            parent->iRSumNum = 0;
        }

        for (int i = 0; i < nptss; i++) {
            Pnt *point = ptss + i;
            if (!point->isGood) continue;

            Pnt *parent = ptst + point->parent;
            parent->iR += point->iR * point->lastHessian;
            parent->iRSumNum += point->lastHessian;
        }

        for (int i = 0; i < nptst; i++) {
            Pnt *parent = ptst + i;
            if (parent->iRSumNum > 0) {
                parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
                parent->isGood = true;
            }
        }

        optReg(srcLvl + 1);
    }

    void CoarseInitializer::propagateUpforMF(int srcLvl)
    {
        assert(srcLvl + 1 < pyrLevelsUsed);
        // set idepth of target
        for(int n = 0; n < camNums; n++)
        {
            int nptss = vnumPoints[srcLvl][n];
            int nptst = vnumPoints[srcLvl + 1][n];
            Pnt *ptss = vpoints[srcLvl][n];
            Pnt *ptst = vpoints[srcLvl + 1][n];

            // set to zero.
            for (int i = 0; i < nptst; i++) 
            {
                Pnt *parent = ptst + i;
                parent->iR = 0;
                parent->iRSumNum = 0;
            }

            for (int i = 0; i < nptss; i++) 
            {
                Pnt *point = ptss + i;
                if (!point->isGood) continue;

                Pnt *parent = ptst + point->parent;
                parent->iR += point->iR * point->lastHessian;   //! 均值*信息矩阵 ∑ (sigma*u)
                parent->iRSumNum += point->lastHessian;         //! 新的信息矩阵 ∑ sigma
            }

            for (int i = 0; i < nptst; i++) 
            {
                Pnt *parent = ptst + i;
                if (parent->iRSumNum > 0) 
                {
                    parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);  //! 高斯归一化积后的均值
                    parent->isGood = true;
                }
            }
        }
        optRegforMF(srcLvl + 1);
    }

    void CoarseInitializer::propagateDown(int srcLvl) {
        assert(srcLvl > 0);
        // set idepth of target

        int nptst = numPoints[srcLvl - 1];
        Pnt *ptss = points[srcLvl];
        Pnt *ptst = points[srcLvl - 1];

        for (int i = 0; i < nptst; i++) {
            Pnt *point = ptst + i;
            Pnt *parent = ptss + point->parent;

            if (!parent->isGood || parent->lastHessian < 0.1) continue;
            if (!point->isGood) {
                point->iR = point->idepth = point->idepth_new = parent->iR;
                point->isGood = true;
                point->lastHessian = 0;
            } else {
                float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) /
                              (point->lastHessian * 2 + parent->lastHessian);
                point->iR = point->idepth = point->idepth_new = newiR;
            }
        }
        optReg(srcLvl - 1);
    }

    // for multi-fisheye image
    void CoarseInitializer::propagateDownforMF(int srcLvl)
    {
        assert(srcLvl > 0);
        // set idepth of target
        for (int n = 0; n < camNums; n++)
        {
            int nptst = vnumPoints[srcLvl - 1][n];  // 当前层
            Pnt *ptss = vpoints[srcLvl][n];
            Pnt *ptst = vpoints[srcLvl - 1][n];

            for (int i = 0; i < nptst; i++) 
            {
                Pnt *point = ptst + i;
                Pnt *parent = ptss + point->parent;

                if (!parent->isGood || parent->lastHessian < 0.1) continue;
                if (!point->isGood) 
                {
                    point->iR = point->idepth = point->idepth_new = parent->iR;   //当前点不好，上一层parent节点深度赋值给当前点
                    point->isGood = true;
                    point->lastHessian = 0;
                } 
                else 
                {
                    // 通过hessian给point和parent加权求得新的iR
			        // iR可以看做是深度的值, 使用的高斯归一化积, Hessian是信息矩阵
                    float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) /
                                (point->lastHessian * 2 + parent->lastHessian);
                    point->iR = point->idepth = point->idepth_new = newiR;   //iR 变量相当于是逆深度的真值
                }
            }
        }

        optRegforMF(srcLvl - 1);    // 当前帧
        
        
    }

    void CoarseInitializer::makeGradients(Eigen::Vector3f **data) {
        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            Eigen::Vector3f *dINew_l = data[lvl];
            Eigen::Vector3f *dINew_lm = data[lvlm1];

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++)
                    dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                                      dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
                                                      dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

            for (int idx = wl; idx < wl * (hl - 1); idx++) {
                dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
                dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
            }
        }
    }

    void CoarseInitializer::setFirst(shared_ptr<CalibHessian> HCalib, shared_ptr<FrameHessian> newFrameHessian) {

        makeK(HCalib);  // 生成图像金字塔的内参,逆内参
        firstFrame = newFrameHessian;

        PixelSelector sel(w[0], h[0]);

        float *statusMap = new float[w[0] * h[0]];
        bool *statusMapB = new bool[w[0] * h[0]];

        float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            sel.currentPotential = 3;
            int npts;
            if (lvl == 0) {
                npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);  //makemaps在setFirst和makeNewTraces的时候用到
            } else {
                npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);  //对于高层，选点直接是grid中梯度最大的点makePixelStatus，只在setFirst的时候用
            }

            if (points[lvl] != 0) delete[] points[lvl];
            points[lvl] = new Pnt[npts];

            // set idepth map to initially 1 everywhere. 初始化点的参数，其中逆深度初始化为1；
            int wl = w[lvl], hl = h[lvl];
            Pnt *pl = points[lvl];
            int nl = 0;
            for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
                for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) {
                    if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) {
                        //assert(patternNum==9);
                        pl[nl].u = x + 0.1;
                        pl[nl].v = y + 0.1;
                        pl[nl].idepth = 1;
                        pl[nl].iR = 1;
                        pl[nl].isGood = true;
                        pl[nl].energy.setZero();
                        pl[nl].lastHessian = 0;
                        pl[nl].lastHessian_new = 0;
                        pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

                        Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
                        float sumGrad2 = 0;
                        for (int idx = 0; idx < patternNum; idx++) {
                            int dx = patternP[idx][0];
                            int dy = patternP[idx][1];
                            float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
                            sumGrad2 += absgrad;
                        }

                        pl[nl].outlierTH = patternNum * setting_outlierTH;

                        nl++;
                        assert(nl <= npts);
                    }
                }


            numPoints[lvl] = nl;
        }
        delete[] statusMap;
        delete[] statusMapB;

        makeNN();

        thisToNext = SE3();
        snapped = false;
        frameID = snappedAt = 0;

        for (int i = 0; i < pyrLevelsUsed; i++)
            dGrads[i].setZero();

    }

    // for multi-fisheye
    void CoarseInitializer::setFirstforMF(shared_ptr<FrameHessian> newFrameHessian)
    {
        firstFrame = newFrameHessian;

        int* w = UMF->wPR;
        int* h = UMF->hPR;
        
        PixelSelector sel(w[0], h[0]);

        // float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
        float densities[] = {0.01, 0.02, 0.05, 0.1, 0.2};
        // float densities[] = {0.002, 0.005, 0.015, 0.025, 0.05};
        // float densities[] = {0.005, 0.01, 0.025, 0.05, 0.1};

        float *statusMap = new float[w[0]*h[0]];
        bool *statusMapB = new bool[w[0]*h[0]];

        // for(int n = 0; n < UMF->camNums; n++)
        // {
        //     float *mapmax0 = firstFrame->vabsSquaredGrad[0][n];

        //     ofstream outim;
        //     string path = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/vabsSquaredGrad_setFirstforMF_" + to_string(n) + ".txt";
        //     outim.open(path.c_str());

        //     int w = UMF->getOriginalSize()(0);
        //     int h = UMF->getOriginalSize()(1);

        //     for (int iw = 0; iw < w;  iw++)
        //         for (int ih = 0; ih < h;  ih++)
        //     {
        //         int idx = iw + ih * h;
        //         if(mapmax0[idx] < 0)
        //         {
        //             int aa = 0;
        //         }
                
        //         outim << iw << " " << ih << " " << mapmax0[idx] << " " << firstFrame->vabsSquaredGrad[0][n][idx] <<endl;
        //     }
        //     outim.close();
        // }
        

        for(int lvl = 0; lvl < pyrLevelsUsed; lvl++)
        {
            sel.currentPotential = 3;
            vpoints[lvl].resize(camNums);
            vnumPoints[lvl].resize(camNums);
            for(int n = 0; n < camNums; n++)
            {
                int npts;
                if(lvl == 0)
                {
                    npts = sel.makeMapsforMF(firstFrame, n, statusMap, densities[lvl] * w[0] * h[0], 1, false, 6);  //makemaps在setFirst和makeNewTraces的时候用到
                }
                else
                {
                    // npts = makePixelStatus(firstFrame->vdIp[lvl][n], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);  //对于高层，选点直接是grid中梯度最大的点makePixelStatus，只在setFirst的时候用
                    npts = makePixelStatus(firstFrame->vdIp[lvl][n], statusMapB, w[lvl], h[lvl], densities[lvl] * w[lvl] * h[lvl]);  //对于高层，选点直接是grid中梯度最大的点makePixelStatus，只在setFirst的时候用
                }

                if (vpoints[lvl][n] != 0) delete[] vpoints[lvl][n];
                vpoints[lvl][n] = new Pnt[npts];

                // set idepth map to initially 1 everywhere. 初始化点的参数，其中逆深度初始化为1；
                int wl = w[lvl], hl = h[lvl];
                Pnt *pl = vpoints[lvl][n];
                int nl = 0;
                for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
                    for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++) 
                    {
                        if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0)) 
                        {
                            //assert(patternNum==9);
                            pl[nl].u = x + 0.1;
                            pl[nl].v = y + 0.1;
                            pl[nl].idepth = 1;
                            pl[nl].iR = 1;
                            pl[nl].isGood = true;
                            pl[nl].energy.setZero();
                            pl[nl].lastHessian = 0;
                            pl[nl].lastHessian_new = 0;
                            pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];
                            pl[nl].hcamnum = n;
                            double X,Y,Z;
                            UMF->LadybugProjectRectifyPtToSphere(n, x + 0.1, y + 0.1, &X, &Y, &Z, lvl);
                            pl[nl].xs = X;
                            pl[nl].ys = Y;
                            pl[nl].zs = Z;

                            Eigen::Vector3f *cpt = firstFrame->vdIp[lvl][n] + x + y * w[lvl];
                            float sumGrad2 = 0;
                            for (int idx = 0; idx < patternNum; idx++) 
                            {
                                int dx = patternP[idx][0];
                                int dy = patternP[idx][1];
                                float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm(); //梯度均方根
                                sumGrad2 += absgrad;
                            }

                            pl[nl].outlierTH = patternNum * setting_outlierTH;

                            nl++;
                            assert(nl <= npts);
                        }
                    }

                vnumPoints[lvl][n] =nl;
                sumPoints[lvl] += nl;
            }
        }


        // // output points
        // string path = "/home/jiangfan/桌面/pan_dso_calib/pano_inital/pcxs.txt";
        // ofstream outp;
        // outp.open(path.c_str());
        // int id = 0;
        // for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        //      {
        //         for(int n = 0; n < camNums; n++)
        //         {
        //             int npts = vnumPoints[0][n];
        //             Pnt *ptsl = vpoints[0][n];
        //             for (int i = 0; i < npts; i++) 
        //             {
        //                 outp << id << " " <<  ptsl[i].xs << " " <<  ptsl[i].ys << " " <<  ptsl[i].zs << " " <<  ptsl[i].idepth << endl;
        //                 id ++ ;
        //             }
                    
        //         }
        //     }
        // outp.close();

        delete[] statusMap;
        delete[] statusMapB;

        makeNNforMF();


        thisToNext = SE3();
        snapped = false;
        frameID = snappedAt = 0;

        for(int i = 0; i < pyrLevelsUsed; i++)
        {
            vdGrads[i].resize(camNums);
            for (int n = 0; n < camNums; n++)
              vdGrads[i][n].setZero();  
        }

    }

    void CoarseInitializer::resetPoints(int lvl) {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            pts[i].energy.setZero();
            pts[i].idepth_new = pts[i].idepth;


            if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood) {
                float snd = 0, sn = 0;
                for (int n = 0; n < 10; n++) {
                    if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
                    snd += pts[pts[i].neighbours[n]].iR;
                    sn += 1;
                }

                if (sn > 0) {
                    pts[i].isGood = true;
                    pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;   // 将当前点的逆深度iR设为临近点的均值
                }
            }
        }
    }

    // for multi-fisheye 重置点的energy, idepth_new参数
    void CoarseInitializer::resetPointsforMF(int lvl)
    {
        for (int n = 0; n < camNums; n++)
        {
            Pnt *pts = vpoints[lvl][n];
            int npts = vnumPoints[lvl][n];
            for (int i = 0; i < npts; i++) 
            {
                Pnt pt = pts[i];
                pt.energy.setZero();
                pt.idepth_new = pt.idepth;
            
                // 如果是最顶层, 则使用周围点平均值来重置
                if (lvl == pyrLevelsUsed - 1 && !pt.isGood) 
                {
                    float snd = 0, sn = 0;
                    for (int n = 0; n < 10; n++) 
                    {
                        if (pt.neighbours[n] == -1 || !pts[pt.neighbours[n]].isGood) continue;
                        snd += pts[pt.neighbours[n]].iR;
                        sn += 1;
                    }

                    if (sn > 0) 
                    {
                        pt.isGood = true;
                        pt.iR = pt.idepth = pt.idepth_new = snd / sn;   // 将当前点的逆深度iR设为临近点的均值
                    }
                }
            }
        }
    }

    void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {

        const float maxPixelStep = 0.25;
        const float idMaxStep = 1e10;
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            if (!pts[i].isGood) continue;


            float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);   // ga + Hab *xb   ga = Ja^T * r
            float step = -b * JbBuffer[i][9] / (1 + lambda);      // x = Haa^-1 * b   


            float maxstep = maxPixelStep * pts[i].maxstep;
            if (maxstep > idMaxStep) maxstep = idMaxStep;

            if (step > maxstep) step = maxstep;
            if (step < -maxstep) step = -maxstep;

            float newIdepth = pts[i].idepth + step;
            if (newIdepth < 1e-3) newIdepth = 1e-3;
            if (newIdepth > 50) newIdepth = 50;
            pts[i].idepth_new = newIdepth;
        }

    }

    // for multi-fisheye 求出状态增量后, 计算被边缘化掉的逆深度, 更新逆深度
    void CoarseInitializer::doStepforMF(int lvl, float lambda, Vec56f inc)
    {
        const float maxPixelStep = 0.25;
        const float idMaxStep = 1e10;
        
        for(int n = 0 ; n < camNums; n++)
        {
            Pnt *pts = vpoints[lvl][n];
            int npts = vnumPoints[lvl][n];
            for (int i = 0; i < npts; i++) {
                if (!pts[i].isGood) continue;

                Vec8f tempinc;
                tempinc.head<6>() = inc.head<6>();
                // 取得对应两相机编号的光度仿射变换参数
                int tocamnum = pts[i].tocamnum;
                int abidx = (n * camNums + tocamnum) * 2 + 6;   // 计算参数位置
                tempinc[6] = inc[abidx];
                tempinc[7] = inc[abidx + 1];
                float b = vJbBuffer[n][i][8] + vJbBuffer[n][i].head<8>().dot(tempinc);   // ga + Hab *xb   ga = Ja^T * r
                float step = -b * vJbBuffer[n][i][9] / (1 + lambda);      // x = Haa^-1 * b   b = ga + Hab *xb


                float maxstep = maxPixelStep * pts[i].maxstep;
                if (maxstep > idMaxStep) maxstep = idMaxStep;

                if (step > maxstep) step = maxstep;
                if (step < -maxstep) step = -maxstep;

                float newIdepth = pts[i].idepth + step;
                if (newIdepth < 1e-3) newIdepth = 1e-3;
                if (newIdepth > 50) newIdepth = 50;
                pts[i].idepth_new = newIdepth;      // 得到优化完成后更新的idepth
                
            }
        }
        
    }

    void CoarseInitializer::applyStep(int lvl) {
        Pnt *pts = points[lvl];
        int npts = numPoints[lvl];
        for (int i = 0; i < npts; i++) {
            if (!pts[i].isGood) {
                pts[i].idepth = pts[i].idepth_new = pts[i].iR;
                continue;
            }
            pts[i].energy = pts[i].energy_new;
            pts[i].isGood = pts[i].isGood_new;
            pts[i].idepth = pts[i].idepth_new;
            pts[i].lastHessian = pts[i].lastHessian_new;
        }
        std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
    }

    void CoarseInitializer::applyStepforMF(int lvl)
    {
        for(int n = 0; n < camNums; n++)
        {
            Pnt *pts = vpoints[lvl][n];
            int npts = vnumPoints[lvl][n];
            for (int i = 0; i < npts; i++) 
            {
                if (!pts[i].isGood) 
                {
                    pts[i].idepth = pts[i].idepth_new = pts[i].iR;
                    continue;
                }
                // 优化成功后，本次优化得到的 **_new 赋值给原始值
                pts[i].energy = pts[i].energy_new;
                pts[i].isGood = pts[i].isGood_new;
                pts[i].idepth = pts[i].idepth_new;
                pts[i].lastHessian = pts[i].lastHessian_new;

            }
            std::swap<Vec10f *>(vJbBuffer, vJbBuffer_new);
        }
    }

    void CoarseInitializer::makeK(shared_ptr<CalibHessian> HCalib) {
        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;  // >>向右位移 相当于除以(2*level)
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {  // 逆内参
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }
    //  CoarseInitializer::makeNN 中计算每个点最邻近的10个点 neighbours，在上一层的最邻近点 parent
    void CoarseInitializer::makeNN() {
        const float NNDistFactor = 0.05;

        typedef nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
                FLANNPointcloud, 2> KDTree;

        // build indices
        FLANNPointcloud pcs[PYR_LEVELS];
        KDTree *indexes[PYR_LEVELS];
        for (int i = 0; i < pyrLevelsUsed; i++) {
            pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
            indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
            indexes[i]->buildIndex();
        }

        const int nn = 10;

        // find NN & parents
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            Pnt *pts = points[lvl];
            int npts = numPoints[lvl];

            int ret_index[nn];
            float ret_dist[nn];
            nanoflann::KNNResultSet<float, int, int> resultSet(nn);
            nanoflann::KNNResultSet<float, int, int> resultSet1(1);

            for (int i = 0; i < npts; i++) {
                //resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
                resultSet.init(ret_index, ret_dist);
                Vec2f pt = Vec2f(pts[i].u, pts[i].v);
                indexes[lvl]->findNeighbors(resultSet, (float *) &pt, nanoflann::SearchParams());
                int myidx = 0;
                float sumDF = 0;
                for (int k = 0; k < nn; k++) {
                    pts[i].neighbours[myidx] = ret_index[k];
                    float df = expf(-ret_dist[k] * NNDistFactor);
                    sumDF += df;
                    pts[i].neighboursDist[myidx] = df;
                    assert(ret_index[k] >= 0 && ret_index[k] < npts);
                    myidx++;
                }
                for (int k = 0; k < nn; k++)
                    pts[i].neighboursDist[k] *= 10 / sumDF;


                if (lvl < pyrLevelsUsed - 1) {
                    resultSet1.init(ret_index, ret_dist);
                    pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
                    indexes[lvl + 1]->findNeighbors(resultSet1, (float *) &pt, nanoflann::SearchParams());

                    pts[i].parent = ret_index[0];
                    pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

                    assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
                } else {
                    pts[i].parent = -1;
                    pts[i].parentDist = -1;
                }
            }
        }
        // done.

        for (int i = 0; i < pyrLevelsUsed; i++)
            delete indexes[i];
    }

    // for multi-fisheye image  CoarseInitializer::makeNN 中计算每个点最邻近的10个点 neighbours，在上一层的最邻近点 parent
    void CoarseInitializer::makeNNforMF() 
    {
        const float NNDistFactor = 0.05;

        typedef nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
                FLANNPointcloud, 2> KDTree;

        // build indices
        vector<FLANNPointcloud> vpcs[PYR_LEVELS];
        vector<KDTree *> vindexes[PYR_LEVELS];
        for (int i = 0; i < pyrLevelsUsed; i++) 
        {
            vpcs[i].resize(camNums);
            vindexes[i].resize(camNums);
            for (int n = 0; n < camNums; n++)
            {
                vpcs[i][n] = FLANNPointcloud(vnumPoints[i][n], vpoints[i][n]);
                vindexes[i][n] = new KDTree(2, vpcs[i][n], nanoflann::KDTreeSingleIndexAdaptorParams(5));
                vindexes[i][n]->buildIndex();
            }
            
        }

        const int nn = 10;

        // find NN & parents
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        {
            for (int n = 0; n < camNums; n++)
            {
                Pnt *pts = vpoints[lvl][n];
                int npts = vnumPoints[lvl][n];

                int ret_index[nn];
                float ret_dist[nn];
                nanoflann::KNNResultSet<float, int, int> resultSet(nn);
                nanoflann::KNNResultSet<float, int, int> resultSet1(1);

                for (int i = 0; i < npts; i++) 
                {
                    //resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
                    resultSet.init(ret_index, ret_dist);
                    Vec2f pt = Vec2f(pts[i].u, pts[i].v);
                    vindexes[lvl][n]->findNeighbors(resultSet, (float *) &pt, nanoflann::SearchParams());
                    int myidx = 0;
                    float sumDF = 0;
                    for (int k = 0; k < nn; k++) {
                        pts[i].neighbours[myidx] = ret_index[k];
                        float df = expf(-ret_dist[k] * NNDistFactor);
                        sumDF += df;
                        pts[i].neighboursDist[myidx] = df;
                        assert(ret_index[k] >= 0 && ret_index[k] < npts);
                        myidx++;
                    }
                    for (int k = 0; k < nn; k++)
                        pts[i].neighboursDist[k] *= 10 / sumDF;


                    if (lvl < pyrLevelsUsed - 1) 
                    {
                        resultSet1.init(ret_index, ret_dist);
                        pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
                        vindexes[lvl + 1][n]->findNeighbors(resultSet1, (float *) &pt, nanoflann::SearchParams());

                        pts[i].parent = ret_index[0];
                        pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

                        assert(ret_index[0] >= 0 && ret_index[0] < vnumPoints[lvl + 1][n]);
                    } else 
                    {
                        pts[i].parent = -1;
                        pts[i].parentDist = -1;
                    }
                }
            }
            
        }
        // done.

        for (int i = 0; i < pyrLevelsUsed; i++)
            for (int n = 0; n < camNums; n++)
                delete vindexes[i][n];
    }
}