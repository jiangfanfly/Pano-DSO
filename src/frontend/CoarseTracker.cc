#include "Feature.h"
#include "Point.h"

#include "frontend/CoarseTracker.h"
#include "internal/GlobalCalib.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/CalibHessian.h"
#include "internal/GlobalFuncs.h"

namespace ldso {

    /**
     * Create an aligned array and send their pointer into rawPtrVec
     * @tparam b bit of each element
     * @tparam T type of allocated buffer
     * @param size size of the buffer
     * @param rawPtrVec allocated pointer buffer, will be deleted after
     * @return allocated pointer
     */
    template<int b, typename T>
    T *allocAligned(int size, std::vector<T *> &rawPtrVec) {
        const int padT = 1 + ((1 << b) / sizeof(T));
        T *ptr = new T[size + padT];
        rawPtrVec.push_back(ptr);
        T *alignedPtr = (T *) ((((uintptr_t) (ptr + padT)) >> b) << b);
        return alignedPtr;
    }

    CoarseTracker::CoarseTracker(int ww, int hh) {

        // make coarse tracking templates.
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            int wl = ww >> lvl;
            int hl = hh >> lvl;

            idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            weightSums[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            weightSums_bak[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

            pc_u[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_v[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
            pc_color[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

        }

        // warped buffers
        buf_warped_idepth = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_u = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_v = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_dx = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_dy = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_residual = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_weight = allocAligned<4, float>(ww * hh, ptrToDelete);
        buf_warped_refColor = allocAligned<4, float>(ww * hh, ptrToDelete);

        w[0] = h[0] = 0;
    }

    CoarseTracker::CoarseTracker(UndistortMultiFisheye* uMF)
    {
        UMF = uMF;
        // make coarse tracking templates.
        int camnums = UMF->camNums;
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        {
            int wl = UMF->wPR[lvl];
            int hl = UMF->hPR[lvl];
            w[lvl] = wl;
            h[lvl] = hl;

            videpth[lvl].resize(camnums);
            vweightSums[lvl].resize(camnums);
            vweightSums_bak[lvl].resize(camnums);
            vpc_u[lvl].resize(camnums);
            vpc_v[lvl].resize(camnums);
            vpc_idepth[lvl].resize(camnums);
            vpc_color[lvl].resize(camnums);
            vpc_n[lvl].resize(camnums);

            for(int n = 0; n < camnums; n++)
            {
                videpth[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);
                vweightSums[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);
                vweightSums_bak[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);

                vpc_u[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);
                vpc_v[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);
                vpc_idepth[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);
                vpc_color[lvl][n] = allocAligned<4, float>(wl * hl, ptrToDelete);
            }

        }

        // warped buffers
        buf_warped_idepth = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_fromcamnum = allocAligned<4, int>(w[0] * w[0] * camnums, ptrToDeletei);
        buf_warped_tocamnum = allocAligned<4, int>(w[0] * w[0] * camnums, ptrToDeletei);
        buf_warped_a = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_b = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_b0 = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_xsT = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_ysT = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_zsT = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_xsF = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_ysF = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_zsF = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_u = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_v = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_dx = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_dy = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_residual = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_weight = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);
        buf_warped_refColor = allocAligned<4, float>(w[0] * w[0] * camnums, ptrToDelete);

    }

    bool CoarseTracker::trackNewestCoarse(
            shared_ptr<FrameHessian> newFrameHessian, SE3 &lastToNew_out,
            AffLight &aff_g2l_out, int coarsestLvl, Vec5 minResForAbort) {

        assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

        lastResiduals.setConstant(NAN);
        lastFlowIndicators.setConstant(1000);

        newFrame = newFrameHessian;

        // iteration in each pyramid level
        int maxIterations[] = {10, 20, 50, 50, 50};
        float lambdaExtrapolationLimit = 0.001;

        // use last track results as an initial guess
        SE3 refToNew_current = lastToNew_out;
        AffLight aff_g2l_current = aff_g2l_out;

        bool haveRepeated = false;

        // coarse-to-fine
        for (int lvl = coarsestLvl; lvl >= 0; lvl--) {
            Mat88 H;
            Vec8 b;
            float levelCutoffRepeat = 1;

            // compute the residual and adjust the huber threshold  
            Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);
            while (resOld[5] > 0.6 && levelCutoffRepeat < 50) {
                // more than 60% is over than threshold, then increate the cut off threshold  阈值设置太大，太多点残差超过阈值
                levelCutoffRepeat *= 2;
                resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);
            }

            // Compute H and b
            calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

            float lambda = 0.01;

            // relAff no used?
            //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
            //                                           aff_g2l_current).cast<float>();

            // L-M iteration
            for (int iteration = 0; iteration < maxIterations[lvl]; iteration++) {
                Mat88 Hl = H;
                for (int i = 0; i < 8; i++) Hl(i, i) *= (1 + lambda);
                Vec8 inc = Hl.ldlt().solve(-b);

                // depends on the mode, if a,b is fixed, don't estimate them
                if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0)    // fix a, b
                {
                    inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
                    inc.tail<2>().setZero();
                }
                if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)    // fix b
                {
                    inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
                    inc.tail<1>().setZero();
                }
                if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))    // fix a  
                {
                    Mat88 HlStitch = Hl;
                    Vec8 bStitch = b;
                    HlStitch.col(6) = HlStitch.col(7);
                    HlStitch.row(6) = HlStitch.row(7);
                    bStitch[6] = bStitch[7];
                    Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
                    inc.setZero();
                    inc.head<6>() = incStitch.head<6>();
                    inc[6] = 0;
                    inc[7] = incStitch[6];
                }

                float extrapFac = 1;
                if (lambda < lambdaExtrapolationLimit)
                    extrapFac = sqrtf(sqrt(lambdaExtrapolationLimit / lambda));
                inc *= extrapFac;

                Vec8 incScaled = inc;
                incScaled.segment<3>(0) *= SCALE_XI_ROT;
                incScaled.segment<3>(3) *= SCALE_XI_TRANS;
                incScaled.segment<1>(6) *= SCALE_A;
                incScaled.segment<1>(7) *= SCALE_B;

                if (!std::isfinite(incScaled.sum())) incScaled.setZero();

                // left multiply the pose and add to a,b
                SE3 refToNew_new = SE3::exp((Vec6) (incScaled.head<6>())) * refToNew_current;
                AffLight aff_g2l_new = aff_g2l_current;
                aff_g2l_new.a += incScaled[6];
                aff_g2l_new.b += incScaled[7];

                // calculate new residual after this update step
                Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

                // decide whether to accept this step
                // res[0]/res[1] is the average energy
                bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

                // relAff no used?
                //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                //                                           lastRef_aff_g2l, aff_g2l_new).cast<float>();
                if (accept) {

                    // decrease lambda
                    calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
                    resOld = resNew;
                    aff_g2l_current = aff_g2l_new;
                    refToNew_current = refToNew_new;
                    lambda *= 0.5;                  // 如果优化成功 lamda设为原来的0.5
                } else {
                    // increase lambda in LM   若失败，lamda设为原来的4倍
                    lambda *= 4;
                    if (lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
                }

                // terminate if increment is small
                if (!(inc.norm() > 1e-3)) {
                    break;
                }
            } // end of L-M iteration

            // set last residual for that level, as well as flow indicators.
            lastResiduals[lvl] = sqrtf((float) (resOld[0] / resOld[1]));   // 平均残差   开方
            lastFlowIndicators = resOld.segment<3>(2);
            if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
                return false;

            // repeat this level level
            if (levelCutoffRepeat > 1 && !haveRepeated) {
                lvl++;
                haveRepeated = true;
            }
        } // end of for: pyramid level

        // set!
        lastToNew_out = refToNew_current;
        aff_g2l_out = aff_g2l_current;

        if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
            || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
            return false;

        Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
                                                   aff_g2l_out).cast<float>();

        if ((setting_affineOptModeA == 0 && (fabsf(logf((float) relAff[0])) > 1.5))
            || (setting_affineOptModeB == 0 && (fabsf((float) relAff[1]) > 200)))
            return false;

        if (setting_affineOptModeA < 0) aff_g2l_out.a = 0;
        if (setting_affineOptModeB < 0) aff_g2l_out.b = 0;

        return true;
    }

    // for mult-fisheye
    bool CoarseTracker::trackNewestCoarseforMF(shared_ptr<FrameHessian> newFrameHessian, SE3 &lastToNew_out, vector<AffLight> &vaff_g2l_out, int coarsestLvl, Vec5 minResForAbort)
    {
        assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

        lastResiduals.setConstant(NAN);
        lastFlowIndicators.setConstant(1000);

        newFrame = newFrameHessian;

        // iteration in each pyramid level
        int maxIterations[] = {10, 20, 50, 50, 50};
        float lambdaExtrapolationLimit = 0.001;

        // use last track results as an initial guess
        SE3 refToNew_current = lastToNew_out;       // 优化初值 前一帧到当前帧
        vector<AffLight> vaff_g2l_current = vaff_g2l_out;
        vector<vector<Vec2f>> vvaffLL;
        int camnums = UMF->camNums;
        vvaffLL.resize(camnums);
        for(int n1 = 0; n1 < camnums; n1++)
        {
            vvaffLL[n1].resize(camnums);
            for(int n2 = 0; n2 < camnums; n2++)
            {
                // 从 ref参考帧 到 当前帧
                vvaffLL[n1][n2] = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, vlastRef_aff_g2l[n1],
                                                vaff_g2l_current[n2]).cast<float>();
            }
            
        }

        bool haveRepeated = false;  // 是否重复计算

        vector<vector<Vec2f>> vvaff_new; 

        // coarse-to-fine   从金字塔顶层向下
        for (int lvl = coarsestLvl; lvl >= 0; lvl--)
        {
            Mat5656 H;
            Vec56 b;
            float levelCutoffRepeat = 1;  // 残差阈值
            // compute the residual and adjust the huber threshold 计算残差和雅克比, 保证最多60%残差大于阈值, 计算正规方程  
            // [0:残差和 1:总点数 2:平移-球面坐标移动距离 3:0 4:旋转平移-球面坐标移动距离 5:残差过大数量的百分比]
            Vec6 resOld = calcResforMF(lvl, refToNew_current, vvaffLL, setting_coarseCutoffTH * levelCutoffRepeat);     // setting_coarseCutoffTH: 20
            while (resOld[5] > 0.6 && levelCutoffRepeat < 50) 
            {
                // more than 60% is over than threshold, then increate the cut off threshold  
                // 超过阈值的多，超过60%, 则放大阈值重新计算，　对应后面迭代优化时,　迭代结束后再进行一次for迭代
                levelCutoffRepeat *= 2;
                resOld = calcResforMF(lvl, refToNew_current, vvaffLL, setting_coarseCutoffTH * levelCutoffRepeat);
            }

            // Compute H and b
            calcGSSSEforMF(lvl, H, b, refToNew_current);

            float lambda = 0.01;

            // L-M iteration
            for (int iteration = 0; iteration < maxIterations[lvl]; iteration++)
            {
                Mat5656 Hl = H;
                for(int i = 0; i < 56; i++)
                {
                    Hl(i, i) *= (1 + lambda); 
                }
                Vec56 inc = Hl.ldlt().solve(-b);

                // // depends on the mode, if a,b is fixed, don't estimate them
                // if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0)    // fix a, b
                // {
                //     inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
                //     inc.tail<50>().setZero();
                // }
                // if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)    // fix b
                // {
                //     inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
                //     inc.tail<1>().setZero();
                // }
                // if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))    // fix a  
                // {
                //     Mat5656 HlStitch = Hl;
                //     Vec56 bStitch = b;
                //     HlStitch.col(6) = HlStitch.col(7);
                //     HlStitch.row(6) = HlStitch.row(7);
                //     bStitch[6] = bStitch[7];
                //     Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
                //     inc.setZero();
                //     inc.head<6>() = incStitch.head<6>();
                //     inc[6] = 0;
                //     inc[7] = incStitch[6];
                // }

                float extrapFac = 1;
                if (lambda < lambdaExtrapolationLimit)
                    extrapFac = sqrtf(sqrt(lambdaExtrapolationLimit / lambda));
                inc *= extrapFac;

                Vec56 incScaled = inc;
                incScaled.segment<3>(0) *= SCALE_XI_TRANS;
                incScaled.segment<3>(3) *= SCALE_XI_ROT;


                for(int n = 0; n < UMF->camNums * UMF->camNums; n++)
                {
                    incScaled.segment<1>(6 + n*2) *= SCALE_A;
                    incScaled.segment<1>(7 + n*2) *= SCALE_B;
                }

                if (!std::isfinite(incScaled.sum())) 
                    incScaled.setZero();

                // left multiply the pose and add to a,b 
                SE3 refToNew_new = SE3::exp((Vec6) (incScaled.head<6>())) * refToNew_current;
                vvaff_new = vvaffLL;
                vector<AffLight> vaff_g2l_new = vaff_g2l_current;
                for(int n1 = 0; n1 < UMF->camNums; n1++)
                {
                    for(int n2 = 0; n2 < UMF->camNums; n2++)
                    {
                        int idx = 2*(n1 * UMF->camNums + n2);
                        vvaff_new[n1][n2][0] += incScaled[6 + idx];
                        vvaff_new[n1][n2][1] += incScaled[7 + idx];
                    }
                    // 严格应该用 5*5个参数对应去更新相应的 ab
                    int idx2 = 2*(n1 * UMF->camNums + n1);
                    vaff_g2l_new[n1].a += incScaled[6 + idx2];
                    vaff_g2l_new[n1].b += incScaled[7 + idx2];
                    
                }

                // calculate new residual after this update step
                Vec6 resNew = calcResforMF(lvl, refToNew_new, vvaff_new, setting_coarseCutoffTH * levelCutoffRepeat);

                // decide whether to accept this step
                // res[0]/res[1] is the average energy
                bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

                printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t \n",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());

                // relAff no used?
                //Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                //                                           lastRef_aff_g2l, aff_g2l_new).cast<float>();
                if (accept) {

                    // decrease lambda
                    calcGSSSEforMF(lvl, H, b, refToNew_new);
                    resOld = resNew;
                    vvaffLL = vvaff_new;
                    vaff_g2l_current = vaff_g2l_new;
                    refToNew_current = refToNew_new;
                    lambda *= 0.5;                  // 如果优化成功 lamda设为原来的0.5
                } else {
                    // increase lambda in LM   若失败，lamda设为原来的4倍
                    lambda *= 4;
                    if (lambda < lambdaExtrapolationLimit) 
                        lambda = lambdaExtrapolationLimit;
                }

                // terminate if increment is small
                if (!(inc.norm() > 1e-3)) {
                    printf("inc too small, break!\n");
                    break;
                }

            }

            // set last residual for that level, as well as flow indicators.
            lastResiduals[lvl] = sqrtf((float) (resOld[0] / resOld[1]));   // 平均残差   开方
            lastFlowIndicators = resOld.segment<3>(2);
            // 某层　残差大于之前0层残差最小时所保存的各层残差中对应　此层残差，则直接返回false
            if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
                return false;

            // repeat this level level
            if (levelCutoffRepeat > 1 && !haveRepeated) 
            {
                lvl++;
                haveRepeated = true;
                printf("REPEAT LEVEL!\n");
            }
        }

         // set!
        lastToNew_out = refToNew_current;
        vaff_g2l_out = vaff_g2l_current;

        float avg2l_a = 0;
        float avg2l_b = 0;
        for(int n1 = 0; n1 < UMF->camNums; n1++)
        {
            avg2l_a += fabsf(vaff_g2l_out[n1].a);
            avg2l_b += fabsf(vaff_g2l_out[n1].b);
        }

        if ((setting_affineOptModeA != 0 && (avg2l_a/UMF->camNums > 1.2))
            || (setting_affineOptModeB != 0 && (avg2l_b/UMF->camNums > 200)))
            return false;

        float relAff_a = 0;
        float relAff_b = 0;
        for(int n1 = 0; n1 < UMF->camNums; n1++)
        {
            for(int n2 = 0; n2 < UMF->camNums; n2++)
            {
                //int idx = 2*(n1 * UMF->camNums + n2);
                relAff_a += fabsf(logf(vvaff_new[n1][n2][0]));
                relAff_b += fabsf(vvaff_new[n1][n2][1]);
            }
        }

        if ((setting_affineOptModeA == 0 && (relAff_a /(UMF->camNums * UMF->camNums) )> 1.5)
            || (setting_affineOptModeB == 0 && (relAff_b /(UMF->camNums * UMF->camNums) )> 200))
            return false;

        if (setting_affineOptModeA < 0)
        {
            for(int n = 0; n < UMF->camNums; n++)
            {
                    vaff_g2l_out[n].a = 0;
            }
        } 
        if (setting_affineOptModeB < 0) 
        {
            for(int n = 0; n < UMF->camNums; n++)
            {
                    vaff_g2l_out[n].b = 0;
            }
        }

        return true;
    }

    void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, shared_ptr<PangolinDSOViewer> wraps)
    {
        if(w[1] == 0) return;


        int lvl = 0;

        {
            std::vector<float> allID;
            for(int i=0;i<h[lvl]*w[lvl];i++)
            {
                if(idepth[lvl][i] > 0)
                    allID.push_back(idepth[lvl][i]);
            }
            std::sort(allID.begin(), allID.end());
            int n = allID.size()-1;

            float minID_new = allID[(int)(n*0.05)];
            float maxID_new = allID[(int)(n*0.95)];

            float minID, maxID;
            minID = minID_new;
            maxID = maxID_new;
            if(minID_pt!=0 && maxID_pt!=0)
            {
                if(*minID_pt < 0 || *maxID_pt < 0)
                {
                    *maxID_pt = maxID;
                    *minID_pt = minID;
                }
                else
                {

                    // slowly adapt: change by maximum 10% of old span.
                    float maxChange = 0.3*(*maxID_pt - *minID_pt);

                    if(minID < *minID_pt - maxChange)
                        minID = *minID_pt - maxChange;
                    if(minID > *minID_pt + maxChange)
                        minID = *minID_pt + maxChange;


                    if(maxID < *maxID_pt - maxChange)
                        maxID = *maxID_pt - maxChange;
                    if(maxID > *maxID_pt + maxChange)
                        maxID = *maxID_pt + maxChange;

                    *maxID_pt = maxID;
                    *minID_pt = minID;
                }
            }

            MinimalImageB3 mf(w[lvl], h[lvl]);
            mf.setBlack();
            for(int i=0;i<h[lvl]*w[lvl];i++)
            {
                int c = lastRef->dIp[lvl][i][0]*0.9f;
                if(c>255) c=255;
                mf.at(i) = Vec3b(c,c,c);
            }
            int wl = w[lvl];
            for(int y=3;y<h[lvl]-3;y++)
                for(int x=3;x<wl-3;x++)
                {
                    int idx=x+y*wl;
                    float sid=0, nid=0;
                    float* bp = idepth[lvl]+idx;

                    if(bp[0] > 0) {sid+=bp[0]; nid++;}
                    if(bp[1] > 0) {sid+=bp[1]; nid++;}
                    if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
                    if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
                    if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

                    if(bp[0] > 0 || nid >= 3)
                    {
                        float id = ((sid / nid)-minID) / ((maxID-minID));
                        mf.setPixelCirc(x,y,makeJet3B(id));
                        //mf.at(idx) = makeJet3B(id);
                    }
                }

            wraps->pushDepthImage(&mf);

            // if(debugSaveImages)
            // {
            //     char buf[1000];
            //     snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
            //     IOWrap::writeImage(buf,&mf);
            // }

        }
    }

    // for multi-fisheye
    void CoarseTracker::debugPlotIDepthMapforMF(float* minID_pt, float* maxID_pt, shared_ptr<PangolinDSOViewer> wraps)
    {
        if(w[1] == 0) return;
        int lvl = 0;

        {
            std::vector<float> allID;
            for(int i=0; i < h[lvl] * w[lvl]; i++)
            {
                for(int n = 0; n < UMF->camNums; n++)
                {
                    if(videpth[lvl][n][i] > 0)
                        allID.push_back(videpth[lvl][n][i]);
                }
            }
            std::sort(allID.begin(), allID.end());
            int n = allID.size()-1;

            float minID_new = allID[(int)(n*0.05)];
            float maxID_new = allID[(int)(n*0.95)];

            float minID, maxID;
            minID = minID_new;
            maxID = maxID_new;
            if(minID_pt!=0 && maxID_pt!=0)
            {
                if(*minID_pt < 0 || *maxID_pt < 0)
                {
                    *maxID_pt = maxID;
                    *minID_pt = minID;
                }
                else
                {

                    // slowly adapt: change by maximum 10% of old span.
                    float maxChange = 0.3*(*maxID_pt - *minID_pt);

                    if(minID < *minID_pt - maxChange)
                        minID = *minID_pt - maxChange;
                    if(minID > *minID_pt + maxChange)
                        minID = *minID_pt + maxChange;


                    if(maxID < *maxID_pt - maxChange)
                        maxID = *maxID_pt - maxChange;
                    if(maxID > *maxID_pt + maxChange)
                        maxID = *maxID_pt + maxChange;

                    *maxID_pt = maxID;
                    *minID_pt = minID;
                }
            }

            // output points
            // string path = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/pc.txt";
            // ofstream outp;
            // outp.open(path.c_str());
            int id = 0;
            

            vector<MinimalImageB3* > vmf;
            vmf.resize(UMF->camNums);
            for(int n = 0; n < UMF->camNums; n++)
            {
                vmf[n] = new MinimalImageB3(w[lvl], h[lvl]);
                vmf[n]->setBlack();
                for(int i = 0; i < h[lvl] * w[lvl]; i++)
                {
                    int c = lastRef->vdIp[lvl][n][i][0]*0.9f;
                    // int c = lastRef->vdfisheyeI[n][i]*0.9f;
                    if(c>255) 
                        c=255;
                    vmf[n]->at(i) = Vec3b(c,c,c);
                }
                int wl = w[lvl];
                int hl = h[lvl];
                for(int y = 3; y < hl-3; y++)
                    for(int x = 3; x < wl-3; x++)
                    {
                        int idx = x + y * wl;
                        float sid = 0, nid = 0;
                        float* bp = videpth[lvl][n] + idx;

                        if(bp[0] > 0) {sid+=bp[0]; nid++;}
                        if(bp[1] > 0) {sid+=bp[1]; nid++;}
                        if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
                        if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
                        if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

                        if(bp[0] > 0 || nid >= 3)
                        {
                            float id = ((sid / nid)-minID) / ((maxID-minID));
                            // double dx,dy;
                            // UMF->LadybugUnRectifyImage(n, x, y, &dx, &dy, 0);
                            // if(dx > wl-3 || dy > hl-3 || dx < 3 || dy < 3)
                            //     continue;
                            // vmf[n]->setPixelCirc(dx, dy, makeJet3B(id));
                            // double X,Y,Z;
                            // UMF->LadybugProjectRectifyPtToSphere(n, x, y, &X, &Y, &Z, lvl);
                            // //outp << idx << " " << X << " " << Y << " " << Z << " " << bp[0] << endl;

                            vmf[n]->setPixelCirc(x,y,makeJet3B(id));
                            //mf.at(idx) = makeJet3B(id);
                        }
                    }
                //vmf.emplace_back(mf);
                

                char buf[1000];
                snprintf(buf, 1000, "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/predicted_%05d_%05d_%01d.jpg", lastRef->frameID, refFrameID, n);
                cv::Mat imga(h[0],w[0],CV_8UC3);
                for(int y=0; y<h[0];y++)
                    for(int x=0;x<w[0];x++)
                    {
                        //float v = (!std::isfinite(fh->vdI[i][camNum][0]) || v>255) ? 255 : fh->vdI[i][camNum][0];
                        //imga.at<cv::Vec3b>(i) = cv::Vec3b(v, fh->vdI[i][camNum][1], fh->vdI[i][camNum][2]);
                        int i=x+y*w[0];
                        imga.at<cv::Vec3b>(y,x) = cv::Vec3b(vmf[n]->data[i][0], vmf[n]->data[i][1], vmf[n]->data[i][2]);
                    }
                cv::imwrite(buf, imga);
            }
            //outp.close();
        
            wraps->pushDepthImageforMF(vmf);

            //if(debugSaveImages)
            //{
                // for(int n = 0; n < UMF->camNums; n++)
                // {
                //     char buf[1000];
                //     snprintf(buf, 1000, "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/predicted_%05d_%05d_%01d.jpg", lastRef->frameID, refFrameID, n);
                //     cv::Mat imga(h[0],w[0],CV_8UC3);
                //     for(int y=0; y<h[0];y++)
                //         for(int x=0;x<w[0];x++)
                //         {
                //             //float v = (!std::isfinite(fh->vdI[i][camNum][0]) || v>255) ? 255 : fh->vdI[i][camNum][0];
                //             //imga.at<cv::Vec3b>(i) = cv::Vec3b(v, fh->vdI[i][camNum][1], fh->vdI[i][camNum][2]);
                //             int i=x+y*w[0];
                //             imga.at<cv::Vec3b>(y,x) = cv::Vec3b(vmf[n]->data[i][0], vmf[n]->data[i][1], vmf[n]->data[i][2]);
                //         }
                //     cv::imwrite(buf, imga);
                // }
            //}
        }
    }


    void CoarseTracker::makeK(shared_ptr<CalibHessian> HCalib) {

        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

    void CoarseTracker::setCoarseTrackingRef(std::vector<shared_ptr<FrameHessian>> &frameHessians) {

        assert(frameHessians.size() > 0);
        lastRef = frameHessians.back();
        makeCoarseDepthL0(frameHessians);
        refFrameID = lastRef->frame->id;
        lastRef_aff_g2l = lastRef->aff_g2l();
        firstCoarseRMSE = -1;
    }

    // for multi-fisheye
    void CoarseTracker::setCoarseTrackingRefforMF(std::vector<shared_ptr<FrameHessian>> &frameHessians) 
    {

        assert(frameHessians.size() > 0);
        lastRef = frameHessians.back();
        makeCoarseDepthL0forMF(frameHessians);          // 生成逆深度估值 使用在当前帧上投影的点的逆深度, 来生成每个金字塔层上点的逆深度值
        refFrameID = lastRef->frame->id;
        vlastRef_aff_g2l = lastRef->aff_g2lforMF();
        firstCoarseRMSE = -1;
    }

    void CoarseTracker::makeCoarseDepthL0(std::vector<shared_ptr<FrameHessian>> frameHessians) {

        // make coarse tracking templates for latstRef.
        memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
        memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

        for (shared_ptr<FrameHessian> fh: frameHessians) {
            for (shared_ptr<Feature> feat: fh->frame->features) {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                        shared_ptr<PointFrameResidual> r = ph->lastResiduals[0].first;
                        assert(r->isActive() && r->target.lock() == lastRef);
                        int u = r->centerProjectedTo[0] + 0.5f;
                        int v = r->centerProjectedTo[1] + 0.5f;
                        float new_idepth = r->centerProjectedTo[2];
                        float weight = sqrtf(1e-3 / (ph->HdiF + 1e-12));

                        idepth[0][u + w[0] * v] += new_idepth * weight;
                        weightSums[0][u + w[0] * v] += weight;
                    }
                }
            }
        }

        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            float *idepth_l = idepth[lvl];
            float *weightSums_l = weightSums[lvl];

            float *idepth_lm = idepth[lvlm1];
            float *weightSums_lm = weightSums[lvlm1];

            for (int y = 0; y < hl; y++)
                for (int x = 0; x < wl; x++) {
                    int bidx = 2 * x + 2 * y * wlm1;
                    idepth_l[x + y * wl] =
                            idepth_lm[bidx] +
                            idepth_lm[bidx + 1] +
                            idepth_lm[bidx + wlm1] +
                            idepth_lm[bidx + wlm1 + 1];

                    weightSums_l[x + y * wl] =
                            weightSums_lm[bidx] +
                            weightSums_lm[bidx + 1] +
                            weightSums_lm[bidx + wlm1] +
                            weightSums_lm[bidx + wlm1 + 1];
                }
        }

        // dilate idepth by 1.
        for (int lvl = 0; lvl < 2; lvl++) {
            int numIts = 1;


            for (int it = 0; it < numIts; it++) {
                int wh = w[lvl] * h[lvl] - w[lvl];
                int wl = w[lvl];
                float *weightSumsl = weightSums[lvl];
                float *weightSumsl_bak = weightSums_bak[lvl];
                memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
                float *idepthl = idepth[lvl];    // dotnt need to make a temp copy of depth, since I only
                // read values with weightSumsl>0, and write ones with weightSumsl<=0.
                for (int i = w[lvl]; i < wh; i++) {
                    if (weightSumsl_bak[i] <= 0) {
                        float sum = 0, num = 0, numn = 0;
                        if (weightSumsl_bak[i + 1 + wl] > 0) {
                            sum += idepthl[i + 1 + wl];
                            num += weightSumsl_bak[i + 1 + wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i - 1 - wl] > 0) {
                            sum += idepthl[i - 1 - wl];
                            num += weightSumsl_bak[i - 1 - wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i + wl - 1] > 0) {
                            sum += idepthl[i + wl - 1];
                            num += weightSumsl_bak[i + wl - 1];
                            numn++;
                        }
                        if (weightSumsl_bak[i - wl + 1] > 0) {
                            sum += idepthl[i - wl + 1];
                            num += weightSumsl_bak[i - wl + 1];
                            numn++;
                        }
                        if (numn > 0) {
                            idepthl[i] = sum / numn;
                            weightSumsl[i] = num / numn;
                        }
                    }
                }
            }
        }


        // dilate idepth by 1 (2 on lower levels).
        for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float *weightSumsl = weightSums[lvl];
            float *weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
            float *idepthl = idepth[lvl];    // dotnt need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++) {
                if (weightSumsl_bak[i] <= 0) {
                    float sum = 0, num = 0, numn = 0;
                    if (weightSumsl_bak[i + 1] > 0) {
                        sum += idepthl[i + 1];
                        num += weightSumsl_bak[i + 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i - 1] > 0) {
                        sum += idepthl[i - 1];
                        num += weightSumsl_bak[i - 1];
                        numn++;
                    }
                    if (weightSumsl_bak[i + wl] > 0) {
                        sum += idepthl[i + wl];
                        num += weightSumsl_bak[i + wl];
                        numn++;
                    }
                    if (weightSumsl_bak[i - wl] > 0) {
                        sum += idepthl[i - wl];
                        num += weightSumsl_bak[i - wl];
                        numn++;
                    }
                    if (numn > 0) {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }

        // normalize idepths and weights.
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            float *weightSumsl = weightSums[lvl];
            float *idepthl = idepth[lvl];
            Eigen::Vector3f *dIRefl = lastRef->dIp[lvl];

            int wl = w[lvl], hl = h[lvl];

            int lpc_n = 0;
            float *lpc_u = pc_u[lvl];
            float *lpc_v = pc_v[lvl];
            float *lpc_idepth = pc_idepth[lvl];
            float *lpc_color = pc_color[lvl];


            for (int y = 2; y < hl - 2; y++)
                for (int x = 2; x < wl - 2; x++) {
                    int i = x + y * wl;

                    if (weightSumsl[i] > 0) {
                        idepthl[i] /= weightSumsl[i];
                        lpc_u[lpc_n] = x;
                        lpc_v[lpc_n] = y;
                        lpc_idepth[lpc_n] = idepthl[i];
                        lpc_color[lpc_n] = dIRefl[i][0];


                        if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0)) {
                            idepthl[i] = -1;
                            continue;    // just skip if something is wrong.
                        }
                        lpc_n++;
                    } else
                        idepthl[i] = -1;

                    weightSumsl[i] = 1;
                }

            pc_n[lvl] = lpc_n;
        }
    }

    // for multi-fisheye  
    void CoarseTracker::makeCoarseDepthL0forMF(std::vector<shared_ptr<FrameHessian>> frameHessians)
    {
        // make coarse tracking templates for latstRef.
        int camnums = UMF->camNums;
        for(int n = 0; n < camnums; n++)
        {
            memset(videpth[0][n], 0, sizeof(float) * w[0] * h[0]);
            memset(vweightSums[0][n], 0, sizeof(float) * w[0] * h[0]);
        }
        

        // 计算其它点在最新帧投影第0层上的各个像素的逆深度权重, 和加权逆深度
        for (shared_ptr<FrameHessian> fh: frameHessians) 
        {
            for (shared_ptr<Feature> feat: fh->frame->features) 
            {
                if (feat->status == Feature::FeatureStatus::VALID &&
                    feat->point->status == Point::PointStatus::ACTIVE) 
                    {

                    shared_ptr<PointHessian> ph = feat->point->mpPH;
                    if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) 
                    {
                        shared_ptr<PointFrameResidual> r = ph->lastResiduals[0].first;
                        assert(r->isActive() && r->target.lock() == lastRef);       // 点的残差是好的, 上一次优化的target是这次的ref
                        int u = r->centerProjectedTo[0] + 0.5f;
                        int v = r->centerProjectedTo[1] + 0.5f;
                        int n = r->tcamnum;
                        float new_idepth = r->centerProjectedTo[2];
                        float weight = sqrtf(1e-3 / (ph->HdiF + 1e-12));

                        videpth[0][n][u + w[0] * v] += new_idepth * weight;
                        vweightSums[0][n][u + w[0] * v] += weight;
                    }
                }
            }
        }

        // 从下层向上层生成逆深度和权重 从1层开始
        for (int lvl = 1; lvl < pyrLevelsUsed; lvl++) 
        {
            int lvlm1 = lvl - 1;
            int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

            for(int n = 0; n < camnums; n++)
            {
                float *idepth_l = videpth[lvl][n];
                float *weightSums_l = vweightSums[lvl][n];

                float *idepth_lm = videpth[lvlm1][n];
                float *weightSums_lm = vweightSums[lvlm1][n];

                for (int y = 0; y < hl; y++)
                    for (int x = 0; x < wl; x++) 
                    {
                        int bidx = 2 * x + 2 * y * wlm1;
                        idepth_l[x + y * wl] =
                                idepth_lm[bidx] +
                                idepth_lm[bidx + 1] +
                                idepth_lm[bidx + wlm1] +
                                idepth_lm[bidx + wlm1 + 1];

                        weightSums_l[x + y * wl] =
                                weightSums_lm[bidx] +
                                weightSums_lm[bidx + 1] +
                                weightSums_lm[bidx + wlm1] +
                                weightSums_lm[bidx + wlm1 + 1];
                    }
            }
        }

        // 0和1层 对于没有深度的像素点, 使用周围斜45度的四个点来填充
        // dilate idepth by 1.
        for (int lvl = 0; lvl < 2; lvl++) 
        {
            int numIts = 1;

            for (int it = 0; it < numIts; it++) 
            {
                int wh = w[lvl] * h[lvl] - w[lvl];
                int wl = w[lvl];
                for(int n = 0; n < camnums; n++)
                {
                    float *weightSumsl = vweightSums[lvl][n];
                    float *weightSumsl_bak = vweightSums_bak[lvl][n];
                    memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
                    float *idepthl = videpth[lvl][n];    // dotnt need to make a temp copy of depth, since I only
                    // read values with weightSumsl>0, and write ones with weightSumsl<=0.
                    for (int i = w[lvl]; i < wh; i++) 
                    {
                        if (weightSumsl_bak[i] <= 0) 
                        {
                            float sum = 0, num = 0, numn = 0;
                            if (weightSumsl_bak[i + 1 + wl] > 0) 
                            {
                                sum += idepthl[i + 1 + wl];
                                num += weightSumsl_bak[i + 1 + wl];
                                numn++;
                            }
                            if (weightSumsl_bak[i - 1 - wl] > 0) 
                            {
                                sum += idepthl[i - 1 - wl];
                                num += weightSumsl_bak[i - 1 - wl];
                                numn++;
                            }
                            if (weightSumsl_bak[i + wl - 1] > 0) 
                            {
                                sum += idepthl[i + wl - 1];
                                num += weightSumsl_bak[i + wl - 1];
                                numn++;
                            }
                            if (weightSumsl_bak[i - wl + 1] > 0) 
                            {
                                sum += idepthl[i - wl + 1];
                                num += weightSumsl_bak[i - wl + 1];
                                numn++;
                            }
                            if (numn > 0) 
                            {
                                idepthl[i] = sum / numn;
                                weightSumsl[i] = num / numn;
                            }
                        }
                    }
                }
                
            }
        }

        // 2层以上, 对于没有深度的像素点, 使用上下左右的四个点来填充
        // dilate idepth by 1 (2 on lower levels).
        for (int lvl = 2; lvl < pyrLevelsUsed; lvl++) 
        {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            for(int n = 0; n < camnums; n++)
            {
                float *weightSumsl = vweightSums[lvl][n];
                float *weightSumsl_bak = vweightSums_bak[lvl][n];
                memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
                float *idepthl = videpth[lvl][n];    // dotnt need to make a temp copy of depth, since I only
                // read values with weightSumsl>0, and write ones with weightSumsl<=0.
                for (int i = w[lvl]; i < wh; i++) 
                {
                    if (weightSumsl_bak[i] <= 0) 
                    {
                        float sum = 0, num = 0, numn = 0;
                        if (weightSumsl_bak[i + 1] > 0) 
                        {
                            sum += idepthl[i + 1];
                            num += weightSumsl_bak[i + 1];
                            numn++;
                        }
                        if (weightSumsl_bak[i - 1] > 0) 
                        {
                            sum += idepthl[i - 1];
                            num += weightSumsl_bak[i - 1];
                            numn++;
                        }
                        if (weightSumsl_bak[i + wl] > 0) 
                        {
                            sum += idepthl[i + wl];
                            num += weightSumsl_bak[i + wl];
                            numn++;
                        }
                        if (weightSumsl_bak[i - wl] > 0) 
                        {
                            sum += idepthl[i - wl];
                            num += weightSumsl_bak[i - wl];
                            numn++;
                        }
                        if (numn > 0) 
                        {
                            idepthl[i] = sum / numn;
                            weightSumsl[i] = num / numn;
                        }
                    }
                }
            }
        }

        // 归一化点的逆深度并赋值给成员变量pc_*
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) 
        {
        
            int wl = w[lvl], hl = h[lvl];

            for(int n = 0; n < camnums; n++)
            {
                int lpc_n = 0;
                float *lpc_u = vpc_u[lvl][n];
                float *lpc_v = vpc_v[lvl][n];
                float *lpc_idepth = vpc_idepth[lvl][n];
                float *lpc_color = vpc_color[lvl][n];

                float *weightSumsl = vweightSums[lvl][n];
                float *idepthl = videpth[lvl][n];
                Eigen::Vector3f *dIRefl = lastRef->vdIp[lvl][n];

                for (int y = 2; y < hl - 2; y++)
                    for (int x = 2; x < wl - 2; x++) 
                    {
                        int i = x + y * wl;

                        if (weightSumsl[i] > 0) 
                        {
                            idepthl[i] /= weightSumsl[i];
                            lpc_u[lpc_n] = x;
                            lpc_v[lpc_n] = y;
                            lpc_idepth[lpc_n] = idepthl[i];
                            lpc_color[lpc_n] = dIRefl[i][0];


                            if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0)) 
                            {
                                idepthl[i] = -1;
                                continue;    // just skip if something is wrong.
                            }
                            lpc_n++;
                        } else
                            idepthl[i] = -1;

                        weightSumsl[i] = 1;
                    }

                vpc_n[lvl][n] = lpc_n;
            }
            
        }
    }

    Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH) {

        float E = 0;
        int numTermsInE = 0;
        int numTermsInWarped = 0;
        int numSaturated = 0;

        int wl = w[lvl];
        int hl = h[lvl];
        Eigen::Vector3f *dINewl = newFrame->dIp[lvl];
        float fxl = fx[lvl];
        float fyl = fy[lvl];
        float cxl = cx[lvl];
        float cyl = cy[lvl];


        Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
        Vec3f t = (refToNew.translation()).cast<float>();
        Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
                                                  aff_g2l).cast<float>();

        float sumSquaredShiftT = 0;
        float sumSquaredShiftRT = 0;
        float sumSquaredShiftNum = 0;

        float maxEnergy = 2 * setting_huberTH * cutoffTH -
                          setting_huberTH * setting_huberTH;    // energy for r=setting_coarseCutoffTH.

        int nl = pc_n[lvl];
        float *lpc_u = pc_u[lvl];
        float *lpc_v = pc_v[lvl];
        float *lpc_idepth = pc_idepth[lvl];
        float *lpc_color = pc_color[lvl];


        for (int i = 0; i < nl; i++) {
            float id = lpc_idepth[i];
            float x = lpc_u[i];
            float y = lpc_v[i];

            Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            float new_idepth = id / pt[2];

            if (lvl == 0 && i % 32 == 0) {
                // translation only (positive)
                Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
                float uT = ptT[0] / ptT[2];
                float vT = ptT[1] / ptT[2];
                float KuT = fxl * uT + cxl;
                float KvT = fyl * vT + cyl;

                // translation only (negative)
                Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
                float uT2 = ptT2[0] / ptT2[2];
                float vT2 = ptT2[1] / ptT2[2];
                float KuT2 = fxl * uT2 + cxl;
                float KvT2 = fyl * vT2 + cyl;

                //translation and rotation (negative)
                Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
                float u3 = pt3[0] / pt3[2];
                float v3 = pt3[1] / pt3[2];
                float Ku3 = fxl * u3 + cxl;
                float Kv3 = fyl * v3 + cyl;

                //translation and rotation (positive)
                //already have it.

                sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
                sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                sumSquaredShiftNum += 2;
            }

            if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0)) continue;


            float refColor = lpc_color[i];
            Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
            if (!std::isfinite((float) hitColor[0])) continue;
            float residual = hitColor[0] - (float) (affLL[0] * refColor + affLL[1]);  // 计算光度残差
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


            if (fabs(residual) > cutoffTH) {
                E += maxEnergy;
                numTermsInE++;
                numSaturated++;
            } else {

                E += hw * residual * residual * (2 - hw);
                numTermsInE++;
                // 后续需要用到的变量保存在
                buf_warped_idepth[numTermsInWarped] = new_idepth; 
                buf_warped_u[numTermsInWarped] = u;   // 归一化坐标
                buf_warped_v[numTermsInWarped] = v;
                buf_warped_dx[numTermsInWarped] = hitColor[1];
                buf_warped_dy[numTermsInWarped] = hitColor[2];
                buf_warped_residual[numTermsInWarped] = residual;
                buf_warped_weight[numTermsInWarped] = hw;
                buf_warped_refColor[numTermsInWarped] = lpc_color[i];   // 参考帧中的灰度值
                numTermsInWarped++;
            }
        }

        while (numTermsInWarped % 4 != 0) {
            buf_warped_idepth[numTermsInWarped] = 0;
            buf_warped_u[numTermsInWarped] = 0;
            buf_warped_v[numTermsInWarped] = 0;
            buf_warped_dx[numTermsInWarped] = 0;
            buf_warped_dy[numTermsInWarped] = 0;
            buf_warped_residual[numTermsInWarped] = 0;
            buf_warped_weight[numTermsInWarped] = 0;
            buf_warped_refColor[numTermsInWarped] = 0;
            numTermsInWarped++;
        }
        buf_warped_n = numTermsInWarped;

        Vec6 rs;
        rs[0] = E;   // 总的残差和
        rs[1] = numTermsInE;  // 图像范围内的总点数
        rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);    // 只算平移（正负）无旋转后 点的像素坐标 移动距离 均值
        rs[3] = 0;
        rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);   // 算平移（正负）和旋转 后点的像素坐标 移动距离 均值
        rs[5] = numSaturated / (float) numTermsInE;   // 残差大的点占总点数的百分比

        return rs;
    }

    // for multi-fisheye 计算当前位姿投影得到的残差(能量值), 并进行一些统计 构造尽量多的点, 有助于跟踪
    Vec6 CoarseTracker::calcResforMF(int lvl, const SE3 &refToNew, vector<vector<Vec2f>> vvaffLL, float cutoffTH)
    {
        float E = 0;
        int numTermsInE = 0;
        int numTermsInWarped = 0;
        int numSaturated = 0;

        int wl = w[lvl];
        int hl = h[lvl];

        Mat33f R = refToNew.rotationMatrix().cast<float>();  //前一帧到当前帧
        Vec3f t = (refToNew.translation()).cast<float>();
        int camnums = UMF->camNums;
        // vector<vector<Vec2f>> vvaffLL;
        // vvaffLL.resize(camnums);
        // for(int n1 = 0; n1 < camnums; n1++)
        // {
        //     vvaffLL[n1].resize(camnums);
        //     for(int n2 = 0; n2 < camnums; n2++)
        //     {
        //         // 从 ref参考帧 到 当前帧
        //         vvaffLL[n1][n2] = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, vlastRef_aff_g2l[n1],
        //                                           vaff_g2l[n2]).cast<float>();
        //     }
            
        // }

        float sumSquaredShiftT = 0;
        float sumSquaredShiftRT = 0;
        float sumSquaredShiftNum = 0;


        float maxEnergy = 2 * setting_huberTH * cutoffTH -
                          setting_huberTH * setting_huberTH;    // energy for r=setting_coarseCutoffTH.

        for(int n = 0; n < camnums; n++)
        {
            Eigen::Vector3f* dINewl = newFrame->vdIp[lvl][n];

            int nl = vpc_n[lvl][n];
            float *lpc_u = vpc_u[lvl][n];
            float *lpc_v = vpc_v[lvl][n];
            float *lpc_idepth = vpc_idepth[lvl][n];
            float *lpc_color = vpc_color[lvl][n];

            for (int i = 0; i < nl; i++)
            {
                float id = lpc_idepth[i];
                float x = lpc_u[i];
                float y = lpc_v[i];

                double X,Y,Z;
                UMF->LadybugProjectRectifyPtToSphere(n, x, y, &X, &Y, &Z, lvl);
                float SphereRadius = UMF->GetSphereRadius();
                // Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) + t * id;
                Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) / id + t;

                float S_norm_pt = SphereRadius / pt.norm();

                float xs = S_norm_pt * pt(0);
                float ys = S_norm_pt * pt(1);
                float zs = S_norm_pt * pt(2);

                double u, v;
                int tocamnum;

                // UMF->LadybugReprojectSpherePtToRectify(xs, ys, zs, &tocamnum, &u, &v, lvl);
                UMF->LadybugReprojectSpherePtToRectifyfixNum(xs, ys, zs, n, &u, &v, lvl);
                tocamnum = n;

                float Ku = u;
                float Kv = v;
                // float new_idepth = id / pt.norm(); // 当前帧上的深度
                float new_idepth = 1 / pt.norm(); // 当前帧上的深度

                // 统计像素的移动
                if (lvl == 0 && i % 32 == 0 )  //* 第0层 每隔32个点 且在相同相机号内
                {

                    Vec3f ptT1 = Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) / id + t;
                    float S_norm_pt1 = SphereRadius / ptT1.norm();
                    float xs1 = S_norm_pt1 * ptT1(0);
                    float ys1 = S_norm_pt1 * ptT1(1);
                    float zs1 = S_norm_pt1 * ptT1(2);
                    // double KuT1, KvT1;
                    // int tocamnum1;
                    // UMF->LadybugReprojectSpherePtToRectify(xs1, ys1, zs1, &tocamnum1, &KuT1, &KvT1, lvl);
                    // if(n != tocamnum1)
                    //     break;

                    // translation only (negative)
                    Vec3f ptT2 = Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) / id - t;
                    float S_norm_pt2 = SphereRadius / ptT2.norm();
                    float xs2 = S_norm_pt2 * ptT2(0);
                    float ys2 = S_norm_pt2 * ptT2(1);
                    float zs2 = S_norm_pt2 * ptT2(2);
                    // double KuT2, KvT2;
                    // int tocamnum2;
                    // UMF->LadybugReprojectSpherePtToRectify(xs2, ys2, zs2, &tocamnum2, &KuT2, &KvT2, lvl);
                    // if(n != tocamnum2)
                    //     break;

                    //translation and rotation (negative)
                    Vec3f ptT3 = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) / id - t;
                    float S_norm_pt3 = SphereRadius / ptT3.norm();
                    float xs3 = S_norm_pt3 * ptT3(0);
                    float ys3 = S_norm_pt3 * ptT3(1);
                    float zs3 = S_norm_pt3 * ptT3(2);
                    // double Ku3, Kv3;
                    // int tocamnum3;
                    // UMF->LadybugReprojectSpherePtToRectify(xs3, ys3, zs3, &tocamnum3, &Ku3, &Kv3, lvl);
                    // if(n != tocamnum3)
                    //     break;

                    //translation and rotation (positive)
                    //already have it.
                    //* 统计像素的移动大小
                    // sumSquaredShiftT += (KuT1 - x) * (KuT1 - x) + (KvT1 - y) * (KvT1 - y);
                    // sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                    // sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                    // sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                    sumSquaredShiftT += (xs1 - X) * (xs1 - X) + (ys1 - Y) * (ys1 - Y) + (zs1 - Z) * (zs1 - Z);
                    sumSquaredShiftT += (xs2 - X) * (xs2 - X) + (ys2 - Y) * (ys2 - Y) + (zs2 - Z) * (zs2 - Z);
                    sumSquaredShiftRT += (xs - X) * (xs - X) + (ys - Y) * (ys - Y) + (zs - Z) * (zs - Z);
                    sumSquaredShiftRT += (xs3 - X) * (xs3 - X) + (ys3 - Y) * (ys3 - Y) + (zs3 - Z) * (zs3 - Z);
                    sumSquaredShiftNum += 2;    // 平移两种，平移加旋转两种
                }

                if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0)) continue;

                float refColor = lpc_color[i];
                Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
                if (!std::isfinite((float) hitColor[0])) continue;
                float residual = hitColor[0] - (float) (vvaffLL[n][tocamnum][0] * refColor + vvaffLL[n][tocamnum][1]);  // 计算光度残差
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


                if (fabs(residual) > cutoffTH) 
                {
                    E += maxEnergy;
                    numTermsInE++;		// E 中数目
			        numSaturated++;		// 大于阈值数目
                } 
                else 
                {
                    E += hw * residual * residual * (2 - hw);
                    numTermsInE++;
                    // 后续需要用到的变量保存在
                    buf_warped_idepth[numTermsInWarped] = id;
                    buf_warped_fromcamnum[numTermsInWarped] = n;
                    buf_warped_tocamnum[numTermsInWarped] = tocamnum;
                    buf_warped_a[numTermsInWarped] = vvaffLL[n][tocamnum][0];
                    buf_warped_b[numTermsInWarped] = vvaffLL[n][tocamnum][1];
                    buf_warped_b0[numTermsInWarped] = vlastRef_aff_g2l[n].b;
                    buf_warped_xsT[numTermsInWarped] = pt(0);
                    buf_warped_ysT[numTermsInWarped] = pt(1);
                    buf_warped_zsT[numTermsInWarped] = pt(2);
                    buf_warped_xsF[numTermsInWarped] = X;
                    buf_warped_ysF[numTermsInWarped] = Y;
                    buf_warped_zsF[numTermsInWarped] = Z;
                    buf_warped_u[numTermsInWarped] = u;   
                    buf_warped_v[numTermsInWarped] = v;
                    buf_warped_dx[numTermsInWarped] = hitColor[1];
                    buf_warped_dy[numTermsInWarped] = hitColor[2];
                    buf_warped_residual[numTermsInWarped] = residual;
                    buf_warped_weight[numTermsInWarped] = hw;
                    buf_warped_refColor[numTermsInWarped] = lpc_color[i];   // 参考帧中的灰度值
                    numTermsInWarped++;
                }
            }
        }

        // //* 16字节对齐, 填充上
        // while(numTermsInWarped%4!=0) 
        // {
        //     buf_warped_idepth[numTermsInWarped] = 0;
        //     buf_warped_u[numTermsInWarped] = 0;
        //     buf_warped_v[numTermsInWarped] = 0;
        //     buf_warped_dx[numTermsInWarped] = 0;
        //     buf_warped_dy[numTermsInWarped] = 0;
        //     buf_warped_residual[numTermsInWarped] = 0;
        //     buf_warped_weight[numTermsInWarped] = 0;
        //     buf_warped_refColor[numTermsInWarped] = 0;
        //     numTermsInWarped++;
        // }
        buf_warped_n = numTermsInWarped;

        Vec6 rs;
        rs[0] = E;   // 总的残差平方和
        rs[1] = numTermsInE;  // 图像范围内的总点数
        rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);    // 只算平移（正负）无旋转后 点的像素坐标 移动距离 均值
        rs[3] = 0;
        rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);   // 算平移（正负）和旋转 后点的像素坐标 移动距离 均值
        rs[5] = numSaturated / (float) numTermsInE;   // 残差大的点占总点数的百分比

        return rs;
        
    }

    void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l) {

        acc.initialize();

        __m128 fxl = _mm_set1_ps(fx[lvl]);
        __m128 fyl = _mm_set1_ps(fy[lvl]);
        __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
        __m128 a = _mm_set1_ps(
                (float) (AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l,
                                                     aff_g2l)[0]));

        __m128 one = _mm_set1_ps(1);
        __m128 minusOne = _mm_set1_ps(-1);
        __m128 zero = _mm_set1_ps(0);

        int n = buf_warped_n;
        assert(n % 4 == 0);
        for (int i = 0; i < n; i += 4) {
            __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);  // dx*fx
            __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);  // dy*fy
            __m128 u = _mm_load_ps(buf_warped_u + i);
            __m128 v = _mm_load_ps(buf_warped_v + i);
            __m128 id = _mm_load_ps(buf_warped_idepth + i);

            // 残差项关于6位姿，2光度仿射参数
            acc.updateSSE_eighted(
                    _mm_mul_ps(id, dx),
                    _mm_mul_ps(id, dy),
                    _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))),
                    _mm_sub_ps(zero, _mm_add_ps(
                            _mm_mul_ps(_mm_mul_ps(u, v), dx),
                            _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),
                    _mm_add_ps(
                            _mm_mul_ps(_mm_mul_ps(u, v), dy),
                            _mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),
                    _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),
                    _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + i))),    // 对a导数 r = I2 - exp(a)*(I1-b0)-b   dr/da = exp(a)(b0-I1)
                    minusOne,
                    _mm_load_ps(buf_warped_residual + i),   // 残差
                    _mm_load_ps(buf_warped_weight + i));    // 权重
        }

        acc.finish();
        H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
        b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

        H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
        H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
        H_out.block<8, 1>(0, 6) *= SCALE_A;
        H_out.block<8, 1>(0, 7) *= SCALE_B;
        H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
        H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
        H_out.block<1, 8>(6, 0) *= SCALE_A;
        H_out.block<1, 8>(7, 0) *= SCALE_B;
        b_out.segment<3>(0) *= SCALE_XI_ROT;
        b_out.segment<3>(3) *= SCALE_XI_TRANS;
        b_out.segment<1>(6) *= SCALE_A;
        b_out.segment<1>(7) *= SCALE_B;
    }

    void CoarseTracker::calcGSSSEforMF(int lvl, Mat5656 &H_out, Vec56 &b_out, const SE3 &refToNew)
    {
        accforMF.initialize();
        // 为了加速　+4 保证创建够四的倍数
        float dp0[buf_warped_n + 4], dp1[buf_warped_n + 4], dp2[buf_warped_n + 4], 
            dp3[buf_warped_n + 4], dp4[buf_warped_n + 4], dp5[buf_warped_n + 4], 
            dp6[buf_warped_n + 4], dp7[buf_warped_n + 4], dd[buf_warped_n + 4], 
            r[buf_warped_n + 4], w[buf_warped_n + 4];
        int fn[buf_warped_n + 4], tn[buf_warped_n + 4]; // from to 相机号

        Vec3f t = (refToNew.translation()).cast<float>();
        Mat33f R = (refToNew.rotationMatrix()).cast<float>();
        for(int i = 0; i < buf_warped_n; i++)
        {
            Eigen::Matrix<float, 2, 6> dxrdpose;
            Eigen::Matrix<float, 2, 1> dxrdrho1;

            computedxrdposedrho1(R, t, Vec3f(buf_warped_xsF[i], buf_warped_ysF[i], buf_warped_zsF[i]), 
                    Vec3f(buf_warped_xsT[i], buf_warped_ysT[i], buf_warped_zsT[i]), buf_warped_tocamnum[i], buf_warped_idepth[i], dxrdrho1, dxrdpose, lvl);

            float dxInterp = buf_warped_dx[i];    
            float dyInterp = buf_warped_dy[i];    

            Eigen::Matrix<float, 1, 6> dIdpose;
            Eigen::Matrix<float, 1, 1> dIdrho1;
            dIdpose = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdpose;    // dI/dpose = dI/dxr * dxr/dXs * dXs/dpose
            dIdrho1 = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdrho1;    // dI/drho1 = dI/dxr * dxr/dXs * dXs/drho1


            dp0[i] = dIdpose(0,0);                // dp0, dp1, dp2, dp3, dp4, dp5 是光度误差对 se(3) 六个量的导数
            dp1[i] = dIdpose(0,1);          
            dp2[i] = dIdpose(0,2);   
            dp3[i] = dIdpose(0,3);
            dp4[i] = dIdpose(0,4);
            dp5[i] = dIdpose(0,5);
            dp6[i] = buf_warped_a[i] * (buf_warped_b0[i] - buf_warped_refColor[i]);           // r21 = wh(I2 - exp(log21)*(I1-b1)-b2),   dr21/dlog21 = wh*exp(log21)*(b1-I1)
            dp7[i] = 1;
            dd[i] = dIdrho1(0,0);   // h
            r[i] = buf_warped_residual[i];   // 用于计算b
            w[i] = buf_warped_weight[i];

            fn[i] = buf_warped_fromcamnum[i];
            tn[i] = buf_warped_tocamnum[i];

// NumericDiff 数值求导
            // vector<float> dp, dpp;
            // dp.resize(7);
            // dpp.resize(7);
            // dpp = NumericDiff(fn[i], buf_warped_u[i], buf_warped_v[i], lvl, R, t, buf_warped_idepth[i], Vec2f(buf_warped_a[i], buf_warped_b[i]));

            // cout << "dIdpose:      " << dp0[i] << " " << dp1[i] << " " << dp2[i] << " " << dp3[i] << " " << dp4[i] << " " << dp5[i] << " "<< dd[i] << " " <<endl;
            // //cout << "dIdposetest:  " << dp[0] << " " << dp[1] << " " << dp[2] << " " << dp[3] << " " << dp[4] << " " << dp[5] << " " << dp[6] << " " <<endl;
            // cout << "dIdposetest:  " << dpp[0] << " " << dpp[1] << " " << dpp[2] << " " << dpp[3] << " " << dpp[4] << " " << dpp[5] << " " << dpp[6] << " " <<endl << endl;

            // int tt = 0;
            // dp0[i] = dpp[0];                
            // dp1[i] = dpp[1];          
            // dp2[i] = dp[2];   
            // dp3[i] = dpp[3];
            // dp4[i] = dpp[4];
            // dp5[i] = dpp[5];
            // dd[i] = dpp[6];   // h

//test end
        }
        // 补全用于加速
        while(buf_warped_n % 4 != 0)
        {
            dp0[buf_warped_n] = 0;           
            dp1[buf_warped_n] = 0;          
            dp2[buf_warped_n] = 0;   
            dp3[buf_warped_n] = 0;
            dp4[buf_warped_n] = 0;
            dp5[buf_warped_n] = 0;
            dp6[buf_warped_n] = 0;      
            dp7[buf_warped_n] = 1;
            dd[buf_warped_n] = 0;  
            r[buf_warped_n] = 0;   // 用于计算b
            w[buf_warped_n] = 0;

            fn[buf_warped_n] = 0;
            tn[buf_warped_n] = 0;
            buf_warped_n++;
        }
        

        int n = buf_warped_n;
        assert(n % 4 == 0);
        for (int i = 0; i < n; i += 4)
        {
            // accforMF.updateSSE_weighted(
            //      _mm_load_ps(dp0 + i),
            //      _mm_load_ps(dp1 + i),
            //      _mm_load_ps(dp2 + i),
            //      _mm_load_ps(dp3 + i),
            //      _mm_load_ps(dp4 + i),
            //      _mm_load_ps(dp5 + i),
            //      _mm_load_ps(dp6 + i),
            //      _mm_load_ps(dp7 + i),
            //      _mm_load_ps(r + i),
            //      _mm_load_ps(w + i),
            //      _mm_set_epi32(fn[i+1], fn[i+2], fn[i+3], fn[i+4]),
            //      _mm_set_epi32(tn[i+1], tn[i+2], tn[i+3], tn[i+4])
            // );

             accforMF.updateSingleWeighted(
                 dp0 [i],
                 dp1 [i],
                 dp2 [i],
                 dp3 [i],
                 dp4 [i],
                 dp5 [i],
                 dp6 [i],
                 dp7 [i],
                 r [i],
                 w [i],
                 fn[i],
                 tn[i]
            );
        }


        accforMF.finish();
        H_out = accforMF.H.topLeftCorner<56, 56>().cast<double>() * (1.0f / n);
        b_out = accforMF.H.topRightCorner<56, 1>().cast<double>() * (1.0f / n);

        H_out.block<56, 3>(0, 0) *= SCALE_XI_TRANS;
        H_out.block<56, 3>(0, 3) *= SCALE_XI_ROT;
        H_out.block<3, 56>(0, 0) *= SCALE_XI_TRANS;
        H_out.block<3, 56>(3, 0) *= SCALE_XI_ROT;
        b_out.segment<3>(0) *= SCALE_XI_TRANS;
        b_out.segment<3>(3) *= SCALE_XI_ROT;

        for(int n = 0; n < 25; n++)
        {
            H_out.block<56, 1>(0, 6 + n*2) *= SCALE_A;
            H_out.block<56, 1>(0, 7 + n*2) *= SCALE_B;
            H_out.block<1, 56>(6 + n*2, 0) *= SCALE_A;
            H_out.block<1, 56>(7 + n*2, 0) *= SCALE_B;
            b_out.segment<1>(6 + n*2) *= SCALE_A;
            b_out.segment<1>(7 + n*2) *= SCALE_B;
        }
        
    }

    // for multi-fisheye Diff test drdT drdrho
    vector<float> CoarseTracker::NumericDiff(int n, double x, double y,int lvl, Mat33f R, Vec3f t, float idepth, Vec2f ab)
    {
// 中值求导
        // vector<float> dp;
        // dp.resize(7);
        // float da =1e-6; 
        // Vec5f r1, r2;
        // Vec3f tnew11(t),tnew12(t);
        // tnew11(0) = t(0) + da;
        // tnew12(0) = t(0) - da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew11,idepth, ab);
        // r2 = testJaccobian(n,x, y, lvl, R, tnew12,idepth, ab);
        // //float dt0 
        // // dp[0] = (r1(2)- r2(2))/(2*da);
        // dp[0] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // Vec3f tnew21(t),tnew22(t);
        // tnew21(1) = t(1) + da;
        // tnew22(1) = t(1) - da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew21,idepth, ab);
        // r2 = testJaccobian(n,x, y, lvl, R, tnew22,idepth, ab);
        // //float dt1 
        // // dp[1] = (r1(2)- r2(2))/(2*da);
        // dp[1] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // Vec3f tnew31(t),tnew32(t);
        // tnew31(2) = t(2) + da;
        // tnew32(2) = t(2) - da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew31,idepth, ab);
        // r2 = testJaccobian(n,x, y, lvl, R, tnew32,idepth, ab);
        // //float dt2 
        // // dp[2] = (r1(2)- r2(2))/(2*da);
        // dp[2] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // SO3 Rnew11(R.cast<double>()),Rnew12(R.cast<double>());
        // Eigen::Vector3d so311= Rnew11.log();
        // Eigen::Vector3d so312= Rnew12.log();
        // so311(0) = so311(0) + da;
        // so312(0) = so312(0) - da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so311).matrix().cast<float>(), t,idepth, ab);
        // r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so312).matrix().cast<float>(), t,idepth, ab);
        // //float dr1 
        // // dp[3] = (r1(2)- r2(2))/(2*da);
        // dp[3] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // SO3 Rnew21(R.cast<double>()),Rnew22(R.cast<double>());
        // Eigen::Vector3d so321= Rnew21.log();
        // Eigen::Vector3d so322= Rnew22.log();
        // so321(1) = so321(1) + da;
        // so322(1) = so322(1) - da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so321).matrix().cast<float>(), t,idepth, ab);
        // r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so322).matrix().cast<float>(), t,idepth, ab);
        // //float dr2 
        // // dp[4] = (r1(2)- r2(2))/(2*da);
        // dp[4] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // SO3 Rnew31(R.cast<double>()),Rnew32(R.cast<double>());
        // Eigen::Vector3d so331= Rnew31.log();
        // Eigen::Vector3d so332= Rnew32.log();
        // so331(2) = so331(2) + da;
        // so332(2) = so332(2) - da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so331).matrix().cast<float>(), t,idepth, ab);
        // r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so332).matrix().cast<float>(), t,idepth, ab);
        // //float dr3 
        // // dp[5] = (r1(2)- r2(2))/(2*da);
        // dp[5] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        // float idepthnew1=idepth;
        // float idepthnew2=idepth;
        // idepthnew1 = idepthnew1 + da;
        // idepthnew2 = idepthnew2 - da;
        // r1 = testJaccobian(n,x, y, lvl, R, t,idepthnew1, ab);
        // r2 = testJaccobian(n,x, y, lvl, R, t,idepthnew2, ab);
        // //float di 
        // // dp[6] = (r1(2)- r2(2))/(2*da);
        // dp[6] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

// // 前向求导
        Vec5f r = testJaccobian(n,x, y, lvl, R, t,idepth, ab);
        vector<float> dp;
        dp.resize(7);
        float da =1e-6; 
        Vec5f r1, r2;
        Vec3f tnew11(t);
        tnew11(0) = t(0) + da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew11,idepth, ab);
        //float dt0 
        dp[0] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        Vec3f tnew21(t);
        tnew21(1) = t(1) + da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew21,idepth, ab);
        //float dt1 
        //dp[1] = (r1- r)/da;
        dp[1] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        Vec3f tnew31(t);
        tnew31(2) = t(2) + da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew31,idepth, ab);
        //float dt2 
        // dp[2] = (r1- r)/da;
        dp[2] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        SO3 Rnew11(R.cast<double>());
        Eigen::Vector3d so311= Rnew11.log();
        so311(0) = so311(0) + da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so311).matrix().cast<float>(), t,idepth, ab);
        //float dr1 
        //dp[3] = (r1- r)/da;
        dp[3] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        SO3 Rnew21(R.cast<double>());
        Eigen::Vector3d so321= Rnew21.log();
        so321(1) = so321(1) + da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so321).matrix().cast<float>(), t,idepth, ab);
        //float dr2 
        // dp[4] = (r1- r)/da;
        dp[4] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        SO3 Rnew31(R.cast<double>());
        Eigen::Vector3d so331= Rnew31.log();
        so331(2) = so331(2) + da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so331).matrix().cast<float>(), t,idepth, ab);
        //float dr3 
        // dp[5] = (r1- r)/da;
        dp[5] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;


        float idepthnew1=idepth;
        idepthnew1 = idepthnew1 + da;
        r1 = testJaccobian(n,x, y, lvl, R, t,idepthnew1, ab);
        //float di 
        // dp[6] = (r1- r)/da;
        dp[6] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;
//
        return dp;
    }

    Vec5f CoarseTracker::testJaccobian(int n, double RectifiedPixalx, double RectifiedPixaly, int lvl, Mat33f R, Vec3f t, float idepth, Vec2f ab)
    {
        bool isGood = true;
        int wl = UMF->wPR[lvl], hl = UMF->hPR[lvl];

        vector<Eigen::Vector3f* > vcolorRef(lastRef->vdIp[lvl]);
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
        UMF->LadybugReprojectSpherePtToRectify(xs, ys, zs, &tocamnum, &Ku, &Kv, lvl);
        //LOG(INFO) << "LadybugReprojectSpherePtToFishEyeImg" << endl ;

        //float new_idepth = point->idepth_new / pt[2];   // d1 * d2


        Vec3f hitColor = getInterpolatedElement33(vcolorNew[tocamnum], Ku, Kv, wl);    // 在当前第二帧 中的 灰度值
        // Vec3f hitColor = getInterpolatedElement33BiCub(vcolorNew[tocamnum], Ku, Kv, wl);

        //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
        float rlR = getInterpolatedElement31(vcolorRef[n], RectifiedPixalx, RectifiedPixaly, wl);  // 在 第一帧 中的 灰度值


        float residual = hitColor[0] - ab[0] * rlR - ab[1];  // 计算光度误差 r = I2 - a21*I1 -b21 (I2 = a21*I1 + b21)   n 第一帧像素所在相机号 ， tocamnum 投影到当前帧 像素所在相机号
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
        if (hw < 1) hw = sqrtf(hw);
        float r = residual;
        float dxInterp = hitColor[1];    
        float dyInterp = hitColor[2];    
        float u = Ku;
        float v = Kv;
        Vec5f a;
        a << u, v, r, dxInterp, dyInterp;
        return a;
    }
//
    // ============================================================================== //
    // Coarse distance map

    CoarseDistanceMap::CoarseDistanceMap(int ww, int hh) {

        fwdWarpedIDDistFinal = new float[ww * hh / 4];
        bfsList1 = new Eigen::Vector2i[ww * hh / 4];
        bfsList2 = new Eigen::Vector2i[ww * hh / 4];
        int fac = 1 << (pyrLevelsUsed - 1);
        coarseProjectionGrid = new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
        coarseProjectionGridNum = new int[ww * hh / (fac * fac)];
        w[0] = h[0] = 0;

    }

    CoarseDistanceMap::CoarseDistanceMap(UndistortMultiFisheye * uMF): UMF(uMF)
    {
        // 在第一层上算的, 所以除4
        int ww = UMF->wPR[0];
        int hh = UMF->hPR[0];
        //fwdWarpedIDDistFinal = new float[ww * hh / 4];
        // bfsList1 = new Eigen::Vector2i[ww * hh / 4];
        // bfsList2 = new Eigen::Vector2i[ww * hh / 4];
        int fac = 1 << (pyrLevelsUsed - 1);
        coarseProjectionGrid = new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
        coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

        int camNums = UMF->camNums;
        vbfsList1.resize(camNums);
        vbfsList2.resize(camNums);
        vfwdWarpedIDDistFinal.resize(camNums);
        for(int n = 0; n < camNums; n++)
        {
            vbfsList1[n] = new Eigen::Vector2i[ww * hh / 4];
            vbfsList2[n] = new Eigen::Vector2i[ww * hh / 4];
            vfwdWarpedIDDistFinal[n] = new float[ww * hh / 4];
        }

        for(int l = 0; l < pyrLevelsUsed; l++)
        {
            w[l] = UMF->wPR[l];
            h[l] = UMF->hPR[l];
        }

    }

    CoarseDistanceMap::~CoarseDistanceMap() {
        //delete[] fwdWarpedIDDistFinal;
        // delete[] bfsList1;
        // delete[] bfsList2;
        int camNums = UMF->camNums;
        for(int n = 0; n < camNums; n++)
        {
            delete[] vbfsList1[n];
            delete[] vbfsList2[n];
            delete[] vfwdWarpedIDDistFinal[n];
        }

        delete[] coarseProjectionGrid;
        delete[] coarseProjectionGridNum;
    }

    void CoarseDistanceMap::makeK(shared_ptr<CalibHessian> HCalib) {

        w[0] = wG[0];
        h[0] = hG[0];

        fx[0] = HCalib->fxl();
        fy[0] = HCalib->fyl();
        cx[0] = HCalib->cxl();
        cy[0] = HCalib->cyl();

        for (int level = 1; level < pyrLevelsUsed; ++level) {
            w[level] = w[0] >> level;
            h[level] = h[0] >> level;
            fx[level] = fx[level - 1] * 0.5;
            fy[level] = fy[level - 1] * 0.5;
            cx[level] = (cx[0] + 0.5) / ((int) 1 << level) - 0.5;
            cy[level] = (cy[0] + 0.5) / ((int) 1 << level) - 0.5;
        }

        for (int level = 0; level < pyrLevelsUsed; ++level) {
            K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
            Ki[level] = K[level].inverse();
            fxi[level] = Ki[level](0, 0);
            fyi[level] = Ki[level](1, 1);
            cxi[level] = Ki[level](0, 2);
            cyi[level] = Ki[level](1, 2);
        }
    }

    void CoarseDistanceMap::makeDistanceMap(std::vector<shared_ptr<FrameHessian>> &frameHessians,
                                            shared_ptr<FrameHessian> frame) {

        int w1 = w[1];
        int h1 = h[1];
        int wh1 = w1 * h1;
        for (int i = 0; i < wh1; i++)
            fwdWarpedIDDistFinal[i] = 1000;


        // make coarse tracking templates for latstRef.
        int numItems = 0;

        for (auto fh : frameHessians) {
            if (frame == fh) continue;

            SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
            Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
            Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

            for (auto feat: fh->frame->features) {
                if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) {
                    auto ph = feat->point->mpPH;
                    Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
                    int u = ptp[0] / ptp[2] + 0.5f;
                    int v = ptp[1] / ptp[2] + 0.5f;
                    if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
                    fwdWarpedIDDistFinal[u + w1 * v] = 0;
                    bfsList1[numItems] = Eigen::Vector2i(u, v);
                    numItems++;
                }
            }
        }

        growDistBFS(numItems);
    }

    // for multi-fisheye
    void CoarseDistanceMap::makeDistanceMapforMF(std::vector<shared_ptr<FrameHessian>> &frameHessians,
                                            shared_ptr<FrameHessian> frame) {

        int w1 = w[1];
        int h1 = h[1];
        int wh1 = w1 * h1;
        int camNums = UMF->camNums;
        for(int n = 0; n < camNums; n++)
        {
            for (int i = 0; i < wh1; i++)
                vfwdWarpedIDDistFinal[n][i] = 1000;
        }
        


        // make coarse tracking templates for latstRef.
        vector<int> vnumItems(camNums, 0);

        for (auto fh : frameHessians) {
            if (frame == fh) continue;

            SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
            // Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);   //从0层到1层转换
            // Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());
            Mat33f R = fhToNew.rotationMatrix().cast<float>() ;
            Vec3f t = fhToNew.translation().cast<float>();
            float SphereRadius = UMF->GetSphereRadius();

            for (auto feat: fh->frame->features) 
            {
                if (feat->point && feat->point->status == Point::PointStatus::ACTIVE) 
                {
                    auto ph = feat->point->mpPH;
                    float xs = ph->xs;
                    float ys = ph->ys;
                    float zs = ph->zs;
                   
                    Vec3f pt = R * Vec3f(xs/SphereRadius, ys/SphereRadius, zs/SphereRadius) / ph->idepth_scaled + t;  
                    
                    float S_norm_pt = SphereRadius/pt.norm();

                    float xs2 = S_norm_pt*pt(0);
                    float ys2 = S_norm_pt*pt(1);
                    float zs2 = S_norm_pt*pt(2);

                    double Ku, Kv;
                    int tocamnum;
                    UMF->LadybugReprojectSpherePtToRectify(xs2, ys2, zs2, &tocamnum, &Ku, &Kv, 1);

                    int u = (int)Ku + 0.5f;
                    int v = (int)Kv + 0.5f;
                    if (!(u > 0 && v > 0 && u < w[1] && v < h[1]) || fh->vmask[1][tocamnum][u + w1*v] == 0) continue;
                    vfwdWarpedIDDistFinal[tocamnum][u + w1 * v] = 0;
                    vbfsList1[tocamnum][vnumItems[tocamnum]] = Eigen::Vector2i(u, v);
                    vnumItems[tocamnum]++;
                }
            }
        }

        growDistBFSforMF(vnumItems);
    }

    void CoarseDistanceMap::growDistBFS(int bfsNum) {

        assert(w[0] != 0);
        int w1 = w[1], h1 = h[1];
        for (int k = 1; k < 40; k++) {
            int bfsNum2 = bfsNum;
            std::swap<Eigen::Vector2i *>(bfsList1, bfsList2);
            bfsNum = 0;

            if (k % 2 == 0) {
                for (int i = 0; i < bfsNum2; i++) {
                    int x = bfsList2[i][0];
                    int y = bfsList2[i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (fwdWarpedIDDistFinal[idx + 1] > k) {
                        fwdWarpedIDDistFinal[idx + 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1] > k) {
                        fwdWarpedIDDistFinal[idx - 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + w1] > k) {
                        fwdWarpedIDDistFinal[idx + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - w1] > k) {
                        fwdWarpedIDDistFinal[idx - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }
                }
            } else {
                for (int i = 0; i < bfsNum2; i++) {
                    int x = bfsList2[i][0];
                    int y = bfsList2[i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (fwdWarpedIDDistFinal[idx + 1] > k) {
                        fwdWarpedIDDistFinal[idx + 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1] > k) {
                        fwdWarpedIDDistFinal[idx - 1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + w1] > k) {
                        fwdWarpedIDDistFinal[idx + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - w1] > k) {
                        fwdWarpedIDDistFinal[idx - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }

                    if (fwdWarpedIDDistFinal[idx + 1 + w1] > k) {
                        fwdWarpedIDDistFinal[idx + 1 + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1 + w1] > k) {
                        fwdWarpedIDDistFinal[idx - 1 + w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx - 1 - w1] > k) {
                        fwdWarpedIDDistFinal[idx - 1 - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
                        bfsNum++;
                    }
                    if (fwdWarpedIDDistFinal[idx + 1 - w1] > k) {
                        fwdWarpedIDDistFinal[idx + 1 - w1] = k;
                        bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
                        bfsNum++;
                    }
                }
            }
        }
    }

    // for multi-fisheye
    void CoarseDistanceMap::growDistBFSforMF(vector<int> vbfsNum)
    {
         int w1 = UMF->wPR[1], h1 = UMF->hPR[1];
         for(int n = 0; n < UMF->camNums; n++)
         {
            for (int k = 1; k < 40; k++) 
            {
                int bfsNum = vbfsNum[n];
                int bfsNum2 = bfsNum;
                std::swap<Eigen::Vector2i *>(vbfsList1[n], vbfsList2[n]);
                bfsNum = 0;

                if (k % 2 == 0) 
                {
                    for (int i = 0; i < bfsNum2; i++) 
                    {
                        int x = vbfsList2[n][i][0];
                        int y = vbfsList2[n][i][1];
                        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                        int idx = x + y * w1;

                        if (vfwdWarpedIDDistFinal[n][idx + 1] > k) {
                            vfwdWarpedIDDistFinal[n][idx + 1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx - 1] > k) {
                            vfwdWarpedIDDistFinal[n][idx - 1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx + w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx + w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y + 1);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx - w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx - w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y - 1);
                            bfsNum++;
                        }
                    }
                } 
                else 
                {
                    for (int i = 0; i < bfsNum2; i++) 
                    {
                        int x = vbfsList2[n][i][0];
                        int y = vbfsList2[n][i][1];
                        if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                        int idx = x + y * w1;

                        if (vfwdWarpedIDDistFinal[n][idx + 1] > k) {
                            vfwdWarpedIDDistFinal[n][idx + 1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx - 1] > k) {
                            vfwdWarpedIDDistFinal[n][idx - 1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx + w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx + w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y + 1);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx - w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx - w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y - 1);
                            bfsNum++;
                        }

                        if (vfwdWarpedIDDistFinal[n][idx + 1 + w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx + 1 + w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y + 1);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx - 1 + w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx - 1 + w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y + 1);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx - 1 - w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx - 1 - w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y - 1);
                            bfsNum++;
                        }
                        if (vfwdWarpedIDDistFinal[n][idx + 1 - w1] > k) {
                            vfwdWarpedIDDistFinal[n][idx + 1 - w1] = k;
                            vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y - 1);
                            bfsNum++;
                        }
                    }
                }
            }
        }
         
    }

    void CoarseDistanceMap::growDistBFSforMF2(int bfsNum, int n) 
    {

        assert(w[0] != 0);
        int w1 = w[1], h1 = h[1];
        for (int k = 1; k < 40; k++) 
        {
            int bfsNum2 = bfsNum;
            std::swap<Eigen::Vector2i *>(vbfsList1[n], vbfsList2[n]);
            bfsNum = 0;

            if (k % 2 == 0) 
            {
                for (int i = 0; i < bfsNum2; i++) 
                {
                    int x = vbfsList2[n][i][0];
                    int y = vbfsList2[n][i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (vfwdWarpedIDDistFinal[n][idx + 1] > k) {
                        vfwdWarpedIDDistFinal[n][idx + 1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx - 1] > k) {
                        vfwdWarpedIDDistFinal[n][idx - 1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx + w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx + w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx - w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx - w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }
                }
            } 
            else 
            {
                for (int i = 0; i < bfsNum2; i++) 
                {
                    int x = vbfsList2[n][i][0];
                    int y = vbfsList2[n][i][1];
                    if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                    int idx = x + y * w1;

                    if (vfwdWarpedIDDistFinal[n][idx + 1] > k) {
                        vfwdWarpedIDDistFinal[n][idx + 1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx - 1] > k) {
                        vfwdWarpedIDDistFinal[n][idx - 1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx + w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx + w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y + 1);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx - w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx - w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x, y - 1);
                        bfsNum++;
                    }

                    if (vfwdWarpedIDDistFinal[n][idx + 1 + w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx + 1 + w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y + 1);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx - 1 + w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx - 1 + w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y + 1);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx - 1 - w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx - 1 - w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x - 1, y - 1);
                        bfsNum++;
                    }
                    if (vfwdWarpedIDDistFinal[n][idx + 1 - w1] > k) {
                        vfwdWarpedIDDistFinal[n][idx + 1 - w1] = k;
                        vbfsList1[n][bfsNum] = Eigen::Vector2i(x + 1, y - 1);
                        bfsNum++;
                    }
                }
            }
        }
    }

    //在点(u, v)附近生成距离场
    void CoarseDistanceMap::addIntoDistFinal(int u, int v) {
        if (w[0] == 0) return;
        bfsList1[0] = Eigen::Vector2i(u, v);
        fwdWarpedIDDistFinal[u + w[1] * v] = 0;
        growDistBFS(1);
    }

    // for multi-fisheye
    void CoarseDistanceMap::addIntoDistFinalforMF(int u, int v, int camnum) {
        if (w[0] == 0) return;
        vbfsList1[camnum][0] = Eigen::Vector2i(u, v);
        vfwdWarpedIDDistFinal[camnum][u + w[1] * v] = 0;
        growDistBFSforMF2(1, camnum);
    }

}
