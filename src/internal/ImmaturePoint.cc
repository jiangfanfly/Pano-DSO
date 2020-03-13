#include "Frame.h"
#include "Settings.h"

#include "internal/GlobalCalib.h"
#include "internal/ImmaturePoint.h"
#include "internal/GlobalFuncs.h"
#include "internal/FrameHessian.h"
#include "internal/ResidualProjections.h"

namespace ldso {

    namespace internal {

        ImmaturePoint::ImmaturePoint(shared_ptr<Frame> hostFrame, shared_ptr<Feature> hostFeat, float type,
                                     shared_ptr<CalibHessian> &HCalib) :
                my_type(type), feature(hostFeat) {
            assert(hostFrame->frameHessian);
            gradH.setZero();
            shared_ptr<FrameHessian> host = hostFrame->frameHessian;
            float u = feature->uv[0], v = feature->uv[1];
            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

                color[idx] = ptc[0];
                if (!std::isfinite(color[idx])) {
                    energyTH = NAN;
                    return;
                }

                gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
                weights[idx] = sqrtf(
                        setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
            }
            energyTH = patternNum * setting_outlierTH;
            energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
        }

        // for multi-fihseye
        ImmaturePoint::ImmaturePoint(shared_ptr<Frame> hostFrame, shared_ptr<Feature> hostFeat, float type, int n, 
            UndistortMultiFisheye* uMF):my_type(type), feature(hostFeat),camnum(n),UMF(uMF)
        {
            assert(hostFrame->frameHessian);
            gradH.setZero();
            shared_ptr<FrameHessian> host = hostFrame->frameHessian;
            float u = feature->uv[0], v = feature->uv[1];
            for (int idx = 0; idx < patternNum; idx++) 
            {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                Vec3f ptc = getInterpolatedElement33BiLin(host->vdI[camnum], u + dx, v + dy, UMF->wPR[0]);

                color[idx] = ptc[0];
                if (!std::isfinite(color[idx])) {
                    energyTH = NAN;
                    return;
                }

                gradH += ptc.tail<2>() * ptc.tail<2>().transpose();
                weights[idx] = sqrtf(
                        setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
            }
            energyTH = patternNum * setting_outlierTH;
            energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
        }

        /*
         * returns
         * * OOB -> point is optimized and marginalized
         * * UPDATED -> point has been updated.
         * * SKIP -> point has not been updated.
         */
        ImmaturePointStatus ImmaturePoint::traceOn(
                shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_KRKi,
                const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine,
                shared_ptr<CalibHessian> HCalib) {

            if (lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;
            float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

            // ============== project min and max. return if one of them is OOB ===================
            // step 1. 检查极线上点的位置
            // check idepthmin, 最近距离  将未成熟的点根据相对位姿和之前的逆深度投影到当前帧上
            Vec3f pr = hostToFrame_KRKi * Vec3f(feature->uv[0], feature->uv[1], 1);
            Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;
            float uMin = ptpMin[0] / ptpMin[2];
            float vMin = ptpMin[1] / ptpMin[2];

            if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5)) {
                // out of boundary
                lastTraceUV = Vec2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;   // 需要被边缘化
            }

            // check idepthmax, maybe infinite
            // 最远距离（注意可能是无限远）
            float dist;
            float uMax;
            float vMax;
            Vec3f ptpMax;   // 按照最远距离来算，在当前帧的投影

            if (std::isfinite(idepth_max)) {
                // 有限远，finite max depth
                ptpMax = pr + hostToFrame_Kt * idepth_max;
                uMax = ptpMax[0] / ptpMax[2];
                vMax = ptpMax[1] / ptpMax[2];

                if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }

                // ============== check their distance. everything below 2px is OK (-> skip). ===================
                dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
                dist = sqrtf(dist);
                if (dist < setting_trace_slackInterval /* =2 by default */ ) {
                    // 极线上两个像素非常接近
                    lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;   // 求坐标均值
                    lastTracePixelInterval = dist;
                    return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
                }
                assert(dist > 0);
            } else {
                dist = maxPixSearch;

                // 任取一个距离，idepth=0.01, so depth=100
                // project to arbitrary depth to get direction.  得到极线
                ptpMax = pr + hostToFrame_Kt * 0.01;
                uMax = ptpMax[0] / ptpMax[2];
                vMax = ptpMax[1] / ptpMax[2];

                // direction.
                float dx = uMax - uMin;
                float dy = vMax - vMin;
                float d = 1.0f / sqrtf(dx * dx + dy * dy);

                // set to [setting_maxPixSearch]. 最大视差搜索
                uMax = uMin + dist * dx * d;
                vMax = vMin + dist * dy * d;

                // may still be out!
                if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }
                assert(dist > 0);
            }

            // set OOB if scale change too big.
            if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5))) {
                lastTraceUV = Vec2f(-1, -1);
                lastTracePixelInterval = 0;
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            }

            // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
            float dx = setting_trace_stepsize * (uMax - uMin);
            float dy = setting_trace_stepsize * (vMax - vMin);
            // // 当b/a接近于0时（此时极线和梯度方向基本平行）,errorInPixel = 0.4 逆深度只更新大约0.4个单位步长；   当b/a大于一定阈值时，则后续步骤直接跳过，该点被标记为IPS_BADCONDITION
            float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));    // 极线与梯度的点乘的平方
            float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));  // 极线旋转90度后与梯度的点乘的平方
            float errorInPixel = 0.2f + 0.2f * (a + b) / a;                   

            if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max)) {   // setting_trace_minImprovementFactor 最小像素间隔
                lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
                lastTracePixelInterval = dist;
                return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
            }

            if (errorInPixel > 10) errorInPixel = 10;

            // ============== do the discrete search ===================  离散搜索
            dx /= dist;
            dy /= dist;

            if (dist > maxPixSearch) {
                uMax = uMin + maxPixSearch * dx;
                vMax = vMin + maxPixSearch * dy;
                dist = maxPixSearch;
            }

            int numSteps = 1.9999f + dist / setting_trace_stepsize;
            Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

            float randShift = uMin * 1000 - floorf(uMin * 1000);
            float ptx = uMin - randShift * dx;
            float pty = vMin - randShift * dy;


            Vec2f rotatetPattern[MAX_RES_PER_POINT];
            for (int idx = 0; idx < patternNum; idx++)
                rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

            if (!std::isfinite(dx) || !std::isfinite(dy)) {
                lastTracePixelInterval = 0;
                lastTraceUV = Vec2f(-1, -1);
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            }

            float errors[100];
            float bestU = 0, bestV = 0, bestEnergy = 1e10;
            int bestIdx = -1;
            if (numSteps >= 100) numSteps = 99;

            for (int i = 0; i < numSteps; i++) {
                float energy = 0;
                for (int idx = 0; idx < patternNum; idx++) {
                    float hitColor = getInterpolatedElement31(frame->dI,
                                                              (float) (ptx + rotatetPattern[idx][0]),
                                                              (float) (pty + rotatetPattern[idx][1]),
                                                              wG[0]);

                    if (!std::isfinite(hitColor)) {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor - (float) (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                    energy += hw * residual * residual * (2 - hw);
                }

                errors[i] = energy;
                if (energy < bestEnergy) {
                    bestU = ptx;
                    bestV = pty;
                    bestEnergy = energy;
                    bestIdx = i;
                }

                ptx += dx;
                pty += dy;
            }


            // find best score outside a +-2px radius.
            float secondBest = 1e10;
            for (int i = 0; i < numSteps; i++) {
                if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) &&
                    errors[i] < secondBest)
                    secondBest = errors[i];
            }
            float newQuality = secondBest / bestEnergy;   // 比较最小误差与次小误差
            if (newQuality < quality || numSteps > 10) quality = newQuality;


            // ============== do GN optimization ===================
            float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
            if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
            int gnStepsGood = 0, gnStepsBad = 0;
            for (int it = 0; it < setting_trace_GNIterations; it++) {
                float H = 1, b = 0, energy = 0;
                for (int idx = 0; idx < patternNum; idx++) {
                    Vec3f hitColor = getInterpolatedElement33(frame->dI,
                                                              (float) (bestU + rotatetPattern[idx][0]),
                                                              (float) (bestV + rotatetPattern[idx][1]), wG[0]);

                    if (!std::isfinite((float) hitColor[0])) {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
                    float dResdDist = dx * hitColor[1] + dy * hitColor[2];
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

                    H += hw * dResdDist * dResdDist;
                    b += hw * residual * dResdDist;
                    energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
                }


                if (energy > bestEnergy) {
                    gnStepsBad++;

                    // do a smaller step from old point.
                    stepBack *= 0.5;
                    bestU = uBak + stepBack * dx;
                    bestV = vBak + stepBack * dy;
                } else {
                    gnStepsGood++;

                    float step = -gnstepsize * b / H;
                    if (step < -0.5) step = -0.5;
                    else if (step > 0.5) step = 0.5;

                    if (!std::isfinite(step)) step = 0;

                    uBak = bestU;
                    vBak = bestV;
                    stepBack = step;

                    bestU += step * dx;
                    bestV += step * dy;
                    bestEnergy = energy;
                }

                if (fabsf(stepBack) < setting_trace_GNThreshold) break;
            }

            // ============== detect energy-based outlier. ===================
            if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH)) {
                lastTracePixelInterval = 0;
                lastTraceUV = Vec2f(-1, -1);
                if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                else
                    return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }

            // ============== set new interval ===================
            if (dx * dx > dy * dy) {   // 当x方向梯度较大时
                idepth_min = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) /
                             (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
                idepth_max = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) /
                             (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
            } else {                   // 当y方向梯度较大时
                idepth_min = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) /
                             (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
                idepth_max = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) /
                             (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
            }
            if (idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);


            if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0)) {
                lastTracePixelInterval = 0;
                lastTraceUV = Vec2f(-1, -1);
                return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }

            lastTracePixelInterval = 2 * errorInPixel;
            lastTraceUV = Vec2f(bestU, bestV);
            return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
        }

        // for multi-fisheye
        /*
         * returns
         * * OOB -> point is optimized and marginalized
         * * UPDATED -> point has been updated.
         * * SKIP -> point has not been updated.
         */
        ImmaturePointStatus ImmaturePoint::traceOnforMF(shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_R, const Vec3f &hostToFrame_t,
                    const vector<vector<Vec2f>> &hostToFrame_affine)
        {
            if (lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;
            float maxPixSearch = (UMF->wPR[0] + UMF->hPR[0]) * setting_maxPixSearch;
            float maxSphPixSearch = 2*M_PI* UMF->GetSphereRadius() * setting_maxPixSearch;

            // ============== project min and max. return if one of them is OOB ===================
            // step 1. 检查极线上点的位置
            // check idepthmin, 最近距离  将未成熟的点根据相对位姿和之前的逆深度投影到当前帧上
            int hostcamnum = feature->camnum;
            float X = feature->xyzs[0];
            float Y = feature->xyzs[1];
            float Z = feature->xyzs[2];
            float SphereRadius = UMF->GetSphereRadius();

            Vec3f pt = hostToFrame_R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) + hostToFrame_t * idepth_min;   
            float S_norm_pt = SphereRadius/pt.norm();
            float xsMin = S_norm_pt*pt(0);
            float ysMin = S_norm_pt*pt(1);
            float zsMin = S_norm_pt*pt(2);

            double uMin, vMin;
            int tocamnum1;
            UMF->LadybugReprojectSpherePtToRectify(xsMin, ysMin, zsMin, &tocamnum1, &uMin, &vMin, 0);

            if (!(uMin > 4 && vMin > 4 && uMin < UMF->wPR[0] - 5 && vMin < UMF->hPR[0] - 5)   // 全景中
                || frame->vmask[0][tocamnum1][(int)uMin + (int)vMin * UMF->wPR[0]] == 0)
            {
                // out of boundary
                lastTraceUV = Vec2f(-1, -1);
                lastTracePixelInterval = 0;
                lastTracePixelIntervalSphere =0;
                return lastTraceStatus = ImmaturePointStatus::IPS_OOB;   // 需要被边缘化
            }

            // check idepthmax, maybe infinite
            // 最远距离（注意可能是无限远）
            float dist;
            double uMax;
            double vMax;
            Vec3f ptpMax;   // 按照最远距离来算，在当前帧的投影
            float distSph;
            double xsMax, ysMax, zsMax;
            int tocamnum2;
            Vec3f XsMin ,XsMax;
            
            if (std::isfinite(idepth_max)) 
            {
                // 有限远，finite max depth

                Vec3f pt = hostToFrame_R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) + hostToFrame_t * idepth_max;  
                //Vec3f pt = R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius)/point->idepth_new + t;  // R21( rho1^-1 * X归一 )+  t21 
                
                float S_norm_pt = SphereRadius/pt.norm();

                xsMax = S_norm_pt*pt(0);
                ysMax = S_norm_pt*pt(1);
                zsMax = S_norm_pt*pt(2);

                UMF->LadybugReprojectSpherePtToRectify(xsMax, ysMax, zsMax, &tocamnum2, &uMax, &vMax, 0);

                if (!(uMax > 4 && vMax > 4 && uMax < UMF->wPR[0] - 5 && vMax < UMF->hPR[0] - 5)) 
                {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }
                if(frame->vmask[0][tocamnum2][(int)uMax + (int)vMax * UMF->wPR[0]] == 0)
                {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }

                samecam = (tocamnum1 == tocamnum2);

                // ============== check their distance. everything below 2px is OK (-> skip). ===================
                dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
                dist = sqrtf(dist);
                // 弧长
                XsMin = Vec3f(xsMin, ysMin, zsMin);
                XsMax = Vec3f(xsMax, ysMax, zsMax);
                distSph = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                if(samecam == true)
                {
                    if (dist < setting_trace_slackInterval /* =2 by default */ ) {
                        // 极线上两个像素非常接近
                        lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;   // 求坐标均值
                        lastTracePixelInterval = dist;
                        return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
                    }
                    assert(dist > 0);
                }
                else
                {
                    // 球面极线距离判断
                    if(distSph < 2 * 0.012*(1 + abs((uMin+uMax)/2 - UMF->wPR[0]/2)/(UMF->wPR[0]/2)));
                    {
                        lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;   // 求坐标均值
                        lastTracePixelInterval = dist;
                        lastTracePixelIntervalSphere = distSph;
                        return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
                    }
                    assert(distSph > 0);
                }
            } 
            else 
            {
                dist = maxPixSearch;
                distSph = maxSphPixSearch;
                // 任取一个距离，idepth=0.01, so depth=100
                // project to arbitrary depth to get direction.  得到极线

                Vec3f pt = hostToFrame_R * Vec3f(X/SphereRadius, Y/SphereRadius, Z/SphereRadius) + hostToFrame_t * 10;  
                float S_norm_pt = SphereRadius/pt.norm();

                xsMax = S_norm_pt*pt(0);
                ysMax = S_norm_pt*pt(1);
                zsMax = S_norm_pt*pt(2);

                UMF->LadybugReprojectSpherePtToRectify(xsMax, ysMax, zsMax, &tocamnum2, &uMax, &vMax, 0);

                samecam = (tocamnum1 == tocamnum2);

                if(samecam == true)
                {
                    // direction.
                    float dx = uMax - uMin;
                    float dy = vMax - vMin;
                    float d = 1.0f / sqrtf(dx * dx + dy * dy);

                    // set to [setting_maxPixSearch]. 最大视差搜索
                    uMax = uMin + dist * dx * d;
                    vMax = vMin + dist * dy * d;

                    UMF->LadybugProjectRectifyPtToSphere(tocamnum2, uMax, uMin, &xsMax, &ysMax, &zsMax, 0);
                }
                else
                {
                    // chrod 弦  弦长
                    //float distchord = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));     
                    XsMin = Vec3f(xsMin, ysMin, zsMin);
                    XsMax = Vec3f(xsMax, ysMax, zsMax);
                    float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                    if(isnan(distarc))
                        distarc = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));
                    float xchord = xsMin + distSph * ((xsMax - xsMin) / distarc);
                    float ychord = ysMin + distSph * ((ysMax - ysMin) / distarc);
                    float zchord = zsMin + distSph * ((zsMax - zsMin) / distarc);

                    Vec3f Xchord = Vec3f(xchord, ychord, zchord);

                    // xsMax = xchord * SphereRadius / Xchord.norm();
                    // ysMax = ychord * SphereRadius / Xchord.norm();
                    // zsMax = zchord * SphereRadius / Xchord.norm();
                    // UMF->LadybugReprojectSpherePtToRectify(xsMax, ysMax, zsMax, &tocamnum2, &uMax, &vMax, 0);
                    float tempxsMax = xchord * SphereRadius / Xchord.norm();
                    float tempysMax = ychord * SphereRadius / Xchord.norm();
                    float tempzsMax = zchord * SphereRadius / Xchord.norm();
                    UMF->LadybugReprojectSpherePtToRectify(tempxsMax, tempysMax, tempzsMax, &tocamnum2, &uMax, &vMax, 0);
                    xsMax = tempxsMax; zsMax = tempzsMax; ysMax = tempzsMax;
                }
                
                // may still be out!
                if (!(uMax > 4 && vMax > 4 && uMax < UMF->wPR[0] - 5 && vMax < UMF->hPR[0] - 5)
                    || frame->vmask[0][tocamnum2][(int)uMax + (int)vMax * UMF->wPR[0]] == 0) 
                {
                    lastTraceUV = Vec2f(-1, -1);
                    lastTracePixelInterval = 0;
                    lastTracePixelIntervalSphere = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                }
                assert(dist > 0);
            }

            Vec2f results;
            float errorInPixel;
            if(samecam == true)
            {
                results = doSearchOptimizeOnPlane(uMax, vMax, uMin, vMin, dist, maxPixSearch, frame, 
                                                    hostToFrame_R, hostToFrame_t, hostToFrame_affine, hostcamnum, tocamnum1);
                errorInPixel = results(0);
                if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max)) 
                {   // setting_trace_minImprovementFactor 最小像素间隔
                    lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
                    lastTracePixelInterval = dist;
                    lastTracePixelIntervalSphere = 0;
                    return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
                }
            }
            else
            {
                if (std::isfinite(idepth_max)) 
                {   // setting_trace_minImprovementFactor 最小像素间隔
                    lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
                    lastTracePixelInterval = 0;
                    lastTracePixelIntervalSphere = distSph;
                    return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
                }
                results = doSearchOptimizeOnSphere(uMax, vMax, uMin, vMin, xsMin, ysMin, zsMin, xsMax, ysMax, zsMax, distSph, maxSphPixSearch, frame, 
                                                    hostToFrame_R, hostToFrame_t, hostToFrame_affine, hostcamnum, tocamnum1, tocamnum2);

            }
            float bestEnergy = results(1);

            // ============== detect energy-based outlier. ===================
            if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH)) {
                lastTracePixelInterval = 0;
                lastTracePixelIntervalSphere = 0;
                lastTraceUV = Vec2f(-1, -1);
                if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                    return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                else
                    return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }



            if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0)) 
            {
                lastTracePixelInterval = 0;
                lastTracePixelIntervalSphere = 0;
                lastTraceUV = Vec2f(-1, -1);
                return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
            }

            lastTracePixelInterval = 2 * errorInPixel;
            //lastTraceUV = Vec2f(bestU, bestV);
            return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
        }

        Vec2f ImmaturePoint::doSearchOptimizeOnPlane(float uMax, float vMax, float uMin, float vMin, float dist, float maxPixSearch, shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_R, const Vec3f &hostToFrame_t,
                    const vector<vector<Vec2f>> &hostToFrame_affine, int hostcamnum, int tocamnum)
        {
            float SphereRadius = UMF->GetSphereRadius();
             // ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
            float dx = setting_trace_stepsize * (uMax - uMin);
            float dy = setting_trace_stepsize * (vMax - vMin);
            // // 当b/a接近于0时（此时极线和梯度方向基本平行）,errorInPixel = 0.4 逆深度只更新大约0.4个单位步长；   当b/a大于一定阈值时，则后续步骤直接跳过，该点被标记为IPS_BADCONDITION
            float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));    // 极线与梯度的点乘的平方
            float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));  // 极线旋转90度后与梯度的点乘的平方
            float errorInPixel = 0.2f + 0.2f * (a + b) / a;                   
            if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max)) 
                return Vec2f(errorInPixel,0);
            

            if (errorInPixel > 10) errorInPixel = 10;

            // ============== do the discrete search ===================  离散搜索
            dx /= dist;
            dy /= dist;

            if (dist > maxPixSearch) {
                uMax = uMin + maxPixSearch * dx;
                vMax = vMin + maxPixSearch * dy;
                dist = maxPixSearch;
            }

            int numSteps = 1.9999f + dist / setting_trace_stepsize;
            //Mat22f Rplane = hostToFrame_R.topLeftCorner<2, 2>();

            float randShift = uMin * 1000 - floorf(uMin * 1000);
            float ptx = uMin - randShift * dx;
            float pty = vMin - randShift * dy;


            // Vec2f rotatetPattern[MAX_RES_PER_POINT];
            // for (int idx = 0; idx < patternNum; idx++)
            //     rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

            if (!std::isfinite(dx) || !std::isfinite(dy)) {
                lastTracePixelInterval = 0;
                lastTracePixelIntervalSphere =0;
                lastTraceUV = Vec2f(-1, -1);
                lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                return Vec2f(0,0);
            }

            float errors[100];
            float bestU = 0, bestV = 0, bestEnergy = 1e10;
            int bestIdx = -1;
            if (numSteps >= 100) numSteps = 99;

            for (int i = 0; i < numSteps; i++) {
                float energy = 0;
                for (int idx = 0; idx < patternNum; idx++) {
                    float hitColor = getInterpolatedElement31(frame->vdI[tocamnum],
                                                              (float) (ptx + patternP[idx][0]),
                                                              (float) (pty + patternP[idx][1]),
                                                               UMF->wPR[0]);

                    if (!std::isfinite(hitColor)) {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor - (float) (hostToFrame_affine[hostcamnum][tocamnum][0] * color[idx] + hostToFrame_affine[hostcamnum][tocamnum][1]);
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                    energy += hw * residual * residual * (2 - hw);
                }

                errors[i] = energy;
                if (energy < bestEnergy) {
                    bestU = ptx;
                    bestV = pty;
                    bestEnergy = energy;
                    bestIdx = i;
                }

                ptx += dx;
                pty += dy;
            }


            // find best score outside a +-2px radius.
            float secondBest = 1e10;
            for (int i = 0; i < numSteps; i++) {
                if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) &&
                    errors[i] < secondBest)
                    secondBest = errors[i];
            }
            float newQuality = secondBest / bestEnergy;   // 比较最小误差与次小误差
            if (newQuality < quality || numSteps > 10) quality = newQuality;


            // ============== do GN optimization ===================
            float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
            if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
            int gnStepsGood = 0, gnStepsBad = 0;
            for (int it = 0; it < setting_trace_GNIterations; it++) {
                float H = 1, b = 0, energy = 0;
                for (int idx = 0; idx < patternNum; idx++) {
                    Vec3f hitColor = getInterpolatedElement33(frame->vdI[tocamnum],
                                                              (float) (bestU + patternP[idx][0]),
                                                              (float) (bestV + patternP[idx][1]), 
                                                              UMF->wPR[0]);

                    if (!std::isfinite((float) hitColor[0])) {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor[0] - (hostToFrame_affine[hostcamnum][tocamnum][0] * color[idx] + hostToFrame_affine[hostcamnum][tocamnum][1]);
                    float dResdDist = dx * hitColor[1] + dy * hitColor[2];
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

                    H += hw * dResdDist * dResdDist;
                    b += hw * residual * dResdDist;
                    energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
                }


                if (energy > bestEnergy) {
                    gnStepsBad++;

                    // do a smaller step from old point.
                    stepBack *= 0.5;
                    bestU = uBak + stepBack * dx;
                    bestV = vBak + stepBack * dy;
                } else {
                    gnStepsGood++;

                    float step = -gnstepsize * b / H;
                    if (step < -0.5) step = -0.5;
                    else if (step > 0.5) step = 0.5;

                    if (!std::isfinite(step)) step = 0;

                    uBak = bestU;
                    vBak = bestV;
                    stepBack = step;

                    bestU += step * dx;
                    bestV += step * dy;
                    bestEnergy = energy;
                }

                if (fabsf(stepBack) < setting_trace_GNThreshold) break;
            }

            vector<float> vd;
            //vd.resize(12);
            float d;
            // 逆深度信息更新
        
                float umin = bestU - errorInPixel * dx;
                float vmin = bestV - errorInPixel * dy;
                double minXs, minYs, minZs;
                UMF->LadybugProjectRectifyPtToSphere(tocamnum, umin, vmin, &minXs, &minYs, &minZs, 0);
                Vec3f axx = hostToFrame_R * Vec3f(minXs, minYs, minYs)/SphereRadius;
                float a1 = axx(0);
                float a2 = axx(1);
                float a3 = axx(2);
                float t1 = hostToFrame_t(0);
                float t2 = hostToFrame_t(1);
                float t3 = hostToFrame_t(2);

                float Ax = minXs*minXs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t1*t1;
                float Bx = minXs*minXs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a1*t1;
                float Cx = minXs*minXs*axx.squaredNorm()-SphereRadius*SphereRadius*a1*a1;

                d = (-Bx + sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bx - sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);

                float Ay = minYs*minYs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t2*t2;
                float By = minYs*minYs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a2*t2;
                float Cy = minYs*minYs*axx.squaredNorm()-SphereRadius*SphereRadius*a2*a2;

                d = (-By + sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);
                d = (-By - sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);

                float Az = minZs*minZs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t3*t3;
                float Bz = minZs*minZs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a3*t3;
                float Cz = minZs*minZs*axx.squaredNorm()-SphereRadius*SphereRadius*a3*a3;

                d = (-Bz + sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bz - sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);


                float umax = bestU + errorInPixel * dx;
                float vmax = bestV + errorInPixel * dy;
                double maxXs, maxYs, maxZs;
                UMF->LadybugProjectRectifyPtToSphere(tocamnum, umax, vmax, &maxXs, &maxYs, &maxZs, 0);
                axx = hostToFrame_R * Vec3f(maxXs, maxYs, maxYs)/SphereRadius;
                a1 = axx(0);
                a2 = axx(1);
                a3 = axx(2);
                t1 = hostToFrame_t(0);
                t2 = hostToFrame_t(1);
                t3 = hostToFrame_t(2);

                Ax = maxXs*maxXs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t1*t1;
                Bx = maxXs*maxXs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a1*t1;
                Cx =maxXs*maxXs*axx.squaredNorm()-SphereRadius*SphereRadius*a1*a1;

                d = (-Bx + sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bx - sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);

                Ay = maxYs*maxYs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t2*t2;
                By = maxYs*maxYs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a2*t2;
                Cy = maxYs*maxYs*axx.squaredNorm()-SphereRadius*SphereRadius*a2*a2;

                d = (-By + sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);
                d = (-By - sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);

                Az = maxZs*maxZs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t3*t3;
                Bz = maxZs*maxZs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a3*t3;
                Cz = maxZs*maxZs*axx.squaredNorm()-SphereRadius*SphereRadius*a3*a3;

                d = (-Bz + sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bz - sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);
            if(vd.empty() == true)
            {
                lastTracePixelInterval = 0;
                lastTracePixelIntervalSphere =0;
                lastTraceUV = Vec2f(-1, -1);
                lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                return Vec2f(0,0);
            }
            idepth_max = *max_element(vd.begin(),vd.end());  
            idepth_min = *min_element(vd.begin(),vd.end()); 
            lastTraceUV = Vec2f(bestU, bestV);

            return Vec2f(errorInPixel, bestEnergy); 
        }

// update
        Vec2f ImmaturePoint::doSearchOptimizeOnSphere(float uMax, float uMin, float vMax, float vMin,float xsMin, float ysMin, float zsMin, float xsMax, float ysMax, float zsMax, float distSph, float maxSphPixSearch, shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_R, const Vec3f &hostToFrame_t,
                    const vector<vector<Vec2f>> &hostToFrame_affine, int hostcamnum, int tocamnum1, int tocamnum2)
        {       
            float SphereRadius = UMF->GetSphereRadius();


            // ============== do the discrete search ===================  离散搜索
            if (distSph > maxSphPixSearch) {
                Vec3f XsMin = Vec3f(xsMin, ysMin, zsMin);
                Vec3f XsMax = Vec3f(xsMax, ysMax, zsMax);
                float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                float xchord = xsMin + (xsMax - xsMin) * (distSph / distarc);
                float ychord = ysMin + (ysMax - ysMin) * (distSph / distarc);
                float zchord = zsMin + (zsMax - zsMin) * (distSph / distarc);

                Vec3f Xchord = Vec3f(xchord, ychord, zchord);

                xsMax = xchord * SphereRadius / Xchord.norm();
                ysMax = ychord * SphereRadius / Xchord.norm();
                zsMax = zchord * SphereRadius / Xchord.norm();
                distSph = maxSphPixSearch;

                double x = xsMax;
                double y = ysMax;
                double z = zsMax;
                double u,v;

                UMF->LadybugReprojectSpherePtToRectify(x, y, z, &tocamnum2, &u, &v, 0);
                uMax = u;
                vMax = v; 
            }

            // 球面极线搜索
            float numSteps = distSph / setting_trace_stepsize;
            // if (!std::isfinite(dx) || !std::isfinite(dy)) {
            //     lastTracePixelInterval = 0;
            //     lastTraceUV = Vec2f(-1, -1);
            //     return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
            // }
            Vec3f XsMin(xsMin, ysMin, zsMin);
            Vec3f XsMax (xsMax, ysMax, zsMax);
            float dxs = (xsMax - xsMin)/distSph;
            double ptx, pty, xs,ys,zs;
            xs = xsMin;
            ys = ysMin;
            zs = zsMin;
            float addarc = 0.0; // 从最小处开始加的弧长 
            float errors[350];
            float bestU = 0, bestV = 0, bestEnergy = 1e10;
            float bestx = 0, besty = 0, bestz = 0;
            float bestarc = 0;
            int bestCamnum =-1;
            int bestIdx = -1;

            float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
            if(isnan(distarc))
                distarc = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));
            // arc弧长每次增加 0.012 计算xs, ys， zs
            for (int i = 0; (i * 0.012) < numSteps; i++) 
            {
                float energy = 0;
                int toc;
                UMF->LadybugReprojectSpherePtToRectify(xs, ys, zs, &toc, &ptx, &pty, 0);
                if(ptx <0 || pty <0 || ptx > UMF->wPR[0] || pty >UMF->hPR[0])
                {
                    int kk=0;
                }
                for (int idx = 0; idx < patternNum; idx++) 
                {
                    float hitColor = getInterpolatedElement31(frame->vdI[toc],
                                                                (float) (ptx + patternP[idx][0]),
                                                                (float) (pty + patternP[idx][1]),
                                                                UMF->wPR[0]);

                    if (!std::isfinite(hitColor)) 
                    {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor - (float) (hostToFrame_affine[hostcamnum][toc][0] * color[idx] + hostToFrame_affine[hostcamnum][toc][1]);
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                    energy += hw * residual * residual * (2 - hw);
                }

                errors[i] = energy;
                if (energy < bestEnergy) 
                {
                    bestU = ptx;
                    bestV = pty;
                    bestx = xs;
                    besty = ys;
                    bestz = zs;
                    bestEnergy = energy;
                    bestIdx = i;
                    bestCamnum = toc;
                }
                addarc += 0.012;
                
                float xchord = xsMin + addarc * ((xsMax - xsMin) / distarc);
                float ychord = ysMin + addarc * ((ysMax - ysMin) / distarc);
                float zchord = zsMin + addarc * ((zsMax - zsMin) / distarc);

                Vec3f Xchord = Vec3f(xchord, ychord, zchord);

                xs = xchord * SphereRadius / Xchord.norm();
                ys = ychord * SphereRadius / Xchord.norm();
                zs = zchord * SphereRadius / Xchord.norm();
                //UMF->LadybugReprojectSpherePtToRectify(xsMax, ysMax, zsMax, &tocamnum2, &ptx, &pty, 0);
            }


            float secondBest = 1e10;
            for (int i = 0; i < numSteps; i++) {
                if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) &&
                    errors[i] < secondBest)
                    secondBest = errors[i];
            }
            float newQuality = secondBest / bestEnergy;   // 比较最小误差与次小误差
            if (newQuality < quality || numSteps > 10) quality = newQuality;

            float dx, dy, dist;
            if(bestCamnum == tocamnum1)  // 与min相同
            {
                dx = setting_trace_stepsize * abs(bestU - uMin);
                dy = setting_trace_stepsize * abs(bestV - vMin);
                dist = sqrt( (bestU - uMin) * (bestU - uMin) + (bestV - vMin) * (bestV - vMin));

                dx /= dist;
                dy /= dist;
            }
            else if(bestCamnum == tocamnum2) 
            {
                dx = setting_trace_stepsize * abs(bestU - uMax);
                dy = setting_trace_stepsize * abs(bestV - vMax);
                dist = sqrt( (bestU - uMax) * (bestU - uMax) + (bestV - vMax) * (bestV - vMax));

                dx /= dist;
                dy /= dist;
            }

            // LOG(INFO) << "release optimize on sphere to 4" << endl;

            // ============== do GN optimization on sphere ===================
            float xBak = bestx, yBak = besty, zBak = bestz, gnstepsize = 1, stepBack = 0, addarcBak = 0;
            addarc = 0;
            if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
            int gnStepsGood = 0, gnStepsBad = 0;
            for (int it = 0; it < setting_trace_GNIterations; it++) 
            {
                float H = 1, l = 0, energy = 0;
                for (int idx = 0; idx < patternNum; idx++) 
                {
                    Vec3f hitColor = getInterpolatedElement33(frame->vdI[bestCamnum],
                                                              (float) (bestU + patternP[idx][0]),
                                                              (float) (bestV + patternP[idx][1]), UMF->wPR[0]);

                    if (!std::isfinite((float) hitColor[0])) 
                    {
                        energy += 1e5;
                        continue;
                    }
                    float residual = hitColor[0] - (hostToFrame_affine[hostcamnum][bestCamnum][0] * color[idx] + hostToFrame_affine[hostcamnum][bestCamnum][1]);
                    float dResdDist = dx * hitColor[1] + dy * hitColor[2];
                    float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

                    H += hw * dResdDist * dResdDist;
                    l += hw * residual * dResdDist;
                    energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
                }

                if (energy > bestEnergy) {
                    gnStepsBad++;

                    // do a smaller step from old point.
                    stepBack *= 0.5;
                    addarcBak += stepBack;
                    float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                    if(isnan(distarc))
                        distarc = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));
                    float xchord = xsMin + addarcBak * ((xsMax - xsMin) / distarc);
                    float ychord = ysMin + addarcBak * ((ysMax - ysMin) / distarc);
                    float zchord = zsMin + addarcBak * ((zsMax - zsMin) / distarc);

                    Vec3f Xchord = Vec3f(xchord, ychord, zchord);

                    bestx = xchord * SphereRadius / Xchord.norm();
                    besty = ychord * SphereRadius / Xchord.norm();
                    bestz = zchord * SphereRadius / Xchord.norm();
                    int temptcc;
                    double u,v;
                    UMF->LadybugReprojectSpherePtToRectify(bestx, besty, bestz, &temptcc, &u, &v, 0);
                    bestU = u; bestV = v;
                    bestarc = addarcBak;

                } else {
                    gnStepsGood++;

                    float step = -gnstepsize * l / H;
                    if (step*0.012 < -0.05) step = -0.05;
                    else if (step*0.012 > 0.05) step = 0.05;  // 球面

                    if (!std::isfinite(step)) step = 0;

                    xBak = bestx;
                    yBak = besty;
                    zBak = bestz;
                    stepBack = step*0.012;

                    addarcBak = addarc;
                    addarc += stepBack;
                    float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                    if(isnan(distarc))
                        distarc = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));
                    float xchord = xsMin + addarc * ((xsMax - xsMin) / distarc);
                    float ychord = ysMin + addarc * ((ysMax - ysMin) / distarc);
                    float zchord = zsMin + addarc * ((zsMax - zsMin) / distarc);

                    Vec3f Xchord = Vec3f(xchord, ychord, zchord);

                    bestx = xchord * SphereRadius / Xchord.norm();
                    besty = ychord * SphereRadius / Xchord.norm();
                    bestz = zchord * SphereRadius / Xchord.norm();
                    int temptcc;
                    double u,v;
                    UMF->LadybugReprojectSpherePtToRectify(bestx, besty, bestz, &temptcc, &u, &v, 0);
                    bestU = u; bestV = v;
                    bestEnergy = energy;
                    bestarc = addarc;
                    
                }

                if (fabsf(stepBack) < setting_trace_GNThreshold * 0.12) break;
            }

            // LOG(INFO) << "release optimize on sphere to 5" << endl;

            vector<float> vd;
            //vd.resize(12);
            float d;
            // 逆深度信息更新
            {
                float minXs, minYs, minZs;
                float minarc = bestarc + 0.012 * 0.4;
                float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                if(isnan(distarc))
                    distarc = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));
                float xchord = xsMin + addarc * ((xsMax - xsMin) / distarc);
                float ychord = ysMin + addarc * ((ysMax - ysMin) / distarc);
                float zchord = zsMin + addarc * ((zsMax - zsMin) / distarc);

                Vec3f Xchord = Vec3f(xchord, ychord, zchord);

                minXs = xchord * SphereRadius / Xchord.norm();
                minYs = ychord * SphereRadius / Xchord.norm();
                minZs = zchord * SphereRadius / Xchord.norm();


                Vec3f axx = hostToFrame_R * Vec3f(minXs, minYs, minYs)/SphereRadius;
                float a1 = axx(0);
                float a2 = axx(1);
                float a3 = axx(2);
                float t1 = hostToFrame_t(0);
                float t2 = hostToFrame_t(1);
                float t3 = hostToFrame_t(2);

                float Ax = minXs*minXs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t1*t1;
                float Bx = minXs*minXs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a1*t1;
                float Cx = minXs*minXs*axx.squaredNorm()-SphereRadius*SphereRadius*a1*a1;

                d = (-Bx + sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bx - sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);

                float Ay = minYs*minYs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t2*t2;
                float By = minYs*minYs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a2*t2;
                float Cy = minYs*minYs*axx.squaredNorm()-SphereRadius*SphereRadius*a2*a2;

                d = (-By + sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);
                d = (-By - sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);

                float Az = minZs*minZs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t3*t3;
                float Bz = minZs*minZs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a3*t3;
                float Cz = minZs*minZs*axx.squaredNorm()-SphereRadius*SphereRadius*a3*a3;

                d = (-Bz + sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bz - sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);

                float maxXs, maxYs, maxZs;
                float maxarc = bestarc - 0.012 * 0.4;
                distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
                if(isnan(distarc))
                    distarc = sqrt((xsMax - xsMin) * (xsMax - xsMin) + (ysMax - ysMin) * (ysMax - ysMin) + (zsMax - zsMin) * (zsMax - zsMin));
                xchord = xsMin + addarc * ((xsMax - xsMin) / distarc);
                ychord = ysMin + addarc * ((ysMax - ysMin) / distarc);
                zchord = zsMin + addarc * ((zsMax - zsMin) / distarc);

                Xchord = Vec3f(xchord, ychord, zchord);

                maxXs = xchord * SphereRadius / Xchord.norm();
                maxYs = ychord * SphereRadius / Xchord.norm();
                maxZs = zchord * SphereRadius / Xchord.norm();


                axx = hostToFrame_R * Vec3f(maxXs, maxYs, maxYs)/SphereRadius;
                a1 = axx(0);
                a2 = axx(1);
                a3 = axx(2);
                t1 = hostToFrame_t(0);
                t2 = hostToFrame_t(1);
                t3 = hostToFrame_t(2);

                Ax = maxXs*maxXs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t1*t1;
                Bx = maxXs*maxXs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a1*t1;
                Cx =maxXs*maxXs*axx.squaredNorm()-SphereRadius*SphereRadius*a1*a1;

                d = (-Bx + sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bx - sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
                if(d >0)
                    vd.emplace_back(d);

                Ay = maxYs*maxYs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t2*t2;
                By = maxYs*maxYs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a2*t2;
                Cy = maxYs*maxYs*axx.squaredNorm()-SphereRadius*SphereRadius*a2*a2;

                d = (-By + sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);
                d = (-By - sqrt(By*By-4*Ay*Cy)) / (2*Ay);
                if(d >0)
                    vd.emplace_back(d);

                Az = maxZs*maxZs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t3*t3;
                Bz = maxZs*maxZs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a3*t3;
                Cz = maxZs*maxZs*axx.squaredNorm()-SphereRadius*SphereRadius*a3*a3;

                d = (-Bz + sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);
                d = (-Bz - sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
                if(d >0)
                    vd.emplace_back(d);
            }
            if(vd.empty() == true)
            {
                lastTracePixelInterval = 0;
                lastTracePixelIntervalSphere =0;
                lastTraceUV = Vec2f(-1, -1);
                lastTraceStatus = ImmaturePointStatus::IPS_OOB;
                return Vec2f(0,0);
            }
            idepth_max = *max_element(vd.begin(),vd.end());  
            idepth_min = *min_element(vd.begin(),vd.end()); 
            lastTraceUV = Vec2f(bestU, bestV); 

            // LOG(INFO) << "release optimize on sphere to over" << endl;

            return Vec2f(0.4, bestEnergy);
        }

//
// 

//
        // Vec2f ImmaturePoint::doSearchOptimizeOnSphere(float uMax, float uMin, float vMax, float vMin,float xsMin, float ysMin, float zsMin, float xsMax, float ysMax, float zsMax, float distSph, float maxSphPixSearch, shared_ptr<FrameHessian> frame, const Mat33f &hostToFrame_R, const Vec3f &hostToFrame_t,
        //             const vector<vector<Vec2f>> &hostToFrame_affine, int hostcamnum, int tocamnum1, int tocamnum2)
        // {       
        //     float SphereRadius = UMF->GetSphereRadius();


        //     // ============== do the discrete search ===================  离散搜索
        //     if (distSph > maxSphPixSearch) {
        //         Vec3f XsMin = Vec3f(xsMin, ysMin, zsMin);
        //         Vec3f XsMax = Vec3f(xsMax, ysMax, zsMax);
        //         float distarc = SphereRadius * acos( XsMax.dot(XsMin) / (XsMax.norm() * XsMin.norm()));
        //         float xchord = xsMin + (xsMax - xsMin) * (distSph / distarc);
        //         float ychord = ysMin + (ysMax - ysMin) * (distSph / distarc);
        //         float zchord = zsMin + (zsMax - zsMin) * (distSph / distarc);

        //         Vec3f Xchord = Vec3f(xchord, ychord, zchord);

        //         xsMax = xchord * SphereRadius / Xchord.norm();
        //         ysMax = ychord * SphereRadius / Xchord.norm();
        //         zsMax = zchord * SphereRadius / Xchord.norm();
        //         distSph = maxSphPixSearch;

        //         double x = xsMax;
        //         double y = ysMax;
        //         double z = zsMax;
        //         double u,v;

        //         UMF->LadybugReprojectSpherePtToRectify(x, y, z, &tocamnum2, &u, &v, 0);
        //         uMax = u;
        //         vMax = v; 
        //     }

        //     // 球面极线搜索
        //     int numSteps = distSph / maxSphPixSearch;
        //     // if (!std::isfinite(dx) || !std::isfinite(dy)) {
        //     //     lastTracePixelInterval = 0;
        //     //     lastTraceUV = Vec2f(-1, -1);
        //     //     return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
        //     // }
        //     Vec3f XsMin(xsMin, ysMin, zsMin);
        //     Vec3f XsMax (xsMax, ysMax, zsMax);
        //     float dxs = (xsMax - xsMin)/distSph;
        //     double ptx, pty, xs,ys,zs;
        //     xs = xsMin;
        //     ys = ysMin;
        //     zs = zsMin;
        //     Vec3f abc = XsMax.cross(XsMin);
        //     float a = abc(0);
        //     float b = abc(1);
        //     float c = abc(2);

        //     float errors[100];
        //     float bestU = 0, bestV = 0, bestEnergy = 1e10;
        //     float bestx = 0, besty = 0, bestz = 0;
        //     int bestCamnum =-1;
        //     int bestIdx = -1;
        //     if (numSteps >= 100) numSteps = 99;

        //     for (int i = 0; i < numSteps; i++) 
        //     {
        //         float energy = 0;
        //         int toc;
        //         UMF->LadybugReprojectSpherePtToRectify(xs, ys, zs, &toc, &ptx, &pty, 0);
        //         for (int idx = 0; idx < patternNum; idx++) 
        //         {
        //             float hitColor = getInterpolatedElement31(frame->vdI[toc],
        //                                                       (float) (ptx + patternP[idx][0]),
        //                                                       (float) (pty + patternP[idx][1]),
        //                                                       UMF->wPR[0]);

        //             if (!std::isfinite(hitColor)) 
        //             {
        //                 energy += 1e5;
        //                 continue;
        //             }
        //             float residual = hitColor - (float) (hostToFrame_affine[hostcamnum][toc][0] * color[idx] + hostToFrame_affine[hostcamnum][toc][1]);
        //             float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
        //             energy += hw * residual * residual * (2 - hw);
        //         }

        //         errors[i] = energy;
        //         if (energy < bestEnergy) 
        //         {
        //             bestU = ptx;
        //             bestV = pty;
        //             bestx = xs;
        //             besty = ys;
        //             bestz = zs;
        //             bestEnergy = energy;
        //             bestIdx = i;
        //             bestCamnum = toc;
        //         }

        //         xs = xs + dxs;
        //         float A = b*b + c * c;
        //         float B = 2*a*xs*c;
        //         float C =b*b*(xs*xs-SphereRadius* SphereRadius) + a*a*xs*xs;
        //         float zs1,zs2;
        //         zs1 =  (-B + sqrt(B*B-4*A*C) )/(2*A);
        //         zs2 =  (-B - sqrt(B*B-4*A*C) )/(2*A);
        //         if(abs(zs1-zs) < abs(zs2-zs))
        //             zs = zs1;
        //         else
        //             zs = zs2;
        //         ys = -(a*xs + c*zs)/b;
        //     }


        //     float secondBest = 1e10;
        //     for (int i = 0; i < numSteps; i++) {
        //         if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) &&
        //             errors[i] < secondBest)
        //             secondBest = errors[i];
        //     }
        //     float newQuality = secondBest / bestEnergy;   // 比较最小误差与次小误差
        //     if (newQuality < quality || numSteps > 10) quality = newQuality;

        //     float dx, dy, dist;
        //     if(bestCamnum == tocamnum1)  // 与min相同
        //     {
        //         dx = setting_trace_stepsize * abs(bestU - uMin);
        //         dy = setting_trace_stepsize * abs(bestV - vMin);
        //         dist = sqrt( (bestU - uMin) * (bestU - uMin) + (bestV - vMin) * (bestV - vMin));

        //         dx /= dist;
        //         dy /= dist;
        //     }
        //     else if(bestCamnum == tocamnum2) 
        //     {
        //         dx = setting_trace_stepsize * abs(bestU - uMax);
        //         dy = setting_trace_stepsize * abs(bestV - vMax);
        //         dist = sqrt( (bestU - uMax) * (bestU - uMax) + (bestV - vMax) * (bestV - vMax));

        //         dx /= dist;
        //         dy /= dist;
        //     }

        //     // ============== do GN optimization on sphere ===================
        //     float xBak = bestx, yBak = besty, zBak = bestz, gnstepsize = 1, stepBack = 0;
        //     if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
        //     int gnStepsGood = 0, gnStepsBad = 0;
        //     for (int it = 0; it < setting_trace_GNIterations; it++) 
        //     {
        //         float H = 1, l = 0, energy = 0;
        //         for (int idx = 0; idx < patternNum; idx++) 
        //         {
        //             Vec3f hitColor = getInterpolatedElement33(frame->vdI[bestCamnum],
        //                                                       (float) (bestU + patternP[idx][0]),
        //                                                       (float) (bestV + patternP[idx][1]), UMF->wPR[0]);

        //             if (!std::isfinite((float) hitColor[0])) 
        //             {
        //                 energy += 1e5;
        //                 continue;
        //             }
        //             float residual = hitColor[0] - (hostToFrame_affine[hostcamnum][bestCamnum][0] * color[idx] + hostToFrame_affine[hostcamnum][bestCamnum][1]);
        //             float dResdDist = dx * hitColor[1] + dy * hitColor[2];
        //             float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

        //             H += hw * dResdDist * dResdDist;
        //             l += hw * residual * dResdDist;
        //             energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
        //         }

        //         if (energy > bestEnergy) {
        //             gnStepsBad++;

        //             // do a smaller step from old point.
        //             stepBack *= 0.5;

        //             bestx = xBak + stepBack * dxs;
        //             float A = b*b + c * c;
        //             float B = 2*a*bestx*c;
        //             float C =b*b*(bestx*bestx-SphereRadius* SphereRadius) + a*a*bestx*bestx;
        //             float zs1,zs2;
        //             zs1 =  (-B + sqrt(B*B-4*A*C) )/(2*A);
        //             zs2 =  (-B - sqrt(B*B-4*A*C) )/(2*A);
        //             if(abs(zs1-bestz) < abs(zs2-bestz))
        //                 bestz = zs1;
        //             else
        //                 bestz = zs2;
        //             besty = -(a*bestx + c*bestz)/b;
        //             double u,v;
        //             UMF->LadybugReprojectSpherePtToRectify(bestx, besty, bestz, &bestCamnum, &u, &v, 0);
        //             bestU = u; bestV = v;
        //         } else {
        //             gnStepsGood++;

        //             float step = -gnstepsize * l / H;
        //             if (step < -0.5) step = -0.5;
        //             else if (step > 0.5) step = 0.5;

        //             if (!std::isfinite(step)) step = 0;

        //             xBak = bestx;
        //             yBak = besty;
        //             zBak = bestz;
        //             stepBack = step;

        //             bestx = bestx + stepBack * dxs;
        //             float A = b*b + c * c;
        //             float B = 2*a*bestx*c;
        //             float C =b*b*(bestx*bestx-SphereRadius* SphereRadius) + a*a*bestx*bestx;
        //             float zs1,zs2;
        //             zs1 =  (-B + sqrt(B*B-4*A*C) )/(2*A);
        //             zs2 =  (-B - sqrt(B*B-4*A*C) )/(2*A);
        //             if(abs(zs1-bestz) < abs(zs2-bestz))
        //                 bestz = zs1;
        //             else
        //                 bestz = zs2;
        //             besty = -(a*bestx + c*bestz)/b;
        //             double u,v;
        //             UMF->LadybugReprojectSpherePtToRectify(bestx, besty, bestz, &bestCamnum, &u, &v, 0);
        //             bestU = u; bestV = v;
        //             bestEnergy = energy;
        //         }

        //         if (fabsf(stepBack) < setting_trace_GNThreshold) break;
        //     }
        //     vector<float> vd;
        //     vd.resize(12);
        //     // 逆深度信息更新
        //     {
        //         float minXs, minYs, minZs;
        //         minXs = bestx + 0.4 * dxs;
        //         float A = b*b + c * c;
        //         float B = 2*a*minXs*c;
        //         float C =b*b*(minXs*minXs-SphereRadius* SphereRadius) + a*a*minXs*minXs;
        //         float zs1,zs2;
        //         zs1 =  (-B + sqrt(B*B-4*A*C) )/(2*A);
        //         zs2 =  (-B - sqrt(B*B-4*A*C) )/(2*A);
        //         if(abs(zs1-bestz) < abs(zs2-bestz))
        //             minZs = zs1;
        //         else
        //             minZs = zs2;
        //         minYs = -(a*minXs + c*minZs)/b;


        //         Vec3f axx = hostToFrame_R * Vec3f(minXs, minYs, minYs)/SphereRadius;
        //         float a1 = axx(0);
        //         float a2 = axx(1);
        //         float a3 = axx(2);
        //         float t1 = hostToFrame_t(0);
        //         float t2 = hostToFrame_t(1);
        //         float t3 = hostToFrame_t(2);

        //         float Ax = minXs*minXs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t1*t1;
        //         float Bx = minXs*minXs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a1*t1;
        //         float Cx = minXs*minXs*axx.squaredNorm()-SphereRadius*SphereRadius*a1*a1;

        //         vd[0] = (-Bx + sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
        //         vd[1] = (-Bx - sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);

        //         float Ay = minYs*minYs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t2*t2;
        //         float By = minYs*minYs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a2*t2;
        //         float Cy = minYs*minYs*axx.squaredNorm()-SphereRadius*SphereRadius*a2*a2;

        //         vd[2] = (-By + sqrt(By*By-4*Ay*Cy)) / (2*Ay);
        //         vd[3] = (-By - sqrt(By*By-4*Ay*Cy)) / (2*Ay);

        //         float Az = minZs*minZs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t3*t3;
        //         float Bz = minZs*minZs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a3*t3;
        //         float Cz = minZs*minZs*axx.squaredNorm()-SphereRadius*SphereRadius*a3*a3;

        //         vd[4] = (-Bz + sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
        //         vd[5] = (-Bz - sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);


        //         float maxXs, maxYs, maxZs;
        //         maxXs = bestx - 0.4 * dxs;
        //         A = b*b + c * c;
        //         B = 2*a*maxXs*c;
        //         C =b*b*(maxXs*maxXs-SphereRadius* SphereRadius) + a*a*maxXs*maxXs;
        //         zs1 =  (-B + sqrt(B*B-4*A*C) )/(2*A);
        //         zs2 =  (-B - sqrt(B*B-4*A*C) )/(2*A);
        //         if(abs(zs1-bestz) < abs(zs2-bestz))
        //             maxZs = zs1;
        //         else
        //             maxZs = zs2;
        //         maxYs = -(a*maxXs + c*maxZs)/b;

        //         axx = hostToFrame_R * Vec3f(maxXs, maxYs, maxYs)/SphereRadius;
        //         a1 = axx(0);
        //         a2 = axx(1);
        //         a3 = axx(2);
        //         t1 = hostToFrame_t(0);
        //         t2 = hostToFrame_t(1);
        //         t3 = hostToFrame_t(2);

        //         Ax = maxXs*maxXs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t1*t1;
        //         Bx = maxXs*maxXs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a1*t1;
        //         Cx =maxXs*maxXs*axx.squaredNorm()-SphereRadius*SphereRadius*a1*a1;

        //         vd[6] = (-Bx + sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);
        //         vd[7] = (-Bx - sqrt(Bx*Bx-4*Ax*Cx)) / (2*Ax);

        //         Ay = maxYs*maxYs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t2*t2;
        //         By = maxYs*maxYs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a2*t2;
        //         Cy = maxYs*maxYs*axx.squaredNorm()-SphereRadius*SphereRadius*a2*a2;

        //         vd[8] = (-By + sqrt(By*By-4*Ay*Cy)) / (2*Ay);
        //         vd[9] = (-By - sqrt(By*By-4*Ay*Cy)) / (2*Ay);

        //         Az = maxZs*maxZs*hostToFrame_t.squaredNorm()-SphereRadius*SphereRadius*t3*t3;
        //         Bz = maxZs*maxZs*2*axx.dot(hostToFrame_t) - SphereRadius*SphereRadius*2*a3*t3;
        //         Cz = maxZs*maxZs*axx.squaredNorm()-SphereRadius*SphereRadius*a3*a3;

        //         vd[10] = (-Bz + sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
        //         vd[11] = (-Bz - sqrt(Bz*Bz-4*Az*Cz)) / (2*Az);
        //     }
        //     idepth_max = *max_element(vd.begin(),vd.end());  
        //     idepth_min = *min_element(vd.begin(),vd.end()); 
        //     lastTraceUV = Vec2f(bestU, bestV); 

        //     return Vec2f(0.4, bestEnergy);
        // }
//

        double ImmaturePoint::linearizeResidual(
                shared_ptr<CalibHessian> HCalib, const float outlierTHSlack,
                shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float &Hdd, float &bd,
                float idepth) {

            if (tmpRes->state_state == ResState::OOB) {
                tmpRes->state_NewState = ResState::OOB;
                return tmpRes->state_energy;
            }

            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();
            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);

            // check OOB due to scale angle change.
            float energyLeft = 0;
            const Eigen::Vector3f *dIl = target->dI;
            const Mat33f &PRE_RTll = precalc->PRE_RTll;
            const Vec3f &PRE_tTll = precalc->PRE_tTll;

            Vec2f affLL = precalc->PRE_aff_mode;

            for (int idx = 0; idx < patternNum; idx++) {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                float drescale, u, v, new_idepth;
                float Ku, Kv;
                Vec3f KliP;

                if (!projectPoint(this->feature->uv[0], this->feature->uv[1], idepth, dx, dy, HCalib,
                                  PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
                    tmpRes->state_NewState = ResState::OOB;
                    return tmpRes->state_energy;
                }


                Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

                if (!std::isfinite((float) hitColor[0])) {
                    tmpRes->state_NewState = ResState::OOB;
                    return tmpRes->state_energy;
                }
                float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

                // depth derivatives.
                float dxInterp = hitColor[1] * HCalib->fxl();
                float dyInterp = hitColor[2] * HCalib->fyl();
                float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

                hw *= weights[idx] * weights[idx];

                Hdd += (hw * d_idepth) * d_idepth;
                bd += (hw * residual) * d_idepth;
            }


            if (energyLeft > energyTH * outlierTHSlack) {
                energyLeft = energyTH * outlierTHSlack;
                tmpRes->state_NewState = ResState::OUTLIER;
            } else {
                tmpRes->state_NewState = ResState::IN;
            }

            tmpRes->state_NewEnergy = energyLeft;
            return energyLeft;
        }

        // for mlti-fisheye 计算当前点逆深度的残差, 正规方程(H和b), 残差状态
        double ImmaturePoint::linearizeResidualforMF( const float outlierTHSlack, shared_ptr<ImmaturePointTemporaryResidual> tmpRes,
                    float &Hdd, float &bd, float idepth)
        {
            if (tmpRes->state_state == ResState::OOB) 
            {
                tmpRes->state_NewState = ResState::OOB;
                return tmpRes->state_energy;
            }

            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();
            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);

            // check OOB due to scale angle change.
            float energyLeft = 0;
            const vector<Eigen::Vector3f *> vdIl = target->vdI;
            const Mat33f &PRE_RTll = precalc->PRE_RTll;
            const Vec3f &PRE_tTll = precalc->PRE_tTll;
            int thiscamnum = this->feature->camnum;

            vector<vector<Vec2f>> vvaffLL = precalc->vvPRE_aff_mode;

            for (int idx = 0; idx < patternNum; idx++) 
            {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];

                float drescale, new_idepth;
                Vec3f xc;       // target 相机坐标系坐标
                float Ku, Kv;
                int tocamnum;
                Vec3f KliP;   // host 归一化球面坐标

                if (!projectPointforMF(thiscamnum, this->feature->uv[0], this->feature->uv[1], idepth, dx, dy, UMF,
                                  PRE_RTll, PRE_tTll, drescale, xc, tocamnum, Ku, Kv, KliP, new_idepth)) 
                {
                    tmpRes->state_NewState = ResState::OOB;
                    return tmpRes->state_energy;
                }

                Vec3f hitColor = (getInterpolatedElement33(vdIl[tocamnum], Ku, Kv, UMF->wPR[0]));

                if (!std::isfinite((float) hitColor[0])) {
                    tmpRes->state_NewState = ResState::OOB;
                    return tmpRes->state_energy;
                }
                float residual = hitColor[0] - (vvaffLL[thiscamnum][tocamnum][0] * color[idx] + vvaffLL[thiscamnum][tocamnum][1]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

                // depth derivatives.
                float dxInterp = hitColor[1];
                float dyInterp = hitColor[2];
                float d_idepth = derive_idepthforMF(UMF, PRE_RTll, PRE_tTll, KliP, idepth, xc, dx, dy, dxInterp, dyInterp, drescale, tocamnum, 0);

                hw *= weights[idx] * weights[idx];

                Hdd += (hw * d_idepth) * d_idepth;
                bd += (hw * residual) * d_idepth;
            }

            if (energyLeft > energyTH * outlierTHSlack) {
                energyLeft = energyTH * outlierTHSlack;
                tmpRes->state_NewState = ResState::OUTLIER;
            } else {
                tmpRes->state_NewState = ResState::IN;
            }

            tmpRes->state_NewEnergy = energyLeft;
            return energyLeft;
        }

        float ImmaturePoint::calcResidual(
                shared_ptr<CalibHessian> HCalib, const float outlierTHSlack,
                shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float idepth) {
            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();
            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);
            float energyLeft = 0;
            const Eigen::Vector3f *dIl = target->dI;
            const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
            const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
            Vec2f affLL = precalc->PRE_aff_mode;

            for (int idx = 0; idx < patternNum; idx++) {
                float Ku, Kv;
                if (!projectPoint(this->feature->uv[0] + patternP[idx][0], this->feature->uv[1] + patternP[idx][1],
                                  idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv)) { return 1e10; }

                Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
                if (!std::isfinite((float) hitColor[0])) {
                    return 1e10;
                }

                float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
            }

            if (energyLeft > energyTH * outlierTHSlack) {
                energyLeft = energyTH * outlierTHSlack;
            }
            return energyLeft;
        }

        float ImmaturePoint::getdPixdd(
                shared_ptr<CalibHessian> HCalib,
                shared_ptr<ImmaturePointTemporaryResidual> tmpRes, float idepth) {

            shared_ptr<FrameHessian> host = feature->host.lock()->frameHessian;
            shared_ptr<FrameHessian> target = tmpRes->target.lock();

            FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]);
            const Vec3f &PRE_tTll = precalc->PRE_tTll;
            float drescale, u = 0, v = 0, new_idepth;
            float Ku, Kv;
            Vec3f KliP;

            projectPoint(this->feature->uv[0], this->feature->uv[1], idepth, 0, 0, HCalib,
                         precalc->PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

            float dxdd = (PRE_tTll[0] - PRE_tTll[2] * u) * HCalib->fxl();
            float dydd = (PRE_tTll[1] - PRE_tTll[2] * v) * HCalib->fyl();
            return drescale * sqrtf(dxdd * dxdd + dydd * dydd);
        }

    }
}