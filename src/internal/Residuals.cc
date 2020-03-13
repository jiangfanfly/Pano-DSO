#include "internal/Residuals.h"
#include "internal/FrameHessian.h"
#include "internal/PointHessian.h"
#include "internal/ResidualProjections.h"
#include "internal/GlobalFuncs.h"
#include "Settings.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

namespace ldso {

    namespace internal {

        double PointFrameResidual::linearize(shared_ptr<CalibHessian> &HCalib) {

            // compute jacobians
            state_NewEnergyWithOutlier = -1;
            if (state_state == ResState::OOB) // 当前状态已经位于外边
            {
                state_NewState = ResState::OOB;
                return state_energy;
            }

            shared_ptr<FrameHessian> f = host.lock();
            shared_ptr<FrameHessian> ftarget = target.lock();
            shared_ptr<PointHessian> fPoint = point.lock();
            FrameFramePrecalc *precalc = &(f->targetPrecalc[ftarget->idx]);

            float energyLeft = 0;
            const Eigen::Vector3f *dIl = ftarget->dI;
            const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
            const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
            const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
            const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
            const float *const color = fPoint->color;
            const float *const weights = fPoint->weights;

            Vec2f affLL = precalc->PRE_aff_mode;
            float b0 = precalc->PRE_b0_mode;

            // 李代数到xy的导数
            Vec6f d_xi_x, d_xi_y;

            // Calib到xy的导数
            Vec4f d_C_x, d_C_y;

            // xy 到 idepth 的导数
            float d_d_x, d_d_y;

            {
                float drescale, u, v, new_idepth;  // data in target
                // NOTE u = X/Z, v=Y/Z in target
                float Ku, Kv;
                Vec3f KliP;

                // 重投影
                shared_ptr<PointHessian> p = point.lock();
                if (!projectPoint(p->u, p->v, p->idepth_zero_scaled, 0, 0, HCalib,
                                  PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }

                centerProjectedTo = Vec3f(Ku, Kv, new_idepth);  // 像素坐标 逆深度在此帧中的

                // 各种导数
                // diff d_idepth   像素坐标Ku，kv对逆深度导数 d1
                d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
                d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

                // diff calib  对相机内参导数
                d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
                d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
                d_C_x[0] = KliP[0] * d_C_x[2];
                d_C_x[1] = KliP[1] * d_C_x[3];

                d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
                d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
                d_C_y[0] = KliP[0] * d_C_y[2];
                d_C_y[1] = KliP[1] * d_C_y[3];

                d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
                d_C_x[1] *= SCALE_F;
                d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
                d_C_x[3] *= SCALE_C;

                d_C_y[0] *= SCALE_F;
                d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
                d_C_y[2] *= SCALE_C;
                d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

                // xy到李代数的导数，形式见十四讲
                d_xi_x[0] = new_idepth * HCalib->fxl();
                d_xi_x[1] = 0;
                d_xi_x[2] = -new_idepth * u * HCalib->fxl();
                d_xi_x[3] = -u * v * HCalib->fxl();
                d_xi_x[4] = (1 + u * u) * HCalib->fxl();
                d_xi_x[5] = -v * HCalib->fxl();

                d_xi_y[0] = 0;
                d_xi_y[1] = new_idepth * HCalib->fyl();
                d_xi_y[2] = -new_idepth * v * HCalib->fyl();
                d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
                d_xi_y[4] = u * v * HCalib->fyl();
                d_xi_y[5] = u * HCalib->fyl();
            }


            {
                J->Jpdxi[0] = d_xi_x;
                J->Jpdxi[1] = d_xi_y;

                J->Jpdc[0] = d_C_x;
                J->Jpdc[1] = d_C_y;

                J->Jpdd[0] = d_d_x;
                J->Jpdd[1] = d_d_y;

            }

            float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
            float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
            float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

            float wJI2_sum = 0;

            for (int idx = 0; idx < patternNum; idx++) {
                float Ku, Kv;
                shared_ptr<PointHessian> p = point.lock();
                if (!projectPoint(p->u + patternP[idx][0], p->v + patternP[idx][1], p->idepth_scaled,
                                  PRE_KRKiTll, PRE_KtTll, Ku, Kv)) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }

                projectedTo[idx][0] = Ku;
                projectedTo[idx][1] = Kv;

                Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
                float residual = hitColor[0] - (float) (affLL[0] * color[idx] + affLL[1]);

                float drdA = (color[idx] - b0);
                if (!std::isfinite((float) hitColor[0])) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }


                float w = sqrtf(setting_outlierTHSumComponent /
                                (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
                w = 0.5f * (w + weights[idx]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += w * w * hw * residual * residual * (2 - hw);

                {
                    if (hw < 1) hw = sqrtf(hw);
                    hw = hw * w;

                    hitColor[1] *= hw;
                    hitColor[2] *= hw;

                    J->resF[idx] = residual * hw;

                    J->JIdx[0][idx] = hitColor[1];
                    J->JIdx[1][idx] = hitColor[2];
                    J->JabF[0][idx] = drdA * hw;
                    J->JabF[1][idx] = hw;

                    JIdxJIdx_00 += hitColor[1] * hitColor[1];
                    JIdxJIdx_11 += hitColor[2] * hitColor[2];
                    JIdxJIdx_10 += hitColor[1] * hitColor[2];

                    JabJIdx_00 += drdA * hw * hitColor[1];
                    JabJIdx_01 += drdA * hw * hitColor[2];
                    JabJIdx_10 += hw * hitColor[1];
                    JabJIdx_11 += hw * hitColor[2];

                    JabJab_00 += drdA * drdA * hw * hw;
                    JabJab_01 += drdA * hw * hw;
                    JabJab_11 += hw * hw;

                    wJI2_sum += hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

                    if (setting_affineOptModeA < 0) J->JabF[0][idx] = 0;
                    if (setting_affineOptModeB < 0) J->JabF[1][idx] = 0;

                }
            }

            J->JIdx2(0, 0) = JIdxJIdx_00;
            J->JIdx2(0, 1) = JIdxJIdx_10;
            J->JIdx2(1, 0) = JIdxJIdx_10;
            J->JIdx2(1, 1) = JIdxJIdx_11;
            J->JabJIdx(0, 0) = JabJIdx_00;
            J->JabJIdx(0, 1) = JabJIdx_01;
            J->JabJIdx(1, 0) = JabJIdx_10;
            J->JabJIdx(1, 1) = JabJIdx_11;
            J->Jab2(0, 0) = JabJab_00;
            J->Jab2(0, 1) = JabJab_01;
            J->Jab2(1, 0) = JabJab_01;
            J->Jab2(1, 1) = JabJab_11;

            state_NewEnergyWithOutlier = energyLeft;

            if (energyLeft > std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH) || wJI2_sum < 2) {
                energyLeft = std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH);
                state_NewState = ResState::OUTLIER;
            } else {
                state_NewState = ResState::IN;
            }

            state_NewEnergy = energyLeft;
            return energyLeft;
        }

        void PointFrameResidual::fixLinearizationF(shared_ptr<EnergyFunctional> ef) {

            Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

            // compute Jp*delta
            __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
                                            + J->Jpdc[0].dot(ef->cDeltaF)
                                            + J->Jpdd[0] * point.lock()->deltaF);
            __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
                                            + J->Jpdc[1].dot(ef->cDeltaF)
                                            + J->Jpdd[1] * point.lock()->deltaF);

            __m128 delta_a = _mm_set1_ps((float) (dp[6]));
            __m128 delta_b = _mm_set1_ps((float) (dp[7]));

            for (int i = 0; i < patternNum; i += 4) {
                // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
                __m128 rtz = _mm_load_ps(((float *) &J->resF) + i);
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx)) + i), Jp_delta_x));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx + 1)) + i), Jp_delta_y));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF)) + i), delta_a));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 1)) + i), delta_b));
                _mm_store_ps(((float *) &res_toZeroF) + i, rtz);
            }

            isLinearized = true;
        }

        void PointFrameResidual::fixLinearizationFforMF(shared_ptr<EnergyFunctional> ef) 
        {

            Vec56f dp = ef->adHTdeltaFforMF[hostIDX + ef->nFrames * targetIDX];

            // compute Jp*delta
            __m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
                                            + J->Jpdd[0] * point.lock()->deltaF);
            __m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
                                            + J->Jpdd[1] * point.lock()->deltaF);

            int abidx = 6 + 2 * (hcamnum * camNums + tcamnum);
            __m128 delta_a = _mm_set1_ps((float) (dp[abidx]));
            __m128 delta_b = _mm_set1_ps((float) (dp[abidx + 1]));

            for (int i = 0; i < patternNum; i += 4) 
            {
                // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
                __m128 rtz = _mm_load_ps(((float *) &J->resF) + i);
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx)) + i), Jp_delta_x));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JIdx + 1)) + i), Jp_delta_y));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF)) + i), delta_a));
                rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (J->JabF + 1)) + i), delta_b));
                _mm_store_ps(((float *) &res_toZeroF) + i, rtz);
            }

            isLinearized = true;
        }

        double PointFrameResidual::linearizeforMF(UndistortMultiFisheye* UMF)
        {
            // compute jacobians
            state_NewEnergyWithOutlier = -1;
            if (state_state == ResState::OOB) // 当前状态已经位于外边
            {
                state_NewState = ResState::OOB;
                return state_energy;
            }

            shared_ptr<FrameHessian> f = host.lock();
            shared_ptr<FrameHessian> ftarget = target.lock();
            shared_ptr<PointHessian> fPoint = point.lock();
            FrameFramePrecalc *precalc = &(f->targetPrecalc[ftarget->idx]);     // 伴随 相对量恢复绝对量

            float energyLeft = 0;
            const vector<Eigen::Vector3f *> vdIl = ftarget->vdI;
            const Mat33f &PRE_RTll = precalc->PRE_RTll;
            const Vec3f &PRE_tTll = precalc->PRE_tTll;
            const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
            const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
            const float *const color = fPoint->color;
            const float *const weights = fPoint->weights;

            vector<vector<Vec2f>> vvaffLL = precalc->vvPRE_aff_mode;    // 待优化的a和b, 就是host和target合的
            vector<float> vb0 = precalc->vPRE_b0_mode;

            hcamnum = point.lock()->camnum;
            camNums = UMF->camNums;

            //! x=0时候求几何的导数, 使用FEJ!! ,逆深度没有使用FEJ
            // 李代数到xy的导数
            Vec6f d_xi_x, d_xi_y;

            // // Calib到xy的导数
            // Vec4f d_C_x, d_C_y;

            // xy 到 idepth 的导数
            float d_d_x, d_d_y;

            {
                float drescale, new_idepth;  // data in target
                // NOTE u = X/Z, v=Y/Z in target
                float Ku, Kv;
                Vec3f xc;
                int tocamnum;
                Vec3f KliP;

                shared_ptr<PointHessian> p = point.lock();
                if (!projectPointforMF(hcamnum, p->u, p->v, p->idepth_zero_scaled, 0, 0, UMF,
                                  PRE_RTll_0, PRE_tTll_0, drescale, xc, tocamnum, Ku, Kv, KliP, new_idepth)) 
                {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }

                // centerProjectedTo = xc * new_idepth;  // 像素坐标 逆深度在此帧中的
                centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

                // 各种导数
                Eigen::Matrix<float, 2, 6> dxrdpose;
                Eigen::Matrix<float, 2, 1> dxrdrho1;
                computedxrdposedrho1(UMF, PRE_RTll_0, PRE_tTll_0, xc, KliP, p->idepth_zero_scaled, 0, 0, drescale, tocamnum, dxrdrho1, dxrdpose, 0);

                // diff d_idepth   像素坐标Ku，kv对逆深度导数 d1
                d_d_x = dxrdrho1(0);
                d_d_y = dxrdrho1(1);

                // xy到李代数的导数，形式见ppt
                d_xi_x[0] = dxrdpose(0,0);
                d_xi_x[1] = dxrdpose(0,1);
                d_xi_x[2] = dxrdpose(0,2);
                d_xi_x[3] = dxrdpose(0,3);
                d_xi_x[4] = dxrdpose(0,4);
                d_xi_x[5] = dxrdpose(0,5);

                d_xi_y[0] = dxrdpose(1,0);
                d_xi_y[1] = dxrdpose(1,1);
                d_xi_y[2] = dxrdpose(1,2);
                d_xi_y[3] = dxrdpose(1,3);
                d_xi_y[4] = dxrdpose(1,4);
                d_xi_y[5] = dxrdpose(1,5);

// test Jacobian
                // Vec3f hitColor = getInterpolatedElement33(vdIl[tocamnum], Ku, Kv, UMF->wPR[0]);    // 在当前第二帧 中的 灰度值
                // float dxInterp = hitColor[1];    
                // float dyInterp = hitColor[2];

                // vector<float> dpp;
                // dpp.resize(7);
                // Eigen::Matrix<float, 1, 6> dIdpose;
                // Eigen::Matrix<float, 1, 1> dIdrho1;
                // dIdpose = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdpose;    // dI/dpose = dI/dxr * dxr/dXs * dXs/dpose
                // dIdrho1 = Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dxrdrho1;    // dI/drho1 = dI/dxr * dxr/dXs * dXs/drho1

                //dpp = NumericDiff(hcamnum, p->u, p->v, 0, PRE_RTll_0, PRE_tTll_0, p->idepth_zero_scaled, Vec2f(vvaffLL[hcamnum][tocamnum][0], vvaffLL[hcamnum][tocamnum][1]));

                // cout << "dIdpose:      " << dIdpose(0) << " " << dIdpose(1) << " " << dIdpose(2) << " " << dIdpose(3) << " " << dIdpose(4) << " " << dIdpose(5) << " "<< dIdrho1(0) << " " <<endl;
                // cout << "dIdposetest: " << dpp[0] << " " << dpp[1] << " " << dpp[2] << " " << dpp[3] << " " << dpp[4] << " " << dpp[5] << " " << dpp[6] << " " <<endl;
                // int tt = 0;
                float idepthnew1 = p->idepth_zero_scaled;
                float idepthnew2 = p->idepth_zero_scaled;
                float da =1e-6; 
                idepthnew1 = idepthnew1 + da;
                idepthnew2 = idepthnew2 - da;
                Vec5f r1, r2;
                r1 = testJaccobian(hcamnum, p->u, p->v, 0, PRE_RTll_0, PRE_tTll_0, idepthnew1, Vec2f(vvaffLL[hcamnum][tocamnum][0], vvaffLL[hcamnum][tocamnum][1]));
                r2 = testJaccobian(hcamnum, p->u, p->v, 0, PRE_RTll_0, PRE_tTll_0, idepthnew2, Vec2f(vvaffLL[hcamnum][tocamnum][0], vvaffLL[hcamnum][tocamnum][1]));
                //float di 
                // dp[6] = (r1(2)- r2(2))/(2*da);
                d_d_x = (r1(0)- r2(0))/(2*da); 
                d_d_y = (r1(1)- r2(1))/(2*da);
//test end 

            }


            {
                J->Jpdxi[0] = d_xi_x;
                J->Jpdxi[1] = d_xi_y;

                // J->Jpdc[0] = d_C_x;
                // J->Jpdc[1] = d_C_y;

                J->Jpdd[0] = d_d_x;
                J->Jpdd[1] = d_d_y;

            }

            float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
            float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
            float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

            float wJI2_sum = 0;

            int tocamnum;
            for (int idx = 0; idx < patternNum; idx++)
            {
                float Ku, Kv;
                shared_ptr<PointHessian> p = point.lock();
                if (!projectPointforMF(hcamnum, UMF,p->u + patternP[idx][0], p->v + patternP[idx][1], p->idepth_scaled,
                                  PRE_RTll, PRE_tTll, Ku, Kv, tocamnum)) 
                {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }
                // if(tcamnum != tocamnum) //  边缘处
                // {
                //     state_NewState = ResState::OOB;
                //     return state_energy;
                // }
                tcamnum = tocamnum;
                //cout << "tcamnum :" << tcamnum << endl;

                projectedTo[idx][0] = Ku;
                projectedTo[idx][1] = Kv;

                Vec3f hitColor = (getInterpolatedElement33(vdIl[tcamnum], Ku, Kv, UMF->wPR[0]));
                float residual = hitColor[0] - (float) (vvaffLL[hcamnum][tcamnum][0] * color[idx] + vvaffLL[hcamnum][tcamnum][1]);

                //* 残差对光度仿射a求导
		        //! 光度参数使用固定线性化点了
                float drdA = (color[idx] - vb0[hcamnum]);
                if (!std::isfinite((float) hitColor[0])) {
                    state_NewState = ResState::OOB;
                    return state_energy;
                }

                //* 和梯度大小成比例的权重
                float w = sqrtf(setting_outlierTHSumComponent /
                                (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
                w = 0.5f * (w + weights[idx]);

                float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
                energyLeft += w * w * hw * residual * residual * (2 - hw);

                {
                    if (hw < 1) hw = sqrtf(hw);
                    hw = hw * w;

                    hitColor[1] *= hw;
                    hitColor[2] *= hw;

                    // 残差 res*w*sqrt(hw)
                    J->resF[idx] = residual * hw;
                    
                    // 图像导数 dx dy
                    J->JIdx[0][idx] = hitColor[1];
                    J->JIdx[1][idx] = hitColor[2];
                    //! 对光度合成后a b的导数 [Ii-b0  1]
			        //! Ij - a*Ii - b  (a = tj*e^aj / ti*e^ai,   b = bj - a*bi) 
			        //bug 正负号有影响 ???
                    J->JabF[0][idx] = drdA * hw;
                    J->JabF[1][idx] = hw;

                    // dIdx&dIdx hessian block
                    JIdxJIdx_00 += hitColor[1] * hitColor[1];
                    JIdxJIdx_11 += hitColor[2] * hitColor[2];
                    JIdxJIdx_10 += hitColor[1] * hitColor[2];

                    // dIdx&dIdab hessian block
                    JabJIdx_00 += drdA * hw * hitColor[1];
                    JabJIdx_01 += drdA * hw * hitColor[2];
                    JabJIdx_10 += hw * hitColor[1];
                    JabJIdx_11 += hw * hitColor[2];

                    // dIdab&dIdab hessian block
                    JabJab_00 += drdA * drdA * hw * hw;
                    JabJab_01 += drdA * hw * hw;
                    JabJab_11 += hw * hw;

                    wJI2_sum += hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]);

                    if (setting_affineOptModeA < 0) J->JabF[0][idx] = 0;
                    if (setting_affineOptModeB < 0) J->JabF[1][idx] = 0;

                }
            }

            J->JIdx2(0, 0) = JIdxJIdx_00;
            J->JIdx2(0, 1) = JIdxJIdx_10;
            J->JIdx2(1, 0) = JIdxJIdx_10;
            J->JIdx2(1, 1) = JIdxJIdx_11;
            J->JabJIdx(0, 0) = JabJIdx_00;
            J->JabJIdx(0, 1) = JabJIdx_01;
            J->JabJIdx(1, 0) = JabJIdx_10;
            J->JabJIdx(1, 1) = JabJIdx_11;
            J->Jab2(0, 0) = JabJab_00;
            J->Jab2(0, 1) = JabJab_01;
            J->Jab2(1, 0) = JabJab_01;
            J->Jab2(1, 1) = JabJab_11;

            state_NewEnergyWithOutlier = energyLeft;

            if (energyLeft > std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH) || wJI2_sum < 2) {
                energyLeft = std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH);
                state_NewState = ResState::OUTLIER;
            } 
            else
            {
                state_NewState = ResState::IN;
            }

            state_NewEnergy = energyLeft;
            return energyLeft;

        }

        
   // for multi-fisheye Diff test drdT drdrho
    vector<float> PointFrameResidual::NumericDiff(int n, double x, double y,int lvl, Mat33f R, Vec3f t, float idepth, Vec2f ab)
    {
// 中值求导
        vector<float> dp;
        dp.resize(7);
        float da =1e-6; 
        Vec5f r1, r2;
        Vec3f tnew11(t),tnew12(t);
        tnew11(0) = t(0) + da;
        tnew12(0) = t(0) - da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew11,idepth, ab);
        r2 = testJaccobian(n,x, y, lvl, R, tnew12,idepth, ab);
        //float dt0 
        // dp[0] = (r1(2)- r2(2))/(2*da);
        dp[0] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        Vec3f tnew21(t),tnew22(t);
        tnew21(1) = t(1) + da;
        tnew22(1) = t(1) - da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew21,idepth, ab);
        r2 = testJaccobian(n,x, y, lvl, R, tnew22,idepth, ab);
        //float dt1 
        // dp[1] = (r1(2)- r2(2))/(2*da);
        dp[1] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        Vec3f tnew31(t),tnew32(t);
        tnew31(2) = t(2) + da;
        tnew32(2) = t(2) - da;
        r1 = testJaccobian(n,x, y, lvl, R, tnew31,idepth, ab);
        r2 = testJaccobian(n,x, y, lvl, R, tnew32,idepth, ab);
        //float dt2 
        // dp[2] = (r1(2)- r2(2))/(2*da);
        dp[2] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        SO3 Rnew11(R.cast<double>()),Rnew12(R.cast<double>());
        Eigen::Vector3d so311= Rnew11.log();
        Eigen::Vector3d so312= Rnew12.log();
        so311(0) = so311(0) + da;
        so312(0) = so312(0) - da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so311).matrix().cast<float>(), t,idepth, ab);
        r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so312).matrix().cast<float>(), t,idepth, ab);
        //float dr1 
        // dp[3] = (r1(2)- r2(2))/(2*da);
        dp[3] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        SO3 Rnew21(R.cast<double>()),Rnew22(R.cast<double>());
        Eigen::Vector3d so321= Rnew21.log();
        Eigen::Vector3d so322= Rnew22.log();
        so321(1) = so321(1) + da;
        so322(1) = so322(1) - da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so321).matrix().cast<float>(), t,idepth, ab);
        r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so322).matrix().cast<float>(), t,idepth, ab);
        //float dr2 
        // dp[4] = (r1(2)- r2(2))/(2*da);
        dp[4] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        SO3 Rnew31(R.cast<double>()),Rnew32(R.cast<double>());
        Eigen::Vector3d so331= Rnew31.log();
        Eigen::Vector3d so332= Rnew32.log();
        so331(2) = so331(2) + da;
        so332(2) = so332(2) - da;
        r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so331).matrix().cast<float>(), t,idepth, ab);
        r2 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so332).matrix().cast<float>(), t,idepth, ab);
        //float dr3 
        // dp[5] = (r1(2)- r2(2))/(2*da);
        dp[5] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

        float idepthnew1=idepth;
        float idepthnew2=idepth;
        idepthnew1 = idepthnew1 + da;
        idepthnew2 = idepthnew2 - da;
        r1 = testJaccobian(n,x, y, lvl, R, t,idepthnew1, ab);
        r2 = testJaccobian(n,x, y, lvl, R, t,idepthnew2, ab);
        //float di 
        // dp[6] = (r1(2)- r2(2))/(2*da);
        dp[6] = r1(3)* (r1(0)- r2(0))/(2*da) + r1(4)* (r1(1)- r2(1))/(2*da);

// // 前向求导
        // Vec5f r = testJaccobian(n,x, y, lvl, R, t,idepth, ab);
        // vector<float> dp;
        // dp.resize(7);
        // float da =1e-6; 
        // Vec5f r1, r2;
        // Vec3f tnew11(t);
        // tnew11(0) = t(0) + da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew11,idepth, ab);
        // //float dt0 
        // dp[0] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        // Vec3f tnew21(t);
        // tnew21(1) = t(1) + da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew21,idepth, ab);
        // //float dt1 
        // //dp[1] = (r1- r)/da;
        // dp[1] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        // Vec3f tnew31(t);
        // tnew31(2) = t(2) + da;
        // r1 = testJaccobian(n,x, y, lvl, R, tnew31,idepth, ab);
        // //float dt2 
        // // dp[2] = (r1- r)/da;
        // dp[2] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        // SO3 Rnew11(R.cast<double>());
        // Eigen::Vector3d so311= Rnew11.log();
        // so311(0) = so311(0) + da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so311).matrix().cast<float>(), t,idepth, ab);
        // //float dr1 
        // //dp[3] = (r1- r)/da;
        // dp[3] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        // SO3 Rnew21(R.cast<double>());
        // Eigen::Vector3d so321= Rnew21.log();
        // so321(1) = so321(1) + da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so321).matrix().cast<float>(), t,idepth, ab);
        // //float dr2 
        // // dp[4] = (r1- r)/da;
        // dp[4] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;

        // SO3 Rnew31(R.cast<double>());
        // Eigen::Vector3d so331= Rnew31.log();
        // so331(2) = so331(2) + da;
        // r1 = testJaccobian(n,x, y, lvl, Sophus::SO3::exp(so331).matrix().cast<float>(), t,idepth, ab);
        // //float dr3 
        // // dp[5] = (r1- r)/da;
        // dp[5] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;


        // float idepthnew1=idepth;
        // idepthnew1 = idepthnew1 + da;
        // r1 = testJaccobian(n,x, y, lvl, R, t,idepthnew1, ab);
        // //float di 
        // // dp[6] = (r1- r)/da;
        // dp[6] = r1(3)* (r1(0)- r(0))/da + r1(4)* (r1(1)- r(1))/da;
//
        return dp;
    }

    Vec5f PointFrameResidual::testJaccobian(int n, double RectifiedPixalx, double RectifiedPixaly, int lvl, Mat33f R, Vec3f t, float idepth, Vec2f ab)
    {
        bool isGood = true;
        
        shared_ptr<FrameHessian> f = host.lock();
        shared_ptr<FrameHessian> ftarget = target.lock();
        UndistortMultiFisheye* UMF = f->UMF;
        int wl = UMF->wPR[lvl], hl = UMF->hPR[lvl];
        vector<Eigen::Vector3f* > vcolorRef(f->vdIp[lvl]);
        vector<Eigen::Vector3f* > vcolorNew(ftarget->vdIp[lvl]);
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
        // tocamnum = n;
        //LOG(INFO) << "LadybugReprojectSpherePtToFishEyeImg" << endl ;

        //float new_idepth = point->idepth_new / pt[2];   // d1 * d2


        Vec3f hitColor = getInterpolatedElement33(vcolorNew[tocamnum], Ku, Kv, wl);    // 在当前第二帧 中的 灰度值
        // Vec3f hitColor = getInterpolatedElement33BiCub(vcolorNew[tocamnum], Ku, Kv, wl);

        //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
        float rlR = getInterpolatedElement31(vcolorRef[n], RectifiedPixalx, RectifiedPixaly, wl);  // 在 第一帧 中的 灰度值


        float residual = hitColor[0] - ab[0] * rlR - ab[1];  // 计算光度误差 r = I2 - a21*I1 -b21 (I2 = a21*I1 + b21)   n 第一帧像素所在相机号 ， tocamnum 投影到当前帧 像素所在相机号
        float r = residual;
        float dxInterp = hitColor[1];    
        float dyInterp = hitColor[2];    
        float u = Ku;
        float v = Kv;
        Vec5f a;
        a << u, v, r, dxInterp, dyInterp;
        return a;
    }

        /*
        double FeatureObsResidual::linearize(shared_ptr<CalibHessian> &HCalib) {

            // compute jacobians
            state_NewEnergyWithOutlier = -1;
            if (state_state == ResState::OOB) { // 当前状态已经位于外边
                state_NewState = ResState::OOB;
                return state_energy;
            }

            shared_ptr<FrameHessian> f = host.lock();
            shared_ptr<FrameHessian> ftarget = target.lock();
            shared_ptr<PointHessian> fPoint = point.lock();

            FrameFramePrecalc *precalc = &(f->targetPrecalc[ftarget->idx]);

            float energyLeft = 0;
            const Eigen::Vector3f *dIl = ftarget->dI;
            const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
            const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
            const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
            const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
            const float *const color = fPoint->color;
            const float *const weights = fPoint->weights;

            Vec2f affLL = precalc->PRE_aff_mode;
            float b0 = precalc->PRE_b0_mode;

            // 李代数到xy的导数
            Vec6f d_xi_x, d_xi_y;

            // Calib到xy的导数
            Vec4f d_C_x, d_C_y;

            // xy 到 idepth 的导数
            float d_d_x, d_d_y;

            float drescale, u, v, new_idepth;  // data in target
            // NOTE u = X/Z, v=Y/Z in target
            float Ku, Kv;
            Vec3f KliP;

            // 重投影
            shared_ptr<PointHessian> p = point.lock();
            if (!projectPoint(p->u, p->v, p->idepth_zero_scaled, 0, 0, HCalib,
                              PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth)) {
                state_NewState = ResState::OOB;
                return state_energy;
            }

            // 各种导数
            // diff d_idepth
            d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
            d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

            // diff calib
            d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));
            d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();
            d_C_x[0] = KliP[0] * d_C_x[2];
            d_C_x[1] = KliP[1] * d_C_x[3];

            d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();
            d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));
            d_C_y[0] = KliP[0] * d_C_y[2];
            d_C_y[1] = KliP[1] * d_C_y[3];

            d_C_x[0] = (d_C_x[0] + u) * SCALE_F;
            d_C_x[1] *= SCALE_F;
            d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;
            d_C_x[3] *= SCALE_C;

            d_C_y[0] *= SCALE_F;
            d_C_y[1] = (d_C_y[1] + v) * SCALE_F;
            d_C_y[2] *= SCALE_C;
            d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;

            // xy到李代数的导数，形式见十四讲
            d_xi_x[0] = new_idepth * HCalib->fxl();
            d_xi_x[1] = 0;
            d_xi_x[2] = -new_idepth * u * HCalib->fxl();
            d_xi_x[3] = -u * v * HCalib->fxl();
            d_xi_x[4] = (1 + u * u) * HCalib->fxl();
            d_xi_x[5] = -v * HCalib->fxl();

            d_xi_y[0] = 0;
            d_xi_y[1] = new_idepth * HCalib->fyl();
            d_xi_y[2] = -new_idepth * v * HCalib->fyl();
            d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
            d_xi_y[4] = u * v * HCalib->fyl();
            d_xi_y[5] = u * HCalib->fyl();


            J->Jpdxi[0] = d_xi_x;
            J->Jpdxi[1] = d_xi_y;

            J->Jpdc[0] = d_C_x;
            J->Jpdc[1] = d_C_y;

            J->Jpdd[0] = d_d_x;
            J->Jpdd[1] = d_d_y;

            Vec2f residual = Vec2f(Ku, Kv) - obsPixel;
            // LOG(INFO) << "proj: " << Ku << ", " << Kv << ", obs: " << obsPixel[0] << ", " << obsPixel[1] << ", res="
            // << residual.transpose() << endl;
            float residualNorm = residual.squaredNorm();
            float w = 1;

            // huber weight
            float hw = fabsf(residualNorm) < setting_huberTH ? 1 : setting_huberTH / fabsf(residualNorm);
            energyLeft += w * w * hw * residual.dot(residual) * (2 - hw);

            if (hw < 1) hw = sqrtf(hw);
            hw = hw * w;
            J->resF = hw * residual;

            state_NewEnergyWithOutlier = energyLeft;

            if (energyLeft > std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH)) {
                energyLeft = std::max<float>(f->frameEnergyTH, ftarget->frameEnergyTH);
                state_NewState = ResState::OUTLIER;
            } else {
                state_NewState = ResState::IN;
            }

            // LOG(INFO) << "Energy = " << energyLeft << endl;
            state_NewEnergy = energyLeft;
            return energyLeft;
        }

        void FeatureObsResidual::fixLinearizationF(shared_ptr<EnergyFunctional> ef) {

            Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX];

            // compute Jp*delta
            float Jp_delta_x = (J->Jpdxi[0].dot(dp.head<6>())
                                + J->Jpdc[0].dot(ef->cDeltaF)
                                + J->Jpdd[0] * point.lock()->deltaF);
            float Jp_delta_y = (J->Jpdxi[1].dot(dp.head<6>())
                                + J->Jpdc[1].dot(ef->cDeltaF)
                                + J->Jpdd[1] * point.lock()->deltaF);

            res_toZeroF[0] = Jp_delta_x + J->resF[0];
            res_toZeroF[1] = Jp_delta_y + J->resF[1];
            isLinearized = true;
        }
         */

    }

}