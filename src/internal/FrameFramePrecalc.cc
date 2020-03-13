#include "internal/FrameFramePrecalc.h"

namespace ldso {
    namespace internal {

        void FrameFramePrecalc::Set(shared_ptr<FrameHessian> host, shared_ptr<FrameHessian> target,
                                    shared_ptr<CalibHessian> HCalib) {

            this->host = host;
            this->target = target;

            SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
            PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
            PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

            SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
            PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
            PRE_tTll = (leftToLeft.translation()).cast<float>();
            distanceLL = leftToLeft.translation().norm();

            Mat33f K = Mat33f::Zero();
            K(0, 0) = HCalib->fxl();
            K(1, 1) = HCalib->fyl();
            K(0, 2) = HCalib->cxl();
            K(1, 2) = HCalib->cyl();
            K(2, 2) = 1;

            PRE_KRKiTll = K * PRE_RTll * K.inverse();
            PRE_RKiTll = PRE_RTll * K.inverse();
            PRE_KtTll = K * PRE_tTll;

            PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(),
                                                       target->aff_g2l()).cast<float>();
            PRE_b0_mode = host->aff_g2l_0().b;
        }

        // for multi-fisheye 计算优化前和优化后的相对位姿, 相对光度变化, 及中间变量
        void FrameFramePrecalc::SetforMF(shared_ptr<FrameHessian> host, shared_ptr<FrameHessian> target,
                                    UndistortMultiFisheye * UMF) 
        {

            this->host = host;
            this->target = target;
            // 优化前host target间位姿变换
            SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();  // Tth host_to_target
            PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
            PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();
        	// 优化后host到target间位姿变换
            SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
            PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
            PRE_tTll = (leftToLeft.translation()).cast<float>();
            distanceLL = leftToLeft.translation().norm();


            // PRE_KRKiTll = K * PRE_RTll * K.inverse();
            // PRE_RKiTll = PRE_RTll * K.inverse();
            // PRE_KtTll = K * PRE_tTll;
            int camnums = UMF->camNums;
            vvPRE_aff_mode.resize(camnums);
            vPRE_b0_mode.resize(camnums);
            for(int nh = 0; nh < camnums; nh++)
            {
                vvPRE_aff_mode[nh].resize(camnums);
                for(int nt = 0; nt < camnums; nt++)
                {
                    vvPRE_aff_mode[nh][nt] = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2lforMF()[nh],
                                                        target->aff_g2lforMF()[nt]).cast<float>();
                }
                vPRE_b0_mode[nh] = host->aff_g2l_0forMF()[nh].b;
            }

        }
    }
}