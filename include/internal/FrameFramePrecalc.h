#pragma once
#ifndef LDSO_FRAME_FRAME_PRECALC_H_
#define LDSO_FRAME_FRAME_PRECALC_H_

#include "NumTypes.h"
#include "internal/FrameHessian.h"
#include "internal/CalibHessian.h"
#include "frontend/Undistort.h"

#include <memory>

using namespace std;

namespace ldso {

    namespace internal {

        /**
         * in the inverse depth parameterized bundle adjustment, an observation is related with two frames: the host and the target
         * but we just need to compute once to each two frame pairs, not each observation
         * this structure is used for this precalculation
         */
        // 其中带0的是FEJ用的初始状态, 不带0的是更新的状态
        struct FrameFramePrecalc {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            void Set(shared_ptr<FrameHessian> host, shared_ptr<FrameHessian> target, shared_ptr<CalibHessian> HCalib);
            void SetforMF(shared_ptr<FrameHessian> host, shared_ptr<FrameHessian> target, UndistortMultiFisheye * UMF);

            weak_ptr<FrameHessian> host; // defines row
            weak_ptr<FrameHessian> target;   // defines column

            // precalc values

            // 优化过程中部分状态使用FEJ，所以需要保存不同状态的值
            // 其中位姿 / 光度参数使用固定线性化点，逆深度 / 内参 / 图像导数都没有固定线性化点
            // 逆深度、内参、位姿这些几何参数使用中心点的值代替pattern内的
            // 逆深度每次都会重新设置线性化点，相当于没有固定
            // T_TW * T_WH = T_TH
            // T=target, H=Host
            Mat33f PRE_RTll = Mat33f::Zero();     // 优化更新状态后的状态增量。
            Mat33f PRE_RTll_0 = Mat33f::Zero();   //固定的线性化点位置的状态增量
            Vec3f PRE_tTll = Vec3f(0, 0, 0);
            Vec3f PRE_tTll_0 = Vec3f(0, 0, 0);
            Mat33f PRE_KRKiTll = Mat33f::Zero();
            Mat33f PRE_RKiTll = Mat33f::Zero();
            Vec2f PRE_aff_mode = Vec2f(0, 0);
            float PRE_b0_mode = 0;
            Vec3f PRE_KtTll = Vec3f(0, 0, 0);
            float distanceLL = 0;

            // for multi-fisheye
            vector<vector<Vec2f>> vvPRE_aff_mode;  // host-target camnum*camnum*2
            vector<float> vPRE_b0_mode;    // host b camnum个
        };
    }
}

#endif // LDSO_FRAME_FRAME_PRECALC_H_
