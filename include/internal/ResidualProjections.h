#pragma once
#ifndef LDSO_RESIDUAL_PROJECTIONS_H_
#define LDSO_RESIDUAL_PROJECTIONS_H_

#include "NumTypes.h"
#include "internal/GlobalCalib.h"
#include "frontend/Undistort.h"

namespace ldso {

    namespace internal {

        EIGEN_STRONG_INLINE float derive_idepth(
                const Vec3f &t, const float &u, const float &v,
                const int &dx, const int &dy, const float &dxInterp,
                const float &dyInterp, const float &drescale) {
            return (dxInterp * drescale * (t[0] - t[2] * u)
                    + dyInterp * drescale * (t[1] - t[2] * v)) * SCALE_IDEPTH;
        }

        EIGEN_STRONG_INLINE float derive_idepthforMF(
                UndistortMultiFisheye * UMF, const Mat33f &R,
                const Vec3f &t, const Vec3f KliP, const float idepth, const Vec3f &Xc,
                const int &dx, const int &dy, const float &dxInterp,
                const float &dyInterp, const float &drescale, int n, int level) 
        {
            Eigen::Matrix2f dIL;
            dIL << 0, -1, 1, 0; // x = -y , y = x

            float mSphereRadius = UMF->GetSphereRadius();
            float xs = Xc(0) * drescale * mSphereRadius;
            float ys = Xc(1) * drescale * mSphereRadius;
            float zs = Xc(2) * drescale * mSphereRadius;

            Eigen::Matrix<float, 2, 3> duXs;

            Eigen::Matrix<float, 3, 1> RaySphere;
            RaySphere(0, 0) = xs;
            RaySphere(1, 0) = ys;
            RaySphere(2, 0) = zs;

            Eigen::Matrix<float, 3, 3> RotationMat = (UMF->mvExParas[n].block<3, 3>(0, 0)).cast<float>();
            Eigen::Matrix<float, 3, 1> CamPos = UMF->mvExParas[n].block<3, 1>(0, 3).cast<float>();
            Eigen::Matrix<float, 3, 1> RayCoor = RaySphere - CamPos;
            Eigen::Matrix<float, 3, 1> RectifiedImgPtCoor = RotationMat.inverse() * RayCoor;
            float xsn = RectifiedImgPtCoor(0, 0);
            float ysn = RectifiedImgPtCoor(1, 0);
            float zsn = RectifiedImgPtCoor(2, 0);

            float f = UMF->GetRectifiedImgFocalLength(n, level);

            duXs << f / zsn, 0, - xsn * f / (zsn * zsn),
                    0, f / zsn, - ysn * f / (zsn * zsn);
            
            Eigen::Matrix<float, 2, 3> dSR;
            dSR = dIL * duXs * RotationMat.inverse();

            float xc = Xc(0);
            float yc = Xc(1);
            float zc = Xc(2);
            float l2 = xc * xc + yc * yc + zc * zc;
            float l23 = l2 * sqrt(l2);

            Eigen::Matrix<float, 3, 3> dXsdXc;
            dXsdXc << mSphereRadius  *(yc * yc + zc * zc) / l23, -xc * yc * mSphereRadius / l23, -xc * zc * mSphereRadius / l23,
                    -xc * yc * mSphereRadius / l23, mSphereRadius * (xc * xc + zc * zc) / l23, -yc * zc * mSphereRadius / l23,
                    -xc * zc * mSphereRadius / l23, -yc * zc * mSphereRadius / l23, mSphereRadius * (xc * xc + yc * yc) / l23;
            
            Eigen::Matrix<float, 3, 1> dXcdroh1;
            dXcdroh1 = - R * Vec3f(KliP(0), KliP(1), KliP(2)) / (idepth * idepth);
            //dXcdroh1 = t;
            
            Eigen::Matrix<float, 3, 1> dXsdrho1;
            dXsdrho1 = dXsdXc * dXcdroh1;

             return Eigen::Matrix<float, 1, 2>(dxInterp, dyInterp) * dSR * dXsdrho1;
        }


        // projection equation:
        // K[u,v]^T = 1/d ( K R K^{-1} [u_p, v_p, 1]^T + K t )
        // compute the Ku, Kv in z=1 plane.
        EIGEN_STRONG_INLINE bool projectPoint(
                const float &u_pt, const float &v_pt,
                const float &idepth,
                const Mat33f &KRKi, const Vec3f &Kt,
                float &Ku, float &Kv) {
            Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth;
            Ku = ptp[0] / ptp[2];
            Kv = ptp[1] / ptp[2];
            return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
        }

        EIGEN_STRONG_INLINE bool projectPointforMF(int camnum, UndistortMultiFisheye* UMF,
                const float &u_pt, const float &v_pt,
                const float &idepth,
                const Mat33f &R, const Vec3f &t,
                float &Ku, float &Kv, int& tocamnum) 
        {
            float SphereRadius = UMF->GetSphereRadius();
            double xs1, ys1, zs1;
            UMF->LadybugProjectRectifyPtToSphere(camnum, u_pt, v_pt, &xs1, &ys1, &zs1, 0);
            Vec3f KliP = Vec3f(xs1/SphereRadius, ys1/SphereRadius, zs1/SphereRadius);
            // Vec3f ptp = R * KliP + t * idepth;    // R21*K^-1*x1 + d1 * t21
            Vec3f ptp = R * KliP/idepth + t ;    // R21*K^-1*x1 + d1 * t21
            float drescale = 1.0f / ptp.norm();  

            double xs2 = ptp[0] * drescale * SphereRadius;    // 归一化球面 * SphereRadius  全景球上
            double ys2 = ptp[1] * drescale * SphereRadius; 
            double zs2 = ptp[2] * drescale * SphereRadius;
            int tc;
            double u,v;

            UMF->LadybugReprojectSpherePtToRectify(xs2, ys2, zs2, &tc, &u, &v, 0);
            Ku = u ;
            Kv = v ;
            tocamnum = tc;
            return Ku > 1.1f && Kv > 1.1f && Ku < UMF->wM3G && Kv < UMF->hM3G;
        }
        

        // equation:
        // K[u,v]^T = 1/d ( K R K^{-1} [u_p, v_p, 1]^T + K t ) * rescale
        //                < ------------ ptp --------------- >
        /**
         * Project one point from host to target
         * @param u_pt pixel coordinate in host
         * @param v_pt pixel coordinate in host
         * @param idepth idepth in host
         * @param dx bias in the designed pattern
         * @param dy bias in the designed pattern
         * @param HCalib calib matrix
         * @param R R_TW
         * @param t t_TW
         * @param [out] drescale
         * @param [out] u
         * @param [out] v
         * @param [out] Ku  pixel coordinate in target
         * @param [out] Kv  pixel coordinate in target
         * @param [out] KliP
         * @param [out] new_idepth
         * @return
         */
        EIGEN_STRONG_INLINE bool projectPoint(
                const float &u_pt, const float &v_pt,
                const float &idepth,
                const int &dx, const int &dy,
                shared_ptr<CalibHessian> const &HCalib,
                const Mat33f &R, const Vec3f &t,
                float &drescale, float &u, float &v,
                float &Ku, float &Kv, Vec3f &KliP, float &new_idepth) {
            KliP = Vec3f(
                    (u_pt + dx - HCalib->cxl()) * HCalib->fxli(),
                    (v_pt + dy - HCalib->cyl()) * HCalib->fyli(),
                    1);    // K^-1*x1   归一化平面

            Vec3f ptp = R * KliP + t * idepth;    // R21*K^-1*x1 + d1 * t21
            drescale = 1.0f / ptp[2];         // d2 * d1^-1
            new_idepth = idepth * drescale;   // d2

            if (!(drescale > 0)) {
                return false;
            }

            u = ptp[0] * drescale;    //  归一化平面坐标
            v = ptp[1] * drescale;
            Ku = u * HCalib->fxl() + HCalib->cxl();
            Kv = v * HCalib->fyl() + HCalib->cyl();

            return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
        }

        //@ 将host帧投影到新的帧, 且可以设置像素偏移dxdy, 得到:
        //@ 参数: [drescale 新比旧逆深度] [uv 新的归一化平面]
        //@		[kukv 新的像素平面] [KliP 旧归一化球面] [new_idepth 新的逆深度]
        EIGEN_STRONG_INLINE bool projectPointforMF(int camnum,
                const float &u_pt, const float &v_pt,
                const float &idepth,
                const int &dx, const int &dy,
                UndistortMultiFisheye * UMF,
                const Mat33f &R, const Vec3f &t,
                float &drescale, Vec3f &xc, int &tocamnum,  
                float &Ku, float &Kv, Vec3f &KliP, float &new_idepth) 
        {
            float SphereRadius = UMF->GetSphereRadius();
            double xs1, ys1, zs1;
            UMF->LadybugProjectRectifyPtToSphere(camnum, u_pt + dx, v_pt + dy, &xs1, &ys1, &zs1, 0);
            KliP = Vec3f(xs1/SphereRadius, ys1/SphereRadius, zs1/SphereRadius);
            // Vec3f ptp = R * KliP + t * idepth;    // R21*K^-1*x1 + d1 * t21
            Vec3f ptp = R * KliP/idepth + t ;    

            xc = ptp;

            drescale = 1.0f / ptp.norm();         // d2 * d1^-1    // target帧逆深度 比 host帧逆深度  球面
            // new_idepth = idepth * drescale;   // d2            // 新的帧上逆深度
            new_idepth = drescale;   // d2            // 新的帧上逆深度

            if (!(drescale > 0)) {
                return false;
            }

            double xs2 = ptp[0] * drescale * SphereRadius;    // 归一化球面 * SphereRadius
            double ys2 = ptp[1] * drescale * SphereRadius; 
            double zs2 = ptp[2] * drescale * SphereRadius;
            double u,v;
            UMF->LadybugReprojectSpherePtToRectify(xs2, ys2, zs2, &tocamnum, &u, &v, 0);
            Ku = u ;
            Kv = v ;

            return Ku > 1.1f && Kv > 1.1f && Ku < UMF->wM3G && Kv < UMF->hM3G;
        }

    }

}

#endif // LDSO_RESIDUAL_PROJECTIONS_H_
