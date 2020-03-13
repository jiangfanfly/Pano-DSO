#pragma once
#ifndef LDSO_RAW_RESIDUAL_JACOBIAN_H_
#define LDSO_RAW_RESIDUAL_JACOBIAN_H_

#include "NumTypes.h"

namespace ldso {

    namespace internal {

        // bunch of Jacobians used in optimization

        struct RawResidualJacobian {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            // ================== new structure: save independently =============.
            VecNRf resF;            // residual, 8x1  pattern

            // the two rows of d[x,y]/d[xi].
            Vec6f Jpdxi[2];         // 2x6  //点对位姿

            // the two rows of d[x,y]/d[C].
            VecCf Jpdc[2];          // 2x4  //点对相机参数

            // the two rows of d[x,y]/d[idepth].
            Vec2f Jpdd;             // 2x1  //点对逆深度

            // the two columns of d[r]/d[x,y].  pattern 8
            VecNRf JIdx[2];         // 8x2  //pattern光度误差对点, 8×2

            // = the two columns of d[r] / d[ab]
            VecNRf JabF[2];         // 8x2  //pattern 光度误差对光度仿射， 8x2

            // 对应的小的hessian
            // = JIdx^T * JIdx (inner product). Only as a shorthand.
            Mat22f JIdx2;               // 2x2  
            // = Jab^T * JIdx (inner product). Only as a shorthand.
            Mat22f JabJIdx;         // 2x2
            // = Jab^T * Jab (inner product). Only as a shorthand.
            Mat22f Jab2;            // 2x2
        };
    }
}

#endif // LDSO_RAW_RESIDUAL_JACOBIAN_H_
