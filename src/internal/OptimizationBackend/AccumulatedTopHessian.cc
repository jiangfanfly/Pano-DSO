#include "internal/OptimizationBackend/AccumulatedTopHessian.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

namespace ldso {

    namespace internal {

        template<int mode>
        void AccumulatedTopHessianSSE::addPoint(shared_ptr<PointHessian> p, EnergyFunctional const *const ef,
                                                int tid) { // 0 = active, 1 = linearized, 2=marginalize


            assert(mode == 0 || mode == 1 || mode == 2);

            VecCf dc = ef->cDeltaF;
            float dd = p->deltaF;

            float bd_acc = 0;
            float Hdd_acc = 0;
            VecCf Hcd_acc = VecCf::Zero();

            for (shared_ptr<PointFrameResidual> &r : p->residuals) {
                if (mode == 0) {
                    if (r->isLinearized || !r->isActive())
                        continue;
                }
                if (mode == 1) {
                    if (!r->isLinearized || !r->isActive())
                        continue;
                }
                if (mode == 2) {    // marginalize, must be already linearized
                    if (!r->isActive())
                        continue;
                    assert(r->isLinearized);
                }

                shared_ptr<RawResidualJacobian> rJ = r->J;
                int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
                Mat18f dp = ef->adHTdeltaF[htIDX];

                VecNRf resApprox;  // 残差 8×1的误差向量
                if (mode == 0)   //  active
                    resApprox = rJ->resF;
                if (mode == 2)   //  marginalize
                    resApprox = r->res_toZeroF;
                if (mode == 1) {  // linearized
                    // compute Jp*delta
                    __m128 Jp_delta_x = _mm_set1_ps(
                            rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd);
                    __m128 Jp_delta_y = _mm_set1_ps(
                            rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd);
                    __m128 delta_a = _mm_set1_ps((float) (dp[6]));
                    __m128 delta_b = _mm_set1_ps((float) (dp[7]));

                    for (int i = 0; i < patternNum; i += 4) {
                        // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
                        __m128 rtz = _mm_load_ps(((float *) &r->res_toZeroF) + i);
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx)) + i), Jp_delta_x));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx + 1)) + i), Jp_delta_y));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF)) + i), delta_a));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 1)) + i), delta_b));
                        _mm_store_ps(((float *) &resApprox) + i, rtz);
                    }
                }

                // need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
                Vec2f JI_r(0, 0);
                Vec2f Jab_r(0, 0);
                float rr = 0;
                for (int i = 0; i < patternNum; i++) {
                    JI_r[0] += resApprox[i] * rJ->JIdx[0][i];  // 残差 * dr21/dx  
                    JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
                    Jab_r[0] += resApprox[i] * rJ->JabF[0][i];  // 残差 * dr21/dab
                    Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
                    rr += resApprox[i] * resApprox[i];
                }
                //计算了 Hessian 矩阵,这里的 Hessian 矩阵是存储了两个帧之间的相互信息  acc 8*8  acc = [J r]^T * [J r]
                // 左上角10*10   4个内参 + 6个位姿
                acc[tid][htIDX].update(
                        rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                        rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                        rJ->JIdx2(0, 0), rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));
                // 右下角 3*3  2个光度参数 + 1个残差
                acc[tid][htIDX].updateBotRight(
                        rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0],
                        rJ->Jab2(1, 1), Jab_r[1], rr);
                // 非对角线部分 右上 10*3
                acc[tid][htIDX].updateTopRight(
                        rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                        rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                        rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
                        rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1),
                        JI_r[0], JI_r[1]);

                Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;  // dr21/dx2 ^T * dr21/dd1
                // 这三个成员变量将用于计算逆深度的优化量。
                bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];    // dr21/dd1 ^T
                Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);                          // dr21/dd1 ^T * dr21/dd1
                Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];   // dr21/dC ^T * dr21/dd1

                nres[tid]++;
            }

            if (mode == 0) {
                p->Hdd_accAF = Hdd_acc;
                p->bd_accAF = bd_acc;
                p->Hcd_accAF = Hcd_acc;
            }
            if (mode == 1 || mode == 2) {
                p->Hdd_accLF = Hdd_acc;
                p->bd_accLF = bd_acc;
                p->Hcd_accLF = Hcd_acc;
            }
            if (mode == 2) {
                p->Hcd_accAF.setZero();
                p->Hdd_accAF = 0;
                p->bd_accAF = 0;
            }

        }

        template void
        AccumulatedTopHessianSSE::addPoint<0>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        template void
        AccumulatedTopHessianSSE::addPoint<1>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        template void
        AccumulatedTopHessianSSE::addPoint<2>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        // for multi-fisheye
        template<int mode>
        void AccumulatedTopHessianSSE::addPointforMF(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid)
        {
            assert(mode == 0 || mode == 1 || mode == 2);

            //VecCf dc = ef->cDeltaF;
            float dd = p->deltaF;

            float bd_acc = 0;
            float Hdd_acc = 0;
            //VecCf Hcd_acc = VecCf::Zero();

            int test = 0;
            // cout << "release to 1" << endl;

            for(shared_ptr<PointFrameResidual> &r : p->residuals)   // 对该点所有残差进行遍历
            {
                if (mode == 0) 
                {
                    if (r->isLinearized || !r->isActive())
                        continue;
                }
                if (mode == 1)    // 计算旧的残差, 之前计算过得
                {
                    if (!r->isLinearized || !r->isActive())
                        continue;
                }
                if (mode == 2)    // // 边缘化计算的情况
                {    // marginalize, must be already linearized
                    if (!r->isActive())
                        continue;
                    assert(r->isLinearized);
                }

                test = 1; 

                shared_ptr<RawResidualJacobian> rJ = r->J;
                int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
                Mat156f dp = ef->adHTdeltaFforMF[htIDX];

                VecNRf resApprox;  // 残差 8×1的误差向量
                if (mode == 0)   //  active
                    resApprox = rJ->resF;
                if (mode == 2)   //  marginalize // 边缘化时使用的
                    resApprox = r->res_toZeroF;
                if (mode == 1)   // linearized
                { 
                    // cout << "release to 2" << endl;
                    // compute Jp*delta
                    __m128 Jp_delta_x = _mm_set1_ps(
                            rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdd[0] * dd);
                    __m128 Jp_delta_y = _mm_set1_ps(
                            rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdd[1] * dd);
                    int abidx =2* (r->hcamnum * r->camNums + r->tcamnum);
                    __m128 delta_a = _mm_set1_ps((float) (dp[6 + abidx]));
                    __m128 delta_b = _mm_set1_ps((float) (dp[7 + abidx]));

                    // LOG(INFO) << "release to 3" << endl;

                    for (int i = 0; i < patternNum; i += 4) 
                    {
                        // PATTERN: rtz = resF - [JI*Jp Ja]*delta.
                        __m128 rtz = _mm_load_ps(((float *) &r->res_toZeroF) + i);
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx)) + i), Jp_delta_x));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JIdx + 1)) + i), Jp_delta_y));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF)) + i), delta_a));
                        rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *) (rJ->JabF + 1)) + i), delta_b));
                        _mm_store_ps(((float *) &resApprox) + i, rtz);
                    }
                }

                // test = 2;
                // cout << "release addPointforMF to 4" << endl;
                // need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
                Vec2f JI_r(0, 0);
                Vec2f Jab_r(0, 0);
                float rr = 0;   // 最右下角 一个 r*r
                for (int i = 0; i < patternNum; i++) 
                {
                    JI_r[0] += resApprox[i] * rJ->JIdx[0][i];  // 残差 * dr21/dx  
                    JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
                    Jab_r[0] += resApprox[i] * rJ->JabF[0][i];  // 残差 * dr21/dab
                    Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
                    rr += resApprox[i] * resApprox[i];
                }

                test = 3;
                // cout << "release to 5" << endl;
                //计算了 Hessian 矩阵,这里的 Hessian 矩阵是存储了两个帧之间的相互信息  accforMF 8*8帧  acc = [J r]^T * [J r]
                // 左上角6*6   6个位姿
                accforMF[tid][htIDX].update(
                        rJ->Jpdxi[0].data(), rJ->Jpdxi[1].data(),
                        rJ->JIdx2(0, 0), rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));
                // 右下角 51*51  5*5*2个光度参数 + 1个残差
                accforMF[tid][htIDX].updateBotRight(
                        rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0],
                        rJ->Jab2(1, 1), Jab_r[1], rr, r->hcamnum, r->tcamnum);
                // 非对角线部分 右上 6*51
                accforMF[tid][htIDX].updateTopRight(
                        rJ->Jpdxi[0].data(), rJ->Jpdxi[1].data(),
                        rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
                        rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1),
                        JI_r[0], JI_r[1], r->hcamnum, r->tcamnum);

                Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;  // dr21/dx2 ^T * dx2/dd1   // dr21/dd1
                // 这三个成员变量将用于计算逆深度的优化量。
                bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];    // dr21/dd1 ^T  //* 残差*逆深度J
                Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);                          // dr21/dd1 ^T * dr21/dd1   //* 光度对逆深度hessian
                //Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];   // dr21/dC ^T * dr21/dd1
                nres[tid]++;
            }

            test = 10;
            // LOG(INFO) << "release to over" << endl;

            if (mode == 0) 
            {
                p->Hdd_accAF = Hdd_acc;
                p->bd_accAF = bd_acc;
                //p->Hcd_accAF = Hcd_acc;
            }
            if (mode == 1 || mode == 2) 
            {
                p->Hdd_accLF = Hdd_acc;
                p->bd_accLF = bd_acc;
                //p->Hcd_accLF = Hcd_acc;
            }
            if (mode == 2) {
                //p->Hcd_accAF.setZero();
                p->Hdd_accAF = 0;
                p->bd_accAF = 0;
            }
        }

        template void
        AccumulatedTopHessianSSE::addPointforMF<0>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        template void
        AccumulatedTopHessianSSE::addPointforMF<1>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);

        template void
        AccumulatedTopHessianSSE::addPointforMF<2>(shared_ptr<PointHessian> p, EnergyFunctional const *const ef, int tid);


        void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior,
                                                    bool useDelta, int tid) {
            H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
            b = VecX::Zero(nframes[tid] * 8 + CPARS);

            for (int h = 0; h < nframes[tid]; h++)
                for (int t = 0; t < nframes[tid]; t++) {
                    int hIdx = CPARS + h * 8;
                    int tIdx = CPARS + t * 8;
                    int aidx = h + nframes[tid] * t;


                    acc[tid][aidx].finish();
                    if (acc[tid][aidx].num == 0) continue;

                    MatPCPC accH = acc[tid][aidx].H.cast<double>();


                    H.block<8, 8>(hIdx, hIdx).noalias() +=
                            EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();

                    H.block<8, 8>(tIdx, tIdx).noalias() +=
                            EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                    H.block<8, 8>(hIdx, tIdx).noalias() +=
                            EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                    H.block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

                    H.block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

                    H.topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

                    b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

                    b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

                    b.head<CPARS>().noalias() += accH.block<CPARS, 1>(0, 8 + CPARS);
                }

            // ----- new: copy transposed parts.
            for (int h = 0; h < nframes[tid]; h++) {
                int hIdx = CPARS + h * 8;
                H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

                for (int t = h + 1; t < nframes[tid]; t++) {
                    int tIdx = CPARS + t * 8;
                    H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
                    H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
                }
            }


            if (usePrior) {
                assert(useDelta);
                H.diagonal().head<CPARS>() += EF->cPrior;
                b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
                for (int h = 0; h < nframes[tid]; h++) {
                    H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
                    b.segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
                }
            }
        }

        // for multi-fisheye 
        void AccumulatedTopHessianSSE::stitchDoubleforMF(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior,
                                                    bool useDelta, int tid) 
        {
            H = MatXX::Zero(nframes[tid] * 16, nframes[tid] * 16);
            b = VecX::Zero(nframes[tid] * 16);

            for (int h = 0; h < nframes[tid]; h++)
                for (int t = 0; t < nframes[tid]; t++) {
                    int hIdx = h * 16;
                    int tIdx = t * 16;
                    int aidx = h + nframes[tid] * t;


                    accforMF[tid][aidx].finish();
                    if (accforMF[tid][aidx].num == 0) continue;

                   MatMFMF accH = accforMF[tid][aidx].H.cast<double>();

                    // 每帧 16 个变量 6 pose 10 = 5*2 光度
                    H.block<6, 6>(hIdx, hIdx).noalias() +=
                            EF->adHostforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 6>(0, 0) * EF->adHostforMF[aidx].block<6, 6>(0, 0).transpose();

                    H.block<6, 6>(tIdx, tIdx).noalias() +=
                            EF->adTargetforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 6>(0, 0) * EF->adTargetforMF[aidx].block<6, 6>(0, 0).transpose();

                    H.block<6, 6>(hIdx, tIdx).noalias() +=
                            EF->adHostforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 6>(0, 0) * EF->adTargetforMF[aidx].block<6, 6>(0, 0).transpose();

                    b.segment<6>(hIdx).noalias() += EF->adHostforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 1>(0, 56);

                    b.segment<6>(tIdx).noalias() += EF->adTargetforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 1>(0, 56);

                    for (int hc = 0; hc < 5; hc++) 
                    {
                        for (int tc = 0; tc < 5; tc++)
                        {    
                            int abidx = 6 + (hc * 5 + tc)*2;
                            int habidx = hIdx + 6 + hc * 2;
                            int tabidx = tIdx + 6 + tc * 2;
                            H.block<2, 2>(habidx, habidx).noalias() += 
                                EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 2>(abidx, abidx) * EF->adHostforMF[aidx].block<2, 2>(abidx, abidx).transpose();

                            H.block<2, 2>(tabidx, tabidx).noalias() += 
                                EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 2>(abidx, abidx) * EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx).transpose();
                            
                            H.block<2, 2>(habidx, tabidx).noalias() += 
                                EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 2>(abidx, abidx) * EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx).transpose();

                            b.segment<2>(habidx).noalias() += EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 1>(abidx, 56);

                            b.segment<2>(tabidx).noalias() += EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 1>(abidx, 56);
                        }
                    }
                }

            // ----- new: copy transposed parts.
            for (int h = 0; h < nframes[tid]; h++) {
                int hIdx =  h * 16;
                for (int t = h + 1; t < nframes[tid]; t++) 
                {
                    int tIdx = t * 16;
                    H.block<16, 16>(hIdx, tIdx).noalias() += H.block<16, 16>(tIdx, hIdx).transpose();
                    H.block<16, 16>(tIdx, hIdx).noalias() = H.block<16, 16>(hIdx, tIdx).transpose();
                }
            }


            if (usePrior) 
            {
                assert(useDelta);
                for (int h = 0; h < nframes[tid]; h++) 
                {
                    H.diagonal().segment<16>(h * 16) += EF->frames[h]->priorforMF;
                    b.segment<16>(h * 16) += EF->frames[h]->priorforMF.cwiseProduct(EF->frames[h]->delta_priorforMF);
                }
            }
        }

        void AccumulatedTopHessianSSE::stitchDoubleInternal(MatXX *H, VecX *b, EnergyFunctional const *const EF,
                                                            bool usePrior, int min, int max, Vec10 *stats, int tid) {
            int toAggregate = NUM_THREADS;
            if (tid == -1) {
                toAggregate = 1;
                tid = 0;
            }    // special case: if we dont do multithreading, dont aggregate.
            if (min == max) return;


            for (int k = min; k < max; k++) {
                int h = k % nframes[0];
                int t = k / nframes[0];

                int hIdx = CPARS + h * 8;
                int tIdx = CPARS + t * 8;
                int aidx = h + nframes[0] * t;

                assert(aidx == k);

                MatPCPC accH = MatPCPC::Zero();

                for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                    acc[tid2][aidx].finish();
                    if (acc[tid2][aidx].num == 0) continue;
                    accH += acc[tid2][aidx].H.cast<double>();
                }

                H[tid].block<8, 8>(hIdx, hIdx).noalias() +=
                        EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();

                H[tid].block<8, 8>(tIdx, tIdx).noalias() +=
                        EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                H[tid].block<8, 8>(hIdx, tIdx).noalias() +=
                        EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

                H[tid].block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

                H[tid].block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

                H[tid].topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

                b[tid].segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);

                b[tid].segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);

                b[tid].head<CPARS>().noalias() += accH.block<CPARS, 1>(0, CPARS + 8);

            }


            // only do this on one thread.
            if (min == 0 && usePrior) {
                H[tid].diagonal().head<CPARS>() += EF->cPrior;
                b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
                for (int h = 0; h < nframes[tid]; h++) {
                    H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
                    b[tid].segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);

                }
            }
        }

        // for multi-fisheye 构造Hessian矩阵, b=Jres矩阵
        void AccumulatedTopHessianSSE::stitchDoubleInternalforMF(MatXX *H, VecX *b, EnergyFunctional const *const EF,
                                                            bool usePrior, int min, int max, Vec10 *stats, int tid) {
            int toAggregate = NUM_THREADS;
            if (tid == -1) {
                toAggregate = 1;
                tid = 0;
            }    // special case: if we dont do multithreading, dont aggregate.
            if (min == max) return;

            // ofstream of1;
            // string str = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/relativeH.txt";
            // of1.open(str.data());
            for (int k = min; k < max; k++) 
            {
                int h = k % nframes[0];
                int t = k / nframes[0];

                int hIdx = h * 16;
                int tIdx = t * 16;
                int aidx = h + nframes[0] * t;

                assert(aidx == k);

                MatMFMF accH = MatMFMF::Zero();

                for (int tid2 = 0; tid2 < toAggregate; tid2++) 
                {
                    accforMF[tid2][aidx].finish();
                    if (accforMF[tid2][aidx].num == 0)  continue;
                    accH += accforMF[tid2][aidx].H.cast<double>();  // 不同线程之间的加起来
                    //of1 << accforMF[tid2][aidx].H.cast<double>() << endl;
                }

                // 每帧 16 个变量 6 pose 10 = 5*2 光度
                // cout << "accH.block<6, 6>(0, 0):" << endl << accH.block<6, 6>(0, 0)  << endl;
                // cout << "EF->adHostforMF[aidx].block<6, 6>(0, 0):" << endl << EF->adHostforMF[aidx].block<6, 6>(0, 0) << endl;
                // cout << "EF->adTargetforMF[aidx].block<6, 6>(0, 0):" << endl << EF->adTargetforMF[aidx].block<6, 6>(0, 0) << endl;
                H[tid].block<6, 6>(hIdx, hIdx).noalias() +=
                        EF->adHostforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 6>(0, 0) * EF->adHostforMF[aidx].block<6, 6>(0, 0).transpose();

                H[tid].block<6, 6>(tIdx, tIdx).noalias() +=
                        EF->adTargetforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 6>(0, 0) * EF->adTargetforMF[aidx].block<6, 6>(0, 0).transpose();

                H[tid].block<6, 6>(hIdx, tIdx).noalias() +=
                        EF->adHostforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 6>(0, 0) * EF->adTargetforMF[aidx].block<6, 6>(0, 0).transpose();

                b[tid].segment<6>(hIdx).noalias() += EF->adHostforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 1>(0, 56);

                b[tid].segment<6>(tIdx).noalias() += EF->adTargetforMF[aidx].block<6, 6>(0, 0) * accH.block<6, 1>(0, 56);

                for (int hc = 0; hc < multicamera_nums; hc++) 
                {
                    for (int tc = 0; tc < multicamera_nums; tc++)
                    {    
                        int abidx = 6 + (hc * multicamera_nums + tc) * 2;
                        int habidx = hIdx + 6 + hc * 2;
                        int tabidx = tIdx + 6 + tc * 2;
                        // if(hc == tc)
                        // {
                        //     cout << "accH.block<2, 2>(abidx, abidx):" << endl << accH.block<2, 2>(abidx, abidx) << endl;
                        //     cout << "EF->adHostforMF[aidx].block<2, 2>(abidx, abidx):" << endl << EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) << endl;
                        //     cout << "EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx):" << endl << EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx) << endl;
                        // }
                        H[tid].block<2, 2>(habidx, habidx).noalias() += 
                            EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 2>(abidx, abidx) * EF->adHostforMF[aidx].block<2, 2>(abidx, abidx).transpose();

                        H[tid].block<2, 2>(tabidx, tabidx).noalias() += 
                            EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 2>(abidx, abidx) * EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx).transpose();
                        
                        H[tid].block<2, 2>(habidx, tabidx).noalias() += 
                            EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 2>(abidx, abidx) * EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx).transpose();

                        b[tid].segment<2>(habidx).noalias() += EF->adHostforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 1>(abidx, 56);

                        b[tid].segment<2>(tabidx).noalias() += EF->adTargetforMF[aidx].block<2, 2>(abidx, abidx) * accH.block<2, 1>(abidx, 56);
                    }
                }

            }
            //of1.close();

            // only do this on one thread.
            if (min == 0 && usePrior) 
            {
                for (int h = 0; h < nframes[tid]; h++) 
                {
                    H[tid].diagonal().segment<16>(h * 16) += EF->frames[h]->priorforMF;
                    b[tid].segment<16>(h * 16) += EF->frames[h]->priorforMF.cwiseProduct(EF->frames[h]->delta_priorforMF);

                }
            }
        }

    };

}