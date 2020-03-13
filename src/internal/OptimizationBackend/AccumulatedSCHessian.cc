#include "internal/OptimizationBackend/EnergyFunctional.h"
#include "internal/OptimizationBackend/AccumulatedSCHessian.h"
#include "internal/PointHessian.h"

namespace ldso {

    namespace internal {

        void AccumulatedSCHessianSSE::addPoint(shared_ptr<PointHessian> p, bool shiftPriorToZero, int tid) {

            int ngoodres = 0;
            for (auto r : p->residuals)
                if (r->isActive())
                    ngoodres++;

            if (ngoodres == 0) {
                p->HdiF = 0;
                p->bdSumF = 0;
                p->idepth_hessian = 0;
                p->maxRelBaseline = 0;
                return;
            }

            float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;
            if (H < 1e-10) H = 1e-10;
            p->idepth_hessian = H;
            p->HdiF = 1.0 / H;   // Hdd^-1  point 逆深度的 hession 矩阵的逆
            p->bdSumF = p->bd_accAF + p->bd_accLF;
            if (shiftPriorToZero) p->bdSumF += p->priorF * p->deltaF;
            VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;
            accHcc[tid].update(Hcd, Hcd, p->HdiF);   // Schur Hession 累积了内参部分
            accbc[tid].update(Hcd, p->bdSumF * p->HdiF);   // Schur  Jb  累积了内参部分

            assert(std::isfinite((float) (p->HdiF)));

            int nFrames2 = nframes[tid] * nframes[tid];
            for (auto r1 : p->residuals) {
                if (!r1->isActive()) continue;
                int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];

                for (auto r2 : p->residuals) {
                    if (!r2->isActive())
                        continue;

                    accD[tid][r1ht + r2->targetIDX * nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);  // Schur Hession 累积了位姿和光度信息
                }

                accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);  // Schur Hession 非对角线部分
                accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);   // // Schur  Jb  累积了位姿和光度信息
            }
        }

        // for multi-fisheye
        void AccumulatedSCHessianSSE::addPointforMF(shared_ptr<PointHessian> p, bool shiftPriorToZero, int tid) 
        {

            int ngoodres = 0;
            for (auto r : p->residuals)
                if (r->isActive())
                    ngoodres++;

            if (ngoodres == 0) 
            {
                p->HdiF = 0;
                p->bdSumF = 0;
                p->idepth_hessian = 0;
                p->maxRelBaseline = 0;
                return;
            }

            // hessian + 边缘化得到hessian + 先验hessian
            float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;
            if (H < 1e-10) H = 1e-10;
            p->idepth_hessian = H;
            p->HdiF = 1.0 / H;   // Hdd^-1  point 逆深度的 hession 矩阵的逆  
            p->bdSumF = p->bd_accAF + p->bd_accLF;
            if (shiftPriorToZero) p->bdSumF += p->priorF * p->deltaF;
            //VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;
            //accHcc[tid].update(Hcd, Hcd, p->HdiF);   // Schur Hession 累积了内参部分 
            //accbc[tid].update(Hcd, p->bdSumF * p->HdiF);   // Schur  Jb  累积了内参部分

            assert(std::isfinite((float) (p->HdiF)));

            int nFrames2 = nframes[tid] * nframes[tid];
            // 两次遍历 构建逆深度的hessian矩阵  (每个点有多个残差)
            for (auto r1 : p->residuals) 
            {
                if (!r1->isActive()) continue;
                int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];     // 点的hostIDX应该是固定的

                for (auto r2 : p->residuals) 
                {
                    if (!r2->isActive())
                        continue;
                    // Hfd_1 * Hdd_inv * Hfd_2^T,  f = [xi, a b]位姿 光度
                    accDforMF[tid][r1ht + r2->targetIDX * nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF, r1->hcamnum, r1->tcamnum, r2->hcamnum, r2->tcamnum);  // Schur Hession 累积了位姿和光度信息
                }

                //accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);  // Schur Hession 非对角线部分
                accEBforMF[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF, r1->hcamnum, r1->tcamnum);   // // Schur  Jb  累积了位姿和光度信息
            }
        }

        void AccumulatedSCHessianSSE::stitchDoubleInternal(
                MatXX *H, VecX *b, EnergyFunctional const *const EF,
                int min, int max, Vec10 *stats, int tid) {
            int toAggregate = NUM_THREADS;
            if (tid == -1) {
                toAggregate = 1;
                tid = 0;
            }    // special case: if we dont do multithreading, dont aggregate.
            if (min == max) return;


            int nf = nframes[0];
            int nframes2 = nf * nf;

            for (int k = min; k < max; k++) {
                int i = k % nf;
                int j = k / nf;

                int iIdx = CPARS + i * 8;
                int jIdx = CPARS + j * 8;
                int ijIdx = i + nf * j;

                Mat8C Hpc = Mat8C::Zero();
                Vec8 bp = Vec8::Zero();

                for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                    accE[tid2][ijIdx].finish();
                    accEB[tid2][ijIdx].finish();
                    Hpc += accE[tid2][ijIdx].A1m.cast<double>();
                    bp += accEB[tid2][ijIdx].A1m.cast<double>();
                }

                H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
                H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;
                b[tid].segment<8>(iIdx) += EF->adHost[ijIdx] * bp;
                b[tid].segment<8>(jIdx) += EF->adTarget[ijIdx] * bp;


                for (int k = 0; k < nf; k++) {
                    int kIdx = CPARS + k * 8;
                    int ijkIdx = ijIdx + k * nframes2;
                    int ikIdx = i + nf * k;

                    Mat88 accDM = Mat88::Zero();

                    for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                        accD[tid2][ijkIdx].finish();
                        if (accD[tid2][ijkIdx].num == 0) continue;
                        accDM += accD[tid2][ijkIdx].A1m.cast<double>();
                    }

                    H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
                    H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
                    H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
                    H[tid].block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
                }
            }

            if (min == 0) {
                for (int tid2 = 0; tid2 < toAggregate; tid2++) {
                    accHcc[tid2].finish();
                    accbc[tid2].finish();
                    H[tid].topLeftCorner<CPARS, CPARS>() += accHcc[tid2].A1m.cast<double>();
                    b[tid].head<CPARS>() += accbc[tid2].A1m.cast<double>();
                }
            }
        }

        // for multi-fisheye
        void AccumulatedSCHessianSSE::stitchDoubleInternalforMF(
                MatXX *H, VecX *b, EnergyFunctional const *const EF,
                int min, int max, Vec10 *stats, int tid) 
        {
            int toAggregate = NUM_THREADS;
            if (tid == -1) 
            {
                toAggregate = 1;
                tid = 0;
            }    // special case: if we dont do multithreading, dont aggregate.
            if (min == max) return;


            int nf = nframes[0];
            int nframes2 = nf * nf;

            for (int k = min; k < max; k++) 
            {
                int i = k % nf;
                int j = k / nf;

                int iIdx = i * 16;
                int jIdx = j * 16;
                int ijIdx = i + nf * j;   // host:i     target:j

                //Mat8C Hpc = Mat8C::Zero();
                Vec56 bp = Vec56::Zero();

                for (int tid2 = 0; tid2 < toAggregate; tid2++) 
                {
                    //accE[tid2][ijIdx].finish();
                    accEBforMF[tid2][ijIdx].finish();
                    //Hpc += accE[tid2][ijIdx].A1m.cast<double>();
                    bp += accEBforMF[tid2][ijIdx].A1m.cast<double>();
                }

                // H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
                // H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;
                b[tid].segment<6>(iIdx) += EF->adHostforMF[ijIdx].block<6, 6>(0, 0) * bp.segment<6>(0);
                b[tid].segment<6>(jIdx) += EF->adTargetforMF[ijIdx].block<6, 6>(0, 0) * bp.segment<6>(0);
                for (int hc = 0; hc < 5; hc++) 
                {
                    for (int tc = 0; tc < 5; tc++)
                    {    
                        int abidx = 6 + (hc * 5 + tc)*2;
                        int iabidx = iIdx + 6 + hc * 2;
                        int jabidx = jIdx + 6 + tc * 2;
                        b[tid].segment<2>(iabidx).noalias() += EF->adHostforMF[ijIdx].block<2, 2>(abidx, abidx) * bp.segment<2>(abidx);
                        b[tid].segment<2>(jabidx).noalias() += EF->adTargetforMF[ijIdx].block<2, 2>(abidx, abidx) * bp.segment<2>(abidx);
                    }
                }


                for (int k = 0; k < nf; k++) 
                {
                    int kIdx = k * 16;
                    int ijkIdx = ijIdx + k * nframes2;
                    int ikIdx = i + nf * k;     //host:i    target:k

                    Mat5656 accDM = Mat5656::Zero();

                    for (int tid2 = 0; tid2 < toAggregate; tid2++) 
                    {
                        accDforMF[tid2][ijkIdx].finish();
                        if (accDforMF[tid2][ijkIdx].num == 0) continue;
                        accDM += accDforMF[tid2][ijkIdx].A1m.cast<double>();
                    }

                    // host:i       target:j,k(j可以等于k)，两两之间需要构建hessian矩阵   同一个点可能有多个energy
                    H[tid].block<6, 6>(iIdx, iIdx) += EF->adHostforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adHostforMF[ikIdx].block<6, 6>(0, 0).transpose();
                    H[tid].block<6, 6>(jIdx, kIdx) += EF->adTargetforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adTargetforMF[ikIdx].block<6, 6>(0, 0).transpose();
                    H[tid].block<6, 6>(jIdx, iIdx) += EF->adTargetforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adHostforMF[ikIdx].block<6, 6>(0, 0).transpose();
                    H[tid].block<6, 6>(iIdx, kIdx) += EF->adHostforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adTargetforMF[ikIdx].block<6, 6>(0, 0).transpose();

                    for (int hc = 0; hc < 5; hc++) 
                    {
                        for (int tc = 0; tc < 5; tc++)
                        {    
                            int abidx = 6 + (hc * 5 + tc)*2;
                            int iabidx = iIdx + 6 + hc * 2;
                            int jabidx = jIdx + 6 + tc * 2;
                            int kabidx = kIdx + 6 + tc * 2;
                            H[tid].block<2, 2>(iabidx, jabidx) += EF->adHostforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adHostforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                            H[tid].block<2, 2>(jabidx, kabidx) += EF->adTargetforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adTargetforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                            H[tid].block<2, 2>(jabidx, iabidx) += EF->adTargetforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adHostforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                            H[tid].block<2, 2>(jabidx, kabidx) += EF->adHostforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adTargetforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                        }
                    }

                }
            }

        }

        void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, const EnergyFunctional *const EF, int tid) {

            int nf = nframes[0];
            int nframes2 = nf * nf;

            H = MatXX::Zero(nf * 8 + CPARS, nf * 8 + CPARS);
            b = VecX::Zero(nf * 8 + CPARS);


            for (int i = 0; i < nf; i++)
                for (int j = 0; j < nf; j++) {
                    int iIdx = CPARS + i * 8;
                    int jIdx = CPARS + j * 8;
                    int ijIdx = i + nf * j;

                    accE[tid][ijIdx].finish();
                    accEB[tid][ijIdx].finish();

                    Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();
                    Vec8 accEBV = accEB[tid][ijIdx].A1m.cast<double>();

                    H.block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * accEM;
                    H.block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * accEM;

                    b.segment<8>(iIdx) += EF->adHost[ijIdx] * accEBV;
                    b.segment<8>(jIdx) += EF->adTarget[ijIdx] * accEBV;

                    for (int k = 0; k < nf; k++) {
                        int kIdx = CPARS + k * 8;
                        int ijkIdx = ijIdx + k * nframes2;
                        int ikIdx = i + nf * k;

                        accD[tid][ijkIdx].finish();
                        if (accD[tid][ijkIdx].num == 0) continue;
                        Mat88 accDM = accD[tid][ijkIdx].A1m.cast<double>();

                        H.block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();

                        H.block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();

                        H.block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();

                        H.block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
                    }
                }

            accHcc[tid].finish();
            accbc[tid].finish();
            H.topLeftCorner<CPARS, CPARS>() = accHcc[tid].A1m.cast<double>();
            b.head<CPARS>() = accbc[tid].A1m.cast<double>();

            // ----- new: copy transposed parts for calibration only.
            for (int h = 0; h < nf; h++) {
                int hIdx = CPARS + h * 8;
                H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
            }
        }

        void AccumulatedSCHessianSSE::stitchDoubleforMF(MatXX &H, VecX &b, const EnergyFunctional *const EF, int tid) 
        {

            int nf = nframes[0];
            int nframes2 = nf * nf;

            H = MatXX::Zero(nf * 16, nf * 16);
            b = VecX::Zero(nf * 16);


            for (int i = 0; i < nf; i++)
                for (int j = 0; j < nf; j++) {
                    int iIdx = i * 16;
                    int jIdx = j * 16;
                    int ijIdx = i + nf * j;

                    //accE[tid][ijIdx].finish();
                    accEBforMF[tid][ijIdx].finish();

                    //Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();
                    Vec56 accEBV = accEBforMF[tid][ijIdx].A1m.cast<double>();

                    // H.block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * accEM;
                    // H.block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * accEM;

                    b.segment<6>(iIdx) += EF->adHostforMF[ijIdx].block<6, 6>(0 ,0) * accEBV.segment<6>(0);;
                    b.segment<6>(jIdx) += EF->adTargetforMF[ijIdx].block<6, 6>(0 ,0) * accEBV.segment<6>(0);

                    for (int hc = 0; hc < 5; hc++) 
                    {
                        for (int tc = 0; tc < 5; tc++)
                        {    
                            int abidx = 6 + (hc * 5 + tc)*2;
                            int iabidx = iIdx + 6 + hc * 2;
                            int jabidx = jIdx + 6 + tc * 2;
                            b.segment<2>(iabidx).noalias() += EF->adHostforMF[ijIdx].block<2, 2>(abidx, abidx) * accEBV.segment<2>(abidx);
                            b.segment<2>(jabidx).noalias() += EF->adTargetforMF[ijIdx].block<2, 2>(abidx, abidx) * accEBV.segment<2>(abidx);
                        }
                    }
                    for (int k = 0; k < nf; k++) 
                    {
                        int kIdx =  k * 16;
                        int ijkIdx = ijIdx + k * nframes2;
                        int ikIdx = i + nf * k;

                        accDforMF[tid][ijkIdx].finish();
                        if (accDforMF[tid][ijkIdx].num == 0) continue;
                        Mat5656 accDM = accDforMF[tid][ijkIdx].A1m.cast<double>();
                        // host:i       target:j,k(j可以等于k)，两两之间需要构建hessian矩阵   同一个点可能有多个energy
                        H.block<6, 6>(iIdx, iIdx) += EF->adHostforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adHostforMF[ikIdx].block<6, 6>(0, 0).transpose();
                        H.block<6, 6>(jIdx, kIdx) += EF->adTargetforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adTargetforMF[ikIdx].block<6, 6>(0, 0).transpose();
                        H.block<6, 6>(jIdx, iIdx) += EF->adTargetforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adHostforMF[ikIdx].block<6, 6>(0, 0).transpose();
                        H.block<6, 6>(iIdx, kIdx) += EF->adHostforMF[ijIdx].block<6, 6>(0, 0) * accDM.block<6, 6>(0, 0) * EF->adTargetforMF[ikIdx].block<6, 6>(0, 0).transpose();

                        for (int hc = 0; hc < 5; hc++) 
                        {
                            for (int tc = 0; tc < 5; tc++)
                            {    
                                int abidx = 6 + (hc * 5 + tc)*2;
                                int iabidx = iIdx + 6 + hc * 2;
                                int jabidx = jIdx + 6 + tc * 2;
                                int kabidx = kIdx + 6 + tc * 2;
                                H.block<2, 2>(iabidx, jabidx) += EF->adHostforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adHostforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                                H.block<2, 2>(jabidx, kabidx) += EF->adTargetforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adTargetforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                                H.block<2, 2>(jabidx, iabidx) += EF->adTargetforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adHostforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                                H.block<2, 2>(jabidx, kabidx) += EF->adHostforMF[ijIdx].block<2, 2>(abidx, abidx) * accDM.block<2, 2>(abidx, abidx) * EF->adTargetforMF[ikIdx].block<2, 2>(abidx, abidx).transpose();
                            }
                        }
                    }
                }
        }

    }

}