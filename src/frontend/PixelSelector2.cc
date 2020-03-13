#include "frontend/PixelSelector2.h"
#include "internal/FrameHessian.h"
#include "internal/GlobalCalib.h"

using namespace ldso::internal;

namespace ldso {

    PixelSelector::PixelSelector(int w, int h) {
        w0 = w;
        h0 = h;
        randomPattern = new unsigned char[w * h];
        std::srand(3141592);    // want to be deterministic.
        for (int i = 0; i < w * h; i++)
            randomPattern[i] = rand() & 0xFF;  // rand()返回值二进制的高16位变成0，低16位不变  0xFF 补码就是-127,普通的数字是+255
        currentPotential = 3;
        gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];   // 将图像分为32*32块
        ths = new float[(w / 32) * (h / 32) + 100];    // 每块生成一个动态阈值
        thsSmoothed = new float[(w / 32) * (h / 32) + 100];    // 32*32的块做阈值平滑处理，赋值给thsSmoothed
    }

    PixelSelector::~PixelSelector() {
        delete[] randomPattern;
        delete[] gradHist;
        delete[] ths;
        delete[] thsSmoothed;
    }

    int computeHistQuantil(int *hist, float below) {
        int th = hist[0] * below + 0.5f;
        for (int i = 0; i < 90; i++) {
            th -= hist[i + 1];
            if (th < 0) return i;
        }
        return 90;
    }

    // 对absSquaredGrad处理，先分割成32*32的块，在每块中，遍历每个点{直方图统计梯度0-50，赋值给gradHist. }， 
    // 然后给每块生成一个动态阈值，这个阈值就是直方图中1梯度开始求和，直到=0梯度点的数量 × setting_minGradHistCut时的梯度 ，  
    // 再+setting_minGradHistAdd， 赋值给ths

    // 然后对32*32的块做阈值平滑处理，赋值给thsSmoothed，这就是最终的动态阈值。后续select处理的时候点>>5，正好是对32做一个取值处理。
    void PixelSelector::makeHists(shared_ptr<FrameHessian> fh) {
        gradHistFrame = fh;
        float *mapmax0 = fh->absSquaredGrad[0];

        int w = wG[0];
        int h = hG[0];

        int w32 = w / 32;
        int h32 = h / 32;
        thsStep = w32;
        // 先得到每块的梯度直方图，然后计算得到阈值
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float *map0 = mapmax0 + 32 * x + 32 * y * w;
                int *hist0 = gradHist;// + 50*(x+y*w32);
                memset(hist0, 0, sizeof(int) * 50);  // sizeof(int)在16位int平台下是2；在32位int平台下是4；在64位int平台下是8。

                for (int j = 0; j < 32; j++)    // 遍历每个小块,统计梯度直方图
                    for (int i = 0; i < 32; i++) {
                        int it = i + 32 * x;
                        int jt = j + 32 * y;
                        if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1) continue;
                        int g = sqrtf(map0[i + j * w]);
                        if (g > 48) g = 48;
                        hist0[g + 1]++;
                        hist0[0]++;   //保存块内像素总数
                    }

                ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;  // 得到第x + y * w32块的阈值，直方图到setting_minGradHistCut范围时的梯度值
            }
        // 然后对32*32的块做阈值平滑处理，赋值给thsSmoothed，这就是最终的动态阈值
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float sum = 0, num = 0;
                if (x > 0) {
                    if (y > 0) {
                        num++;
                        sum += ths[x - 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x - 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x - 1 + (y) * w32];
                }

                if (x < w32 - 1) {
                    if (y > 0) {
                        num++;
                        sum += ths[x + 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x + 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x + 1 + (y) * w32];
                }

                if (y > 0) {
                    num++;
                    sum += ths[x + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + (y + 1) * w32];
                }
                num++;
                sum += ths[x + y * w32];

                thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
            }
    }

    //  for multi-fisheye image
    void PixelSelector::makeHists(shared_ptr<FrameHessian> fh, int camNum)
    {
        gradHistFrame = fh;
        float *mapmax0 = fh->vabsSquaredGrad[0][camNum];

        // ofstream outim;
        // string path = "/home/jiangfan/桌面/pan_dso_calib/pano_tracking/vabsSquaredGrad_makeHists" + to_string(camNum) + ".txt";
        // outim.open(path.c_str());

        int w = fh->UMF->getOriginalSize()(0);
        int h = fh->UMF->getOriginalSize()(1);

        // for (int iw = 0; iw < w;  iw++)
        //     for (int ih = 0; ih < h;  ih++)
        // {
        //     int idx = iw + ih * h;
        //     outim << iw << " " << ih << " " << mapmax0[idx] << " " << fh->vabsSquaredGrad[0][camNum][idx] <<endl;
        // }
        // outim.close();

        int w32 = w / 32;
        int h32 = h / 32;
        thsStep = w32;
        // 先得到每块的梯度直方图，然后计算得到阈值
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float *map0 = mapmax0 + 32 * x + 32 * y * w;
                int *hist0 = gradHist;// + 50*(x+y*w32);
                memset(hist0, 0, sizeof(int) * 50);  // sizeof(int)在16位int平台下是2；在32位int平台下是4；在64位int平台下是8。

                for (int j = 0; j < 32; j++)    // 遍历每个小块,统计梯度直方图
                    for (int i = 0; i < 32; i++) {
                        int it = i + 32 * x;
                        int jt = j + 32 * y;
                        if (it > w - 4 || jt > h - 4 || it < 4 || jt < 4) continue;
                        if(map0[i + j * w] < 0)
                        {
                            cout << "Cam Num:" << camNum << " x,y:" << it << " " << jt 
                                << "  grad min:" << map0[i + j * w] <<  "  grad orign:" << mapmax0[32 * x+i + (32 * y+j) * w] <<endl;
                        }
                        int g = sqrtf(map0[i + j * w]);
                        if (g > 48) g = 48;
                        hist0[g + 1]++;
                        hist0[0]++;   //保存块内像素总数
                    }

                ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;  // 得到第x + y * w32块的阈值，直方图到setting_minGradHistCut范围时的梯度值
            }
        // 然后对32*32的块做阈值平滑处理，赋值给thsSmoothed，这就是最终的动态阈值
        for (int y = 0; y < h32; y++)
            for (int x = 0; x < w32; x++) {
                float sum = 0, num = 0;
                if (x > 0) {
                    if (y > 0) {
                        num++;
                        sum += ths[x - 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x - 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x - 1 + (y) * w32];
                }

                if (x < w32 - 1) {
                    if (y > 0) {
                        num++;
                        sum += ths[x + 1 + (y - 1) * w32];
                    }
                    if (y < h32 - 1) {
                        num++;
                        sum += ths[x + 1 + (y + 1) * w32];
                    }
                    num++;
                    sum += ths[x + 1 + (y) * w32];
                }

                if (y > 0) {
                    num++;
                    sum += ths[x + (y - 1) * w32];
                }
                if (y < h32 - 1) {
                    num++;
                    sum += ths[x + (y + 1) * w32];
                }
                num++;
                sum += ths[x + y * w32];

                thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
            }
    }

    // makemaps在setFirst和makeNewTraces的时候用到。   
    int PixelSelector::makeMaps(const shared_ptr<FrameHessian> fh, float *map_out, float density,
                                int recursionsLeft, bool plot, float thFactor) {

        float numHave = 0;
        float numWant = density;
        float quotia;
        int idealPotential = currentPotential;

        if (fh != gradHistFrame) makeHists(fh);   // 生成32*32块梯度直方图，计算相应块直方图的动态阈值

        // select!
        Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

        // sub-select!
        numHave = n[0] + n[1] + n[2];
        quotia = numWant / numHave;

        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential + 1) * (currentPotential + 1);
        idealPotential = sqrtf(K / numWant) - 1;    // round down.
        if (idealPotential < 1) idealPotential = 1;

        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
            // re-sample to get more points!
            // potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        } else if (recursionsLeft > 0 && quotia < 0.25) {
            // re-sample to get less points!
            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;
            currentPotential = idealPotential;
            return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
        }

        int numHaveSub = numHave;
        if (quotia < 0.95) {
            int wh = wG[0] * hG[0];
            int rn = 0;
            unsigned char charTH = 255 * quotia;
            for (int i = 0; i < wh; i++) {
                if (map_out[i] != 0) {
                    if (randomPattern[rn] > charTH) {
                        map_out[i] = 0;
                        numHaveSub--;
                    }
                    rn++;
                }
            }
        }

        currentPotential = idealPotential;

        return numHaveSub;
    }

    //  for multi-fisheye image
    int PixelSelector::makeMapsforMF(const shared_ptr<FrameHessian> fh, int camNum, float *map_out, float density, int recursionsLeft, bool plot, float thFactor)
    {
        float numHave = 0;
        float numWant = density;
        float quotia;
        int idealPotential = currentPotential;

        if (fh != gradHistFrame) makeHists(fh, camNum);   // 生成32*32块梯度直方图，计算相应块直方图的动态阈值

        // select!
        Eigen::Vector3i n = this->selectforMF(fh, camNum, map_out, currentPotential, thFactor);

        // sub-select!
        numHave = n[0] + n[1] + n[2];
        quotia = numWant / numHave;

        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential + 1) * (currentPotential + 1);
        idealPotential = sqrtf(K / numWant) - 1;    // round down.
        if (idealPotential < 1) idealPotential = 1;

        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1) {
            // re-sample to get more points!
            // potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            currentPotential = idealPotential;
            return makeMapsforMF(fh, camNum, map_out, density, recursionsLeft - 1, plot, thFactor);
        } else if (recursionsLeft > 0 && quotia < 0.25) {
            // re-sample to get less points!
            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;
            currentPotential = idealPotential;
            return makeMapsforMF(fh, camNum, map_out, density, recursionsLeft - 1, plot, thFactor);
        }

        int numHaveSub = numHave;
        if (quotia < 0.95) {
            int wh = w0 * h0;
            int rn = 0;
            unsigned char charTH = 255 * quotia;
            for (int i = 0; i < wh; i++) {
                if (map_out[i] != 0) {
                    if (randomPattern[rn] > charTH) {
                        map_out[i] = 0;
                        numHaveSub--;
                    }
                    rn++;
                }
            }
        }

        currentPotential = idealPotential;
        gradHistFrame = nullptr;

        if(true)
        {
            int w = w0;
            int h = h0;


            MinimalImageB3 img(w,h);
            MinimalImageB3 img1(w,h);

            for(int i=0;i<w*h;i++)
            {
                float c = fh->vdI[camNum][i][0]*0.7;
                if(c>255) c=255;
                img.at(i) = Vec3b(c,c,c);
                img1.at(i) = Vec3b(c,c,c);
                float m = fh->vmask[0][camNum][i];
            }

            for(int y=0; y<h;y++)
                for(int x=0;x<w;x++)
                {
                    int i=x+y*w;
                    if(map_out[i] == 1)
                        img.setPixelCirc(x,y,Vec3b(0,255,0));
                    else if(map_out[i] == 2)
                        img.setPixelCirc(x,y,Vec3b(255,0,0));
                    else if(map_out[i] == 4)
                        img.setPixelCirc(x,y,Vec3b(0,0,255));
                }
        

        cv::Mat imga(h0,w0,CV_8UC3);
        cv::Mat imgb(h0,w0,CV_8UC3);
        //cv::Mat imgG(h0,w0,CV_8UC1);
        cv::Mat imgM(h0,w0,CV_8UC1);
        cv::Mat imgF(h0,w0,CV_8UC1);
        for(int y=0; y<h;y++)
            for(int x=0;x<w;x++)
        {
            //float v = (!std::isfinite(fh->vdI[i][camNum][0]) || v>255) ? 255 : fh->vdI[i][camNum][0];
            //imga.at<cv::Vec3b>(i) = cv::Vec3b(v, fh->vdI[i][camNum][1], fh->vdI[i][camNum][2]);
            int i=x+y*w;
            imga.at<cv::Vec3b>(y,x) = cv::Vec3b(img.at(i)(0), img.at(i)(1), img.at(i)(2));
            imgb.at<cv::Vec3b>(y,x) = cv::Vec3b(img1.at(i)(0), img1.at(i)(1), img1.at(i)(2));
            //imgG.at<uchar>(y,x) = fh->vabsSquaredGrad[0][camNum][i];
            imgM.at<uchar>(y,x) = fh->vmask[0][camNum][i];
            imgF.at<uchar>(y,x) = fh->vdfisheyeI[camNum][i];
        }
        string outpath = "/home/jiangfan/桌面/pan_dso_calib/pano_inital/";
        string outimg = outpath +  to_string(camNum) + "img.jpg";
        cv::imwrite(outimg, imga);

        // string outimg1 = outpath +  to_string(camNum) + "img_orign.jpg";
        // cv::imwrite(outimg1, imgb);

        // string outimgg = outpath +  to_string(camNum) + "img_g.jpg";
        // cv::imwrite(outimgg, imgG);

        // string outimgm = outpath +  to_string(camNum) + "img_m.jpg";
        // cv::imwrite(outimgm, imgM);

        // string outimgf = outpath +  to_string(camNum) + "img_f.jpg";
        // cv::imwrite(outimgf, imgF);

        }

        return numHaveSub;
    }
    // pot步长
    Eigen::Vector3i PixelSelector::select(const shared_ptr<FrameHessian> fh, float *map_out, int pot,
                                          float thFactor) {
        Eigen::Vector3f const *const map0 = fh->dI;

        float *mapmax0 = fh->absSquaredGrad[0];
        float *mapmax1 = fh->absSquaredGrad[1];
        float *mapmax2 = fh->absSquaredGrad[2];


        int w = wG[0];
        int w1 = wG[1];
        int w2 = wG[2];
        int h = hG[0];


        const Vec2f directions[16] = {
                Vec2f(0, 1.0000),
                Vec2f(0.3827, 0.9239),
                Vec2f(0.1951, 0.9808),
                Vec2f(0.9239, 0.3827),
                Vec2f(0.7071, 0.7071),
                Vec2f(0.3827, -0.9239),
                Vec2f(0.8315, 0.5556),
                Vec2f(0.8315, -0.5556),
                Vec2f(0.5556, -0.8315),
                Vec2f(0.9808, 0.1951),
                Vec2f(0.9239, -0.3827),
                Vec2f(0.7071, -0.7071),
                Vec2f(0.5556, 0.8315),
                Vec2f(0.9808, -0.1951),
                Vec2f(1.0000, 0.0000),
                Vec2f(0.1951, -0.9808)};

        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));

        float dw1 = setting_gradDownweightPerLevel;
        float dw2 = dw1 * dw1;

        int n3 = 0, n2 = 0, n4 = 0;
        for (int y4 = 0; y4 < h; y4 += (4 * pot))   // 4倍步长遍历
            for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
                int my3 = std::min((4 * pot), h - y4);
                int mx3 = std::min((4 * pot), w - x4);
                int bestIdx4 = -1;
                float bestVal4 = 0;
                Vec2f dir4 = directions[randomPattern[n2] & 0xF];
                for (int y3 = 0; y3 < my3; y3 += (2 * pot))   // 在4倍这个大块中2倍步长遍历
                    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                        int x34 = x3 + x4;
                        int y34 = y3 + y4;
                        int my2 = std::min((2 * pot), h - y34);
                        int mx2 = std::min((2 * pot), w - x34);
                        int bestIdx3 = -1;
                        float bestVal3 = 0;
                        Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                        for (int y2 = 0; y2 < my2; y2 += pot)   // 在2倍这个大块中1倍步长遍历
                            for (int x2 = 0; x2 < mx2; x2 += pot) {
                                int x234 = x2 + x34;
                                int y234 = y2 + y34;
                                int my1 = std::min(pot, h - y234);
                                int mx1 = std::min(pot, w - x234);
                                int bestIdx2 = -1;
                                float bestVal2 = 0;
                                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                                for (int y1 = 0; y1 < my1; y1 += 1)
                                    for (int x1 = 0; x1 < mx1; x1 += 1) {
                                        assert(x1 + x234 < w);
                                        assert(y1 + y234 < h);
                                        int idx = x1 + x234 + w * (y1 + y234);
                                        int xf = x1 + x234;
                                        int yf = y1 + y234;

                                        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue;


                                        float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];
                                        float pixelTH1 = pixelTH0 * dw1;
                                        float pixelTH2 = pixelTH1 * dw2;


                                        float ag0 = mapmax0[idx];
                                        if (ag0 > pixelTH0 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();  // 获取后两为x,y方向梯度值
                                            float dirNorm = fabsf((float) (ag0d.dot(dir2)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag0;

                                            if (dirNorm > bestVal2) {
                                                bestVal2 = dirNorm;
                                                bestIdx2 = idx;
                                                bestIdx3 = -2;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx3 == -2) continue;

                                        float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1];
                                        if (ag1 > pixelTH1 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir3)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag1;

                                            if (dirNorm > bestVal3) {
                                                bestVal3 = dirNorm;
                                                bestIdx3 = idx;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx4 == -2) continue;

                                        float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                                                            (int) (yf * 0.25f + 0.125) * w2];
                                        if (ag2 > pixelTH2 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir4)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag2;

                                            if (dirNorm > bestVal4) {
                                                bestVal4 = dirNorm;
                                                bestIdx4 = idx;
                                            }
                                        }
                                    }

                                if (bestIdx2 > 0) {
                                    map_out[bestIdx2] = 1;
                                    bestVal3 = 1e10;
                                    n2++;
                                }
                            }

                        if (bestIdx3 > 0) {
                            map_out[bestIdx3] = 2;
                            bestVal4 = 1e10;
                            n3++;
                        }
                    }

                if (bestIdx4 > 0) {
                    map_out[bestIdx4] = 4;
                    n4++;
                }
            }


        return Eigen::Vector3i(n2, n3, n4);
    }

    // for multi-fisheye image
    Eigen::Vector3i PixelSelector::selectforMF(const shared_ptr<FrameHessian> fh, int camNum, float *map_out, int pot, float thFactor)
    {
        Eigen::Vector3f const *const map0 = fh->vdI[camNum];

        float *mapmax0 = fh->vabsSquaredGrad[0][camNum];
        float *mapmax1 = fh->vabsSquaredGrad[1][camNum];
        float *mapmax2 = fh->vabsSquaredGrad[2][camNum];

        int w = fh->UMF->wPR[0];
        int w1 = fh->UMF->wPR[1];
        int w2 = fh->UMF->wPR[2];
        int h = fh->UMF->hPR[0];

        const Vec2f directions[16] = {
                Vec2f(0, 1.0000),
                Vec2f(0.3827, 0.9239),
                Vec2f(0.1951, 0.9808),
                Vec2f(0.9239, 0.3827),
                Vec2f(0.7071, 0.7071),
                Vec2f(0.3827, -0.9239),
                Vec2f(0.8315, 0.5556),
                Vec2f(0.8315, -0.5556),
                Vec2f(0.5556, -0.8315),
                Vec2f(0.9808, 0.1951),
                Vec2f(0.9239, -0.3827),
                Vec2f(0.7071, -0.7071),
                Vec2f(0.5556, 0.8315),
                Vec2f(0.9808, -0.1951),
                Vec2f(1.0000, 0.0000),
                Vec2f(0.1951, -0.9808)};

        memset(map_out, 0, w * h * sizeof(PixelSelectorStatus));

        float dw1 = setting_gradDownweightPerLevel;
        float dw2 = dw1 * dw1;

        int n3 = 0, n2 = 0, n4 = 0;
        for (int y4 = 0; y4 < h; y4 += (4 * pot))   // 4倍步长遍历
            for (int x4 = 0; x4 < w; x4 += (4 * pot)) {
                int my3 = std::min((4 * pot), h - y4);
                int mx3 = std::min((4 * pot), w - x4);
                int bestIdx4 = -1;
                float bestVal4 = 0;
                Vec2f dir4 = directions[randomPattern[n2] & 0xF];
                for (int y3 = 0; y3 < my3; y3 += (2 * pot))   // 在4倍这个大块中2倍步长遍历
                    for (int x3 = 0; x3 < mx3; x3 += (2 * pot)) {
                        int x34 = x3 + x4;
                        int y34 = y3 + y4;
                        int my2 = std::min((2 * pot), h - y34);
                        int mx2 = std::min((2 * pot), w - x34);
                        int bestIdx3 = -1;
                        float bestVal3 = 0;
                        Vec2f dir3 = directions[randomPattern[n2] & 0xF];
                        for (int y2 = 0; y2 < my2; y2 += pot)   // 在2倍这个大块中1倍步长遍历
                            for (int x2 = 0; x2 < mx2; x2 += pot) {
                                int x234 = x2 + x34;
                                int y234 = y2 + y34;
                                int my1 = std::min(pot, h - y234);
                                int mx1 = std::min(pot, w - x234);
                                int bestIdx2 = -1;
                                float bestVal2 = 0;
                                Vec2f dir2 = directions[randomPattern[n2] & 0xF];
                                for (int y1 = 0; y1 < my1; y1 += 1)
                                    for (int x1 = 0; x1 < mx1; x1 += 1) {
                                        assert(x1 + x234 < w);
                                        assert(y1 + y234 < h);
                                        int idx = x1 + x234 + w * (y1 + y234);
                                        int xf = x1 + x234;
                                        int yf = y1 + y234;

                                        if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4) continue;
                                        int ii = xf + yf * w;
                                        if(fh->vmask[0][camNum][ii] == 0) continue;


                                        float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];
                                        float pixelTH1 = pixelTH0 * dw1;
                                        float pixelTH2 = pixelTH1 * dw2;


                                        float ag0 = mapmax0[idx];
                                        if (ag0 > pixelTH0 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();  // 获取后两为x,y方向梯度值
                                            float dirNorm = fabsf((float) (ag0d.dot(dir2)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag0;

                                            if (dirNorm > bestVal2) {
                                                bestVal2 = dirNorm;
                                                bestIdx2 = idx;
                                                bestIdx3 = -2;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx3 == -2) continue;

                                        float ag1 = mapmax1[(int) (xf * 0.5f + 0.25f) + (int) (yf * 0.5f + 0.25f) * w1];
                                        if (ag1 > pixelTH1 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir3)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag1;

                                            if (dirNorm > bestVal3) {
                                                bestVal3 = dirNorm;
                                                bestIdx3 = idx;
                                                bestIdx4 = -2;
                                            }
                                        }
                                        if (bestIdx4 == -2) continue;

                                        float ag2 = mapmax2[(int) (xf * 0.25f + 0.125) +
                                                            (int) (yf * 0.25f + 0.125) * w2];
                                        if (ag2 > pixelTH2 * thFactor) {
                                            Vec2f ag0d = map0[idx].tail<2>();
                                            float dirNorm = fabsf((float) (ag0d.dot(dir4)));
                                            if (!setting_selectDirectionDistribution) dirNorm = ag2;

                                            if (dirNorm > bestVal4) {
                                                bestVal4 = dirNorm;
                                                bestIdx4 = idx;
                                            }
                                        }
                                    }

                                if (bestIdx2 > 0) {
                                    map_out[bestIdx2] = 1;
                                    bestVal3 = 1e10;
                                    n2++;
                                }
                            }

                        if (bestIdx3 > 0) {
                            map_out[bestIdx3] = 2;
                            bestVal4 = 1e10;
                            n3++;
                        }
                    }

                if (bestIdx4 > 0) {
                    map_out[bestIdx4] = 4;
                    n4++;
                }
            }


        return Eigen::Vector3i(n2, n3, n4);
    }

}