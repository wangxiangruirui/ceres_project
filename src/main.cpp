#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <vector>

// 用于保存弹道模型的六个待估参数
struct BallisticParams {
    double x0{0.0};   // 起始时刻 t0 时的 X 坐标（以像素为单位）
    double y0{0.0};   // 起始时刻 t0 时的 Y 坐标（以向上为正的像素坐标表示）
    double vx0{0.0};  // 起始时刻的水平速度 vx0（像素/秒）
    double vy0{0.0};  // 起始时刻的竖直速度 vy0（像素/秒，向上为正）
    double g{300.0};  // 有待拟合的重力加速度大小（像素/秒^2）
    double k{0.1};    // 阻力系数，描述速度指数衰减的强度（1/秒）
};

// Ceres 的残差项：给定一个观测点（obsX, obsY）与对应的 Δt，计算模型预测值与观测值的差
struct BallisticResidual {
    BallisticResidual(double dt, double obsX, double obsY)
        : dt_(dt), obsX_(obsX), obsY_(obsY) {}

    template <typename T>
    bool operator()(const T* const params, T* residuals) const {
        // params 指向长度为 6 的参数数组，依次为 [x0, y0, vx0, vy0, g, k]
        const T x0 = params[0];
        const T y0 = params[1];
        const T vx0 = params[2];
        const T vy0 = params[3];
        const T g = params[4];
        const T k = params[5];

        const T dt = T(dt_);
        // 关键公式：阻尼指数项 e^{-kΔt}
        const T expTerm = ceres::exp(-k * dt);
        const T oneMinusExp = T(1.0) - expTerm;

        // 根据题目提供的公式，计算模型预测的 x(t)、y(t)
        const T x = x0 + (vx0 / k) * oneMinusExp;
        const T y = y0 + ((vy0 + g / k) / k) * oneMinusExp - (g / k) * dt;

        // 残差 = 观测值 - 预测值。Ceres 会尝试让残差收敛到 0
        residuals[0] = T(obsX_) - x;
        residuals[1] = T(obsY_) - y;
        return true;
    }

    double dt_;    // 相对于起始时间 t0 的时间差 Δt
    double obsX_;  // 观测得到的 X 坐标（像素）
    double obsY_;  // 观测得到的 Y 坐标（向上为正的像素）
};

// 根据给定参数预测某个时间差 dt 对应的弹丸位置（返回坐标以屏幕向上为正）
cv::Point2d evaluateBallistic(const BallisticParams& params, double dt) {
    if (dt < 0.0) {
        // dt 小于 0 表示还未发射，直接返回初始位置
        return {params.x0, params.y0};
    }

    const double expTerm = std::exp(-params.k * dt);
    const double oneMinusExp = 1.0 - expTerm;

    const double x = params.x0 + (params.vx0 / params.k) * oneMinusExp;
    const double y = params.y0 + ((params.vy0 + params.g / params.k) / params.k) * oneMinusExp - (params.g / params.k) * dt;
    return {x, y};
}

// 使用 OpenCV 对每帧图像中的小球进行检测，返回圆心坐标；若未找到则返回空
std::optional<cv::Point2d> detectBallCenter(const cv::Mat& frame) {
    if (frame.empty()) {
        return std::nullopt;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(7, 7), 1.5);
    
    // 首先尝试使用霍夫圆检测：对圆形目标具有较好效果
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1.2,
                     static_cast<double>(gray.rows) / 16.0, 120, 20, 3, 0);

    if (!circles.empty()) {
        const cv::Vec3f& c = circles[0];
        return cv::Point2d(c[0], c[1]);
    }

    // 若霍夫变换失败，则改用阈值分割 + 轮廓检测作为备用方案
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv::THRESH_BINARY_INV, 11, 2);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double bestArea = 0.0;
    cv::Point2d bestCenter;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < 10.0) {
            continue;  // 面积太小，可能是噪声，忽略
        }

        cv::Moments m = cv::moments(contour);
        if (std::abs(m.m00) < 1e-5) {
            continue;  // 避免除以零
        }

        cv::Point2d center(m.m10 / m.m00, m.m01 / m.m00);
        if (area > bestArea) {
            bestArea = area;
            bestCenter = center;
        }
    }

    if (bestArea > 0.0) {
        return bestCenter;  // 返回面积最大的候选（假设为小球）
    }

    return std::nullopt;
}

// 将观测点与拟合曲线绘制成一张 PNG，直观对比模型效果
void drawTrajectoryPlot(const std::vector<double>& deltaTimes,
                        const std::vector<cv::Point2d>& centersUp,
                        const BallisticParams& params,
                        const std::string& outputPath) {
    if (deltaTimes.empty() || centersUp.empty()) {
        return;
    }

    constexpr int width = 1000;
    constexpr int height = 700;
    constexpr int margin = 80;

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double xMin = centersUp.front().x;
    double xMax = centersUp.front().x;
    double yMin = centersUp.front().y;
    double yMax = centersUp.front().y;

    for (const auto& pt : centersUp) {
        xMin = std::min(xMin, pt.x);
        xMax = std::max(xMax, pt.x);
        yMin = std::min(yMin, pt.y);
        yMax = std::max(yMax, pt.y);
    }

    const double tMax = deltaTimes.back();
    const int samples = 400;
    for (int i = 0; i <= samples; ++i) {
        double dt = tMax * i / samples;
        const cv::Point2d predicted = evaluateBallistic(params, dt);
        xMin = std::min(xMin, predicted.x);
        xMax = std::max(xMax, predicted.x);
        yMin = std::min(yMin, predicted.y);
        yMax = std::max(yMax, predicted.y);
    }

    const double xSpan = std::max(1e-3, xMax - xMin);
    const double ySpan = std::max(1e-3, yMax - yMin);

    auto projectPoint = [&](double x, double y) {
        double normX = (x - xMin) / xSpan;
        double normY = (y - yMin) / ySpan;

        int plotX = margin + static_cast<int>((width - 2 * margin) * normX);
        int plotY = height - margin - static_cast<int>((height - 2 * margin) * normY);
        return cv::Point(plotX, plotY);
    };

    cv::rectangle(canvas, cv::Point(margin, margin),
                  cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2);

    cv::putText(canvas, "Ballistic Trajectory (Y up)", cv::Point(margin, margin - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    std::ostringstream annotation;
    annotation << std::fixed << std::setprecision(3)
               << "g=" << params.g << " px/s^2, k=" << params.k << " 1/s";
    cv::putText(canvas, annotation.str(), cv::Point(margin, height - margin + 40),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    for (const auto& pt : centersUp) {
        cv::circle(canvas, projectPoint(pt.x, pt.y), 4, cv::Scalar(0, 180, 0), cv::FILLED);
    }

    std::vector<cv::Point> curvePoints;
    curvePoints.reserve(samples + 1);
    for (int i = 0; i <= samples; ++i) {
        double dt = tMax * i / samples;
        const cv::Point2d predicted = evaluateBallistic(params, dt);
        curvePoints.emplace_back(projectPoint(predicted.x, predicted.y));
    }
    cv::polylines(canvas, curvePoints, false, cv::Scalar(0, 0, 255), 2);

    cv::imwrite(outputPath, canvas);
}

// 将当前拟合参数转化成完整的方程字符串，方便打印给用户
std::string formatEquation(const BallisticParams& params) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Δt = t - t0\n";
    oss << "x(t) = " << params.x0 << " + (" << params.vx0 << " / " << params.k
        << ") * (1 - exp(-" << params.k << " * Δt))\n";
    oss << "y(t) = " << params.y0 << " + ((" << params.vy0 << " + " << params.g << " / "
        << params.k << ") / " << params.k << ") * (1 - exp(-" << params.k
        << " * Δt)) - (" << params.g << " / " << params.k << ") * Δt";
    return oss.str();
}

int main() {
    const std::string videoPath = "resources/video.mp4";
    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened()) {
        std::cerr << "无法打开视频文件: " << videoPath << std::endl;
        return 1;
    }

    const double fps = 60.0;  // 题目给定的视频帧率，用来把帧序号转换成时间

    // deltaTimes: 每个检测点对应的 Δt；centersImage: 图像坐标（向下为正）
    // centersUp: 转换为向上为正的坐标系；centersByFrame: 记录具体帧号对应的检测点
    std::vector<double> deltaTimes;
    std::vector<cv::Point2d> centersImage;
    std::vector<cv::Point2d> centersUp;
    std::unordered_map<int, cv::Point2d> centersByFrame;

    cv::Mat frame;
    int frameIndex = 0;
    double t0 = -1.0;  // 记录第一帧成功检测到小球时的时间，后续都相对它计算 Δt
    int frameWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    while (capture.read(frame)) {
        if (frameWidth <= 0) {
            frameWidth = frame.cols;
        }
        if (frameHeight <= 0) {
            frameHeight = frame.rows;
        }

        const double time = frameIndex / fps;
        if (auto centerOpt = detectBallCenter(frame)) {
            const cv::Point2d center = *centerOpt;
            if (t0 < 0.0) {
                t0 = time;
            }
            const double dt = time - t0;
            const double yUp = static_cast<double>(frameHeight) - center.y;

            deltaTimes.push_back(dt);
            centersImage.push_back(center);
            centersUp.emplace_back(center.x, yUp);
            centersByFrame.emplace(frameIndex, center);
        }

        ++frameIndex;
    }

    capture.release();

    if (centersUp.size() < 5) {
        std::cerr << "检测到的圆心数量不足以拟合弹道模型。" << std::endl;
        return 1;
    }

    if (frameHeight <= 0) {
        double maxY = centersImage.front().y;
        for (const auto& pt : centersImage) {
            maxY = std::max(maxY, pt.y);
        }
        frameHeight = static_cast<int>(std::ceil(maxY)) + 1;
    }

    // 为 Ceres 设置一个不错的初始猜测，有助于收敛
    BallisticParams params;
    params.x0 = centersUp.front().x;
    params.y0 = centersUp.front().y;
    params.g = 300.0;
    params.k = 0.1;

    if (deltaTimes.back() > 0.0) {
        const double totalTime = deltaTimes.back();
        const cv::Point2d first = centersUp.front();
        const cv::Point2d last = centersUp.back();
    params.vx0 = (last.x - first.x) / totalTime;
    params.vy0 = (last.y - first.y) / totalTime;
    } else {
        params.vx0 = 0.0;
        params.vy0 = 0.0;
    }

    // Ceres 要求参数以 double 数组形式传入，并可在优化过程中原地修改
    double parameterBlock[6] = {params.x0, params.y0, params.vx0, params.vy0, params.g, params.k};

    ceres::Problem problem;
    for (size_t i = 0; i < deltaTimes.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<BallisticResidual, 2, 6>(
                new BallisticResidual(deltaTimes[i], centersUp[i].x, centersUp[i].y)),
            nullptr,
            parameterBlock);
    }

    // 约束 g 和 k 的取值范围，符合题目要求
    problem.SetParameterLowerBound(parameterBlock, 4, 100.0);
    problem.SetParameterUpperBound(parameterBlock, 4, 1000.0);
    problem.SetParameterLowerBound(parameterBlock, 5, 0.01);
    problem.SetParameterUpperBound(parameterBlock, 5, 1.0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;   // 对于中小规模问题，稠密 QR 求解器足够稳定
    options.max_num_iterations = 200;                // 上限迭代次数，可根据需要调整
    options.minimizer_progress_to_stdout = false;    // 设为 true 可以查看迭代日志

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);


    // 读取求解后的参数
    BallisticParams fitted;
    fitted.x0 = parameterBlock[0];
    fitted.y0 = parameterBlock[1];
    fitted.vx0 = parameterBlock[2];
    fitted.vy0 = parameterBlock[3];
    fitted.g = parameterBlock[4];
    fitted.k = parameterBlock[5];

    // 计算并打印平均残差值
    size_t num_residuals = deltaTimes.size() * 2;  // 每个观测点有 x 和 y 两个残差
    double sum_squared_residuals = 2.0 * summary.final_cost;  // final_cost = 0.5 * sum(residuals^2)
    double mean_residual = std::sqrt(sum_squared_residuals / num_residuals);
    std::cout << "平均残差值: " << std::fixed << std::setprecision(6) << mean_residual << std::endl;

    std::cout << "拟合成功的参数 (单位 px/s, px/s^2):" << std::endl;
    std::cout << std::fixed << std::setprecision(6)
              << "x0=" << fitted.x0 << ", y0=" << fitted.y0
              << ", vx0=" << fitted.vx0 << ", vy0=" << fitted.vy0
              << ", g=" << fitted.g << ", k=" << fitted.k << std::endl;
    std::cout << "拟合方程:" << std::endl;
    std::cout << formatEquation(fitted) << std::endl;

    // 确保 results 目录存在，用于保存图片和视频
    const std::filesystem::path resultsDir("results");
    std::error_code ec;
    std::filesystem::create_directories(resultsDir, ec);

    const std::filesystem::path plotPath = resultsDir / "trajectory_plot.png";
    drawTrajectoryPlot(deltaTimes, centersUp, fitted, plotPath.string());

    // 再次读取原视频，用于叠加拟合曲线并写出新视频
    cv::VideoCapture playback(videoPath);

    int width = static_cast<int>(playback.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(playback.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (width <= 0) {
        width = frameWidth;
    }
    if (height <= 0) {
        height = frameHeight;
    }

    int fourcc = static_cast<int>(playback.get(cv::CAP_PROP_FOURCC));
    if (fourcc == 0) {
        fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    }

    const std::filesystem::path videoOutputPath = resultsDir / "video_with_fit.mp4";
    cv::VideoWriter writer(videoOutputPath.string(), fourcc, fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        writer.open(videoOutputPath.string(), fourcc, fps, cv::Size(width, height));
    }
    if (!writer.isOpened()) {
        std::cerr << "无法创建输出视频文件。" << std::endl;
        return 1;
    }

    std::vector<cv::Point> fittedPath;
    const double maxDeltaT = deltaTimes.back();
    if (maxDeltaT > 0.0) {
        const int samples = 600;
        fittedPath.reserve(samples + 1);
        for (int i = 0; i <= samples; ++i) {
            double dt = maxDeltaT * i / samples;
            const cv::Point2d predictedUp = evaluateBallistic(fitted, dt);
            const double yImage = static_cast<double>(height) - predictedUp.y;
            cv::Point pt(static_cast<int>(std::round(predictedUp.x)),
                         static_cast<int>(std::round(yImage)));
            if (pt.x >= 0 && pt.x < width && pt.y >= 0 && pt.y < height) {
                fittedPath.push_back(pt);
            }
        }
    }

    frameIndex = 0;
    while (playback.read(frame)) {
        const double time = frameIndex / fps;
        double dt = (t0 >= 0.0) ? (time - t0) : -1.0;

        if (!fittedPath.empty()) {
            cv::polylines(frame, fittedPath, false, cv::Scalar(0, 0, 255), 2);
        }

        if (dt >= 0.0) {
            const cv::Point2d predictedUp = evaluateBallistic(fitted, dt);
            const double yImage = static_cast<double>(height) - predictedUp.y;
            cv::Point predictedPt(static_cast<int>(std::round(predictedUp.x)),
                                  static_cast<int>(std::round(yImage)));
            if (predictedPt.x >= 0 && predictedPt.x < frame.cols &&
                predictedPt.y >= 0 && predictedPt.y < frame.rows) {
                cv::circle(frame, predictedPt, 4, cv::Scalar(0, 0, 255), cv::FILLED);
            }
        }

        auto it = centersByFrame.find(frameIndex);
        if (it != centersByFrame.end()) {
            cv::circle(frame, it->second, 4, cv::Scalar(0, 255, 0), cv::FILLED);
        }

        if (frameIndex == 0) {
            std::ostringstream text;
            text << std::fixed << std::setprecision(3)
                 << "vx0=" << fitted.vx0 << ", vy0=" << fitted.vy0;
            cv::putText(frame, text.str(), cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            std::ostringstream text2;
            text2 << std::fixed << std::setprecision(3)
                  << "g=" << fitted.g << ", k=" << fitted.k;
            cv::putText(frame, text2.str(), cv::Point(20, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }

        writer.write(frame);
        ++frameIndex;
    }

    playback.release();
    writer.release();

    return 0;
}
