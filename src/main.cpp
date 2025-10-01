#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
#include <algorithm>
struct BallisticParams { double x0, y0, vx0, vy0, g, k; };
struct Detection { cv::Point2d center; double radius; bool fromHough; };
struct Residual {
	Residual(double dt_, double ox_, double oy_) : dt(dt_), ox(ox_), oy(oy_) {}
	template <class T>
	bool operator()(const T* p, T* r) const {
		T e = ceres::exp(-p[5] * T(dt));
		T o = T(1) - e;
		T x = p[0] + (p[2] / p[5]) * o;
		T y = p[1] + ((p[3] + p[4] / p[5]) / p[5]) * o - (p[4] / p[5]) * T(dt);
		r[0] = T(ox) - x;
		r[1] = T(oy) - y;
		return true;
	}
	double dt, ox, oy;
};

static std::optional<Detection> detectByContours(const cv::Mat& image, std::optional<double> expectedRadius = std::nullopt) {
	if (image.empty()) return std::nullopt;
	cv::Mat gray;
	if (image.channels() == 1) gray = image;
	else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.0);
	int minDim = std::min(gray.rows, gray.cols);
	int block = minDim >= 11 ? 11 : (minDim | 1);
	if (block < 3) block = 3;
	if ((block & 1) == 0) ++block;
	cv::Mat binary;
	cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
						  cv::THRESH_BINARY_INV, block, 2);
	int kernelSize = minDim >= 5 ? 5 : 3;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
	cv::morphologyEx(binary, binary, cv::MORPH_OPEN, element);
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	double minArea = std::max(5.0, 0.0005 * static_cast<double>(gray.rows) * gray.cols);
	double maxArea = 0.4 * static_cast<double>(gray.rows) * gray.cols;
	double bestScore = 0.0;
	Detection best{};
	best.fromHough = false;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area < minArea || area > maxArea) continue;
		double perimeter = cv::arcLength(contour, true);
		if (perimeter < 1e-5) continue;
		double circularity = 4.0 * CV_PI * area / (perimeter * perimeter + 1e-6);
		if (circularity < 0.6) continue;
		cv::Point2f centerF; float radiusF;
		cv::minEnclosingCircle(contour, centerF, radiusF);
		if (radiusF < 3.0f || radiusF > 80.0f) continue;
		double radiusWeight = 1.0;
		if (expectedRadius) radiusWeight = 1.0 / (1.0 + std::abs(radiusF - *expectedRadius));
		double score = circularity * area * radiusWeight;
		if (score > bestScore) {
			bestScore = score;
			best.center = cv::Point2d(centerF.x, centerF.y);
			best.radius = radiusF;
			best.fromHough = false;
		}
	}
	if (bestScore > 0.0) return best;
	return std::nullopt;
}

static std::optional<Detection> detectByHough(const cv::Mat& image, std::optional<double> expectedRadius = std::nullopt) {
	if (image.empty()) return std::nullopt;
	cv::Mat gray;
	if (image.channels() == 1) gray = image;
	else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(7, 7), 1.5);
	std::vector<cv::Vec3f> circles;
	int minRadius = 3;
	int maxRadius = std::min({ std::max(gray.rows, gray.cols) / 2, 120 });
	if (expectedRadius) {
		minRadius = std::max(3, (int)std::round(*expectedRadius * 0.7));
		maxRadius = std::max(minRadius + 2, (int)std::round(*expectedRadius * 1.4));
	}
	cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1.2,
					 static_cast<double>(gray.rows) / 16.0, 120, 18, minRadius, maxRadius);
	if (!circles.empty()) {
		Detection d;
		d.center = cv::Point2d(circles[0][0], circles[0][1]);
		d.radius = circles[0][2];
		d.fromHough = true;
		return d;
	}
	return std::nullopt;
}


static std::optional<Detection> detectBallCenter(const cv::Mat& frame, const cv::Rect* roi = nullptr, bool allowHough = true, std::optional<double> expectedRadius = std::nullopt) {
	const cv::Mat region = roi ? frame(*roi) : frame;
	std::optional<Detection> result;
	if (allowHough) result = detectByHough(region, expectedRadius);
	if (!result) result = detectByContours(region, expectedRadius);
	if (!result && allowHough && !expectedRadius) result = detectByHough(region, expectedRadius);
	if (result && roi) {
		result->center.x += roi->x;
		result->center.y += roi->y;
	}
	return result;
}

static std::string eq(const BallisticParams& p) {
	std::ostringstream o; o.setf(std::ios::fixed); o << std::setprecision(6)
		<< "Δt = t - t0\n"
		<< "x(t) = " << p.x0 << " + (" << p.vx0 << " / " << p.k << ") * (1 - exp(-" << p.k << " * Δt))\n"
		<< "y(t) = " << p.y0 << " + ((" << p.vy0 << " + " << p.g << " / " << p.k << ") / " << p.k << ") * (1 - exp(-" << p.k << " * Δt)) - (" << p.g << " / " << p.k << ") * Δt";
	return o.str();
}

int main() {
	cv::VideoCapture cap("resources/video.mp4");
	if (!cap.isOpened()) return 1;
	const double fps = 60.0;
	int frameCount = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
	if (frameCount < 0) frameCount = 0;
	std::vector<double> ts; ts.reserve(frameCount);
	std::vector<cv::Point2d> pts; pts.reserve(frameCount);
	cv::Mat f;
	int idx = 0;
	double t0 = -1;
	int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	if (H <= 0) H = 0;
	std::optional<Detection> last;
	while (cap.read(f)) {
		double t = idx / fps;
		std::optional<Detection> detect;
		cv::Rect roi;
		if (last && f.cols > 0 && f.rows > 0) {
			double searchRadius = std::clamp(last->radius * 2.5, 20.0, 200.0);
			int r = std::max(10, (int)std::round(searchRadius));
			int maxX = std::max(f.cols - 1, 0);
			int maxY = std::max(f.rows - 1, 0);
			int cx = (int)std::round(last->center.x);
			int cy = (int)std::round(last->center.y);
			int x = std::clamp(cx - r, 0, maxX);
			int y = std::clamp(cy - r, 0, maxY);
			int w = std::min(r * 2, f.cols - x);
			int h = std::min(r * 2, f.rows - y);
			if (w > 0 && h > 0) {
				roi = cv::Rect(x, y, w, h);
				detect = detectBallCenter(f, &roi, true, last->radius);
			}
		}
		std::optional<double> expectedRadius;
		if (last) expectedRadius = last->radius;
		if (!detect) detect = detectBallCenter(f, nullptr, true, expectedRadius);
		if (detect && last) {
			double dist = cv::norm(detect->center - last->center);
			double maxJump = std::max(last->radius * 4.0, 80.0);
			if (dist > maxJump) {
				detect = detectBallCenter(f, nullptr, true, last->radius);
			}
		}
		if (detect) {
			last = detect;
			if (t0 < 0) t0 = t;
			double dt = t - t0;
			double yup = (H ? H : f.rows) - detect->center.y;
			ts.push_back(dt);
			pts.emplace_back(detect->center.x, yup);
		} else {
			last.reset();
		}
		++idx;
	}
	cap.release();
	if (pts.empty()) return 0;
	BallisticParams p;
	p.x0 = pts.front().x;
	p.y0 = pts.front().y;
	p.g = 300;
	p.k = 0.1;
	if (ts.back() > 0) {
		double T = ts.back();
		auto a = pts.front();
		auto b = pts.back();
		p.vx0 = (b.x - a.x) / T;
		p.vy0 = (b.y - a.y) / T;
	} else {
		p.vx0 = p.vy0 = 0;
	}
	double param[6] = { p.x0, p.y0, p.vx0, p.vy0, p.g, p.k };
	ceres::Problem prob;
	for (size_t i = 0; i < ts.size(); ++i) {
		prob.AddResidualBlock(
			new ceres::AutoDiffCostFunction<Residual, 2, 6>(
				new Residual(ts[i], pts[i].x, pts[i].y)),
			nullptr,
			param);
	}
	prob.SetParameterLowerBound(param, 4, 100);
	prob.SetParameterUpperBound(param, 4, 1000);
	prob.SetParameterLowerBound(param, 5, 0.01);
	prob.SetParameterUpperBound(param, 5, 1.0);
	ceres::Solver::Options opt;
	opt.linear_solver_type = ceres::DENSE_QR;
	opt.max_num_iterations = 200;
	opt.num_threads = std::thread::hardware_concurrency();
	opt.minimizer_progress_to_stdout = false;
	ceres::Solver::Summary sum;
	ceres::Solve(opt, &prob, &sum);
	BallisticParams r{ param[0], param[1], param[2], param[3], param[4], param[5] };
	size_t n = ts.size() * 2;
	double mean = std::sqrt(2 * sum.final_cost / n);
	std::cout.setf(std::ios::fixed);
	std::cout << std::setprecision(6)
			  << "mean_residual=" << mean << "\n"
			  << "x0=" << r.x0 << " y0=" << r.y0
			  << " vx0=" << r.vx0 << " vy0=" << r.vy0
			  << " g=" << r.g << " k=" << r.k << "\n"
			  << eq(r) << "\n";
	return 0;
}
