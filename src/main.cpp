#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <optional>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <thread>
struct BallisticParams { double x0, y0, vx0, vy0, g, k; };
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

static std::optional<cv::Point2d> detectBallCenter(const cv::Mat& frame) {
	if (frame.empty()) return std::nullopt;
	cv::Mat gray; cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, gray, cv::Size(7, 7), 1.5);
	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1.2,
					 static_cast<double>(gray.rows) / 16.0, 120, 20, 3, 0);
	if (!circles.empty()) return cv::Point2d(circles[0][0], circles[0][1]);
	cv::Mat binary;
	cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
						  cv::THRESH_BINARY_INV, 11, 2);
	cv::morphologyEx(binary, binary, cv::MORPH_OPEN,
					 cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	double bestArea = 0.0; cv::Point2d bestCenter;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area < 10.0) continue;
		cv::Moments m = cv::moments(contour);
		if (std::abs(m.m00) < 1e-5) continue;
		cv::Point2d center(m.m10 / m.m00, m.m01 / m.m00);
		if (area > bestArea) { bestArea = area; bestCenter = center; }
	}
	if (bestArea > 0.0) return bestCenter;
	return std::nullopt;
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
	if (frameCount < 0) frameCount = 0; // some codecs may not provide
	std::vector<double> ts; ts.reserve(frameCount);
	std::vector<cv::Point2d> pts; pts.reserve(frameCount);
	cv::Mat f;
	int idx = 0;
	double t0 = -1;
	int H = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT); if (H <= 0) H = 0;
	while (cap.read(f)) {
		double t = idx / fps;
		auto c = detectBallCenter(f);
		if (c) {
			if (t0 < 0) t0 = t;
			double dt = t - t0;
			double yup = (H ? H : f.rows) - c->y;
			ts.push_back(dt);
			pts.emplace_back(c->x, yup);
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
	ceres::Solver::Summary sum; ceres::Solve(opt, &prob, &sum);
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
