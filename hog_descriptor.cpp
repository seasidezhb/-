//http://gori-naru.blogspot.jp/2012/11/hog.html

#include <iostream>
#include <vector>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;


// ヒストグラムのビン数
#define N_BIN 9
// 何度ずつに分けて投票するか（分解能）
#define THETA (180 / N_BIN)

// 積分画像生成
vector<Mat> calculateIntegralHOG(const Mat& image) {
	// X, Y方向に微分
	Mat xsobel, ysobel;
	Sobel(image, xsobel, CV_32F, 1, 0);
	Sobel(image, ysobel, CV_32F, 0, 1);

	// 角度別の画像を生成しておく
	vector<Mat> bins(N_BIN);
	for (int i = 0; i < N_BIN; i++)
		bins[i] = Mat::zeros(image.size(), CV_32F);

	// X, Y微分画像を勾配方向と強度に変換
	Mat Imag, Iang;
	cartToPolar(xsobel, ysobel, Imag, Iang, true);
	// 勾配方向を[0, 180)にする
	add(Iang, Scalar(180), Iang, Iang < 0);
	add(Iang, Scalar(-180), Iang, Iang >= 180);
	// 勾配方向を[0, 1, ..., 8]にする準備（まだfloat）
	Iang /= THETA;

	// 勾配方向を強度で重みをつけて、角度別に投票する
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int ind = Iang.at<float>(y, x);
			bins[ind].at<float>(y, x) += Imag.at<float>(y, x);
		}
	}

	// 角度別に積分画像生成
	vector<Mat> integrals(N_BIN);
	for (int i = 0; i < N_BIN; i++) {
		// 積分画像をつくる、OpenCVの関数がある
		integral(bins[i], integrals[i]);
	}

	return integrals;
}

// ある矩形領域の勾配ヒストグラムを求める
// ここでいう矩形はHOG特徴量のセルに該当
void calculateHOGInCell(Mat& hogCell, Rect roi, const vector<Mat>& integrals) {
	int x0 = roi.x, y0 = roi.y;
	int x1 = x0 + roi.width, y1 = y0 + roi.height;
	for (int i = 0; i < N_BIN; i++) {
		Mat integral = integrals[i];
		float a = integral.at<double>(y0, x0);
		float b = integral.at<double>(y1, x1);
		float c = integral.at<double>(y0, x1);
		float d = integral.at<double>(y1, x0);
		hogCell.at<float>(0, i) = (a + b) - (c + d);
	}
}
// セルの大きさ（ピクセル数）
#define CELL_SIZE 20
// ブロックの大きさ（セル数）奇数
#define BLOCK_SIZE 3
// ブロックの大きさの半分（ピクセル数）
#define R (CELL_SIZE*(BLOCK_SIZE)*0.5)

// HOG特徴量を計算する
// pt: ブロックの中心点
Mat getHOG(Point pt, const vector<Mat>& integrals) {
	// ブロックが画像からはみ出していないか確認
	if (pt.x - R < 0 ||
		pt.y - R < 0 ||
		pt.x + R >= integrals[0].cols ||
		pt.y + R >= integrals[0].rows
		) {
		return Mat();
	}

	// 与点を中心としたブロックで、
	// セルごとに勾配ヒストグラムを求めて連結
	Mat hist(Size(N_BIN*BLOCK_SIZE*BLOCK_SIZE, 1), CV_32F);
	Point tl(0, pt.y - R);
	int c = 0;
	for (int i = 0; i < BLOCK_SIZE; i++) {
		tl.x = pt.x - R;
		for (int j = 0; j < BLOCK_SIZE; j++) {
			calculateHOGInCell(hist.colRange(c, c + N_BIN),
				Rect(tl, tl + Point(CELL_SIZE, CELL_SIZE)),
				integrals);
			tl.x += CELL_SIZE;
			c += N_BIN;
		}
		tl.y += CELL_SIZE;
	}
	// L2ノルムで正規化
	normalize(hist, hist, 1, 0, NORM_L2);
	return hist;
}

int main() {
	// 画像をグレイスケールで読み込む
	string fileName = "4.jpeg";
	Mat originalImage = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);

	// 積分画像生成
	vector<Mat> integrals = calculateIntegralHOG(originalImage);
	// ある点(x, y)のHOG特徴量を求めるには
	// Mat hist = getHOG(Point(x, y), integrals);
	// とする。histはSize(81, 1) CV_32FのMat


	/* ****************** *
	* 以下、表示のための処理
	* ****************** */

	// 表示用画像を用意（半分の輝度に）
	Mat image = originalImage.clone();
	image *= 0.5;

	// 格子点でHOG計算
	Mat meanHOGInBlock(Size(N_BIN, 1), CV_32F);
	for (int y = CELL_SIZE / 2; y < image.rows; y += CELL_SIZE) {
		for (int x = CELL_SIZE / 2; x < image.cols; x += CELL_SIZE) {
			// (x, y)でのHOGを取得
			Mat hist = getHOG(Point(x, y), integrals);
			// ブロックが画像からはみ出ていたら continue
			if (hist.empty()) continue;

			// ブロックごとに勾配方向ヒストグラム生成
			meanHOGInBlock = Scalar(0);
			for (int i = 0; i < N_BIN; i++) {
				for (int j = 0; j < BLOCK_SIZE*BLOCK_SIZE; j++) {
					meanHOGInBlock.at<float>(0, i) += hist.at<float>(0, i + j*N_BIN);
				}
			}
			// L2ノルムで正規化（強い方向が強調される）
			normalize(meanHOGInBlock, meanHOGInBlock, 1, 0, CV_L2);

			// 角度ごとに線を描画
			Point center(x, y);
			for (int i = 0; i < N_BIN; i++) {
				double theta = (i * THETA + 90.0) * CV_PI / 180.0;
				Point rd(CELL_SIZE*0.5*cos(theta), CELL_SIZE*0.5*sin(theta));
				Point rp = center - rd;
				Point lp = center - -rd;
				line(image, rp, lp, Scalar(255 * meanHOGInBlock.at<float>(0, i), 255, 255));
			}
		}
	}

	// 表示
	imshow("out", image);
	imwrite("hog.jpg", image);
	waitKey(0);

	return 0;
}
