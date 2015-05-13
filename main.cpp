#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

RNG rng(12345);


void SimplestCB(Mat& in, Mat& out, float percent) {
    assert(in.channels() == 3);
    assert(percent > 0 && percent < 100);
 
    float half_percent = percent / 200.0f;
 
    vector<Mat> tmpsplit; split(in,tmpsplit);
    for(int i=0;i<3;i++) {
	    //find the low and high precentile values (based on the input percentile)
	    Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
	    cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	    int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
	    int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));

	    //saturate below the low percentile and above the high percentile
	    tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
	    tmpsplit[i].setTo(highval,tmpsplit[i] > highval);

	    //scale the channel
	    normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
    }
    merge(tmpsplit,out);
}


Mat getSkin( const Mat & input )
{
	int Y_MIN = 0, Y_MAX = 255, Cr_MIN = 133, Cr_MAX = 173, Cb_MIN = 77, Cb_MAX = 127;
	Mat skin;
	cvtColor(input, skin, CV_BGR2YCrCb);
	cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
	return skin;
}

// Aplica una mascara al frame utilizando grayFrame como mascara
// Cualquier pixel "negro" en grayFrame sera transformado en un pixel
// negro en frame, el resultado se guardara en result
void applyMask( const Mat & grayFrame, const Mat & frame, Mat & result, int maskVal = 0)
{
	result = frame.clone();
	for( int y  =0; y < grayFrame.rows; y++ )
		for( int x = 0; x < grayFrame.cols; x++ )
		{
			uchar val = grayFrame.at<uchar>( Point(x,y) );
			if( val != maskVal ) continue;

			result.at<Vec3b>(Point(x,y)).val[0] = 0;
			result.at<Vec3b>(Point(x,y)).val[1] = 0;
			result.at<Vec3b>(Point(x,y)).val[2] = 0;
		}

}
// Funcion para encontrar la sonrisa dentro de la imagen usando una imagen en escala de grises
Rect EncontrarSonrisa( const Mat & frame )
{

	uchar minPossibleValue = 50;
	// Obtener el valor mas "claro" (posible punto de la sonrisa)
	uchar min = 0;
	int minY = 0, minX = 0;
	for( int y = 0; y < frame.rows; y++)
	{
		for( int x = 0; x < frame.cols; x++ )
		{
			// obtener el pixel
			uchar pix = frame.at<uchar>( Point(x,y) );
			if( pix > min && pix >= minPossibleValue )
			{
				minY = y;
				minX = x;
				min = pix;
			}
		}
	}

	// Falta incluir un BFS aquí para elegir el rectangulo completo
	// que conforma la sonirsa

	return Rect( Point(minX-40,minY-40), Point(minX+40,minY+40));
}

main(int argc, const char* argv[])
{

	Mat frame, grayFrame, result, skin;
	VideoCapture  capture;

	//Se inicia la camara
	capture.open(0);
	if( !capture.isOpened() )
	{
		std::cout << "no se encontro la camara" << std::endl;
		return -1;
	}

	//Crea la ventana
	cvNamedWindow("Proyecto", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Skin", CV_WINDOW_AUTOSIZE);
	//Captura de la camara
	while(1 )
	{
		capture >> frame ;
		// Algoritmo de balance de colores, no necesariamente funciona ):
		//SimplestCB(frame,frame,1);

		skin = getSkin(frame);
		//Convierte el frame a escala de grises
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		blur(grayFrame,grayFrame, Size(5,5));

		//Canny contours algorithm
		Mat canny_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		//Canny( grayFrame, canny_output, 0,0, 3);		
		//findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0) );
		//Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3);
		//for( int i = 0; i < contours.size(); i++ )
		//{
		//Scalar color = Scalar( rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
		//drawContours( grayFrame, contours, 3, color, 3, CV_AA, hierarchy, 0, Point() );
		//}

		//Histogram Equalization
		equalizeHist(grayFrame, grayFrame);

		//erociona y dilatacion
		int size = 6;
		Mat element = getStructuringElement(MORPH_CROSS, Size(2*size+1,2*size+1), Point(size,size) );
		erode(grayFrame, grayFrame, element );
		dilate(grayFrame, grayFrame, element );

		dilate(skin, skin, element);

		inRange(grayFrame, Scalar(200), Scalar(255), grayFrame);
		applyMask(grayFrame, frame, result);
		applyMask(skin, result, result, 255);
		cvtColor(result, result, CV_BGR2GRAY);

		//Para dibujar el rectangulo 
		Rect rect = EncontrarSonrisa( result );
		rectangle( frame, rect , cvScalar(0,0,255));

		//Muestra el frame
		imshow("Proyecto", grayFrame);
		imshow("Original", frame);
		imshow("Result", result);
		imshow("Skin", skin);
		cvWaitKey(1);
	}
	return 0;
}
