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

Mat getSkin( const Mat & input )
{
	int Y_MIN = 0, Y_MAX = 255, Cr_MIN = 133, Cr_MAX = 173, Cb_MIN = 77, Cb_MAX = 127;
	Mat skin;
	cvtColor(input, skin, CV_BGR2YCrCb);
	cv::inRange(skin,cv::Scalar(Y_MIN,Cr_MIN,Cb_MIN),cv::Scalar(Y_MAX,Cr_MAX,Cb_MAX),skin);
	return skin;
}


// Aplica una mascara al frame utilizando grayFrame como mascara
// Cualquier pixel "negro" (o el valor especificado en maskVal)
// en grayFrame sera transformado en un pixel negro en frame, 
// el resultado se guardara en result
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
			if( pix > min )
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

void dibujarInterface( Mat & frame )
{
	int rows = frame.rows / 10; // 10 porciento filas
	rectangle( frame, Point(0,0), Point(frame.cols, rows), Scalar(0,0,0),CV_FILLED );
	rectangle( frame, Point(0,frame.rows-rows), Point(frame.cols,frame.rows), Scalar(255,255,255),CV_FILLED );
	putText( frame, "Imagen con sonrisa", Point(frame.cols/4, frame.rows-rows/2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,0) );

	Mat logo = imread("smile.jpg", CV_LOAD_IMAGE_COLOR);
	int logoW = rows - rows/4;
	int logoX = frame.cols - logoW - 4; // end of the frame - logo width - 4 margin
	int logoY = 4; 
	resize( logo, logo, Size( logoW, logoW ) ); // resize del logo
	logo.copyTo( frame.rowRange( logoY, logoY+logoW ).colRange( logoX , logoX+logoW ) );
}

main(int argc, const char* argv[])
{

	Mat frame, uiframe, grayFrame, result, skin, frame1, frame2, frame3,frame4, frame5;
	VideoCapture  capture;
	int lowerBound = 200;
	int upperBound = 255;
	
	//Se inicia la camara
	
	capture.open(0);
	if( !capture.isOpened() )
	{
		std::cout << "no se encontro la camara" << std::endl;
		return -1;
	}

	//Crea la ventana
	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("Rango inferior", "Result", &lowerBound, 255 );
	cvCreateTrackbar("Rango superior", "Result", &upperBound, 255 );

	//Captura de la camara
	while(1)
	{
		for(int i = 0; i < 5; i++ ) capture >> frame;
		capture >> frame;
		uiframe = frame.clone();

		// Dibuja las barras inferiores y superiores
		dibujarInterface(uiframe);
		imshow("Result", uiframe );
		cvWaitKey();

		//Obtiene la mascara de la piel
		skin = getSkin(frame);

		//Convierte el frame a escala de grises
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		//Histogram Equalization
		equalizeHist(grayFrame, frame1);

		blur(frame1,frame2, Size(5,5));

		//erociona y dilatacion
		int size = 6;
		Mat element = getStructuringElement(MORPH_CROSS, Size(2*size+1,2*size+1), Point(size,size) );
		erode(frame2, frame3, element );
		dilate(frame3, frame4, element );

		inRange(frame4, Scalar(lowerBound), Scalar(upperBound), frame5);

		applyMask(frame5, frame, result);
		applyMask(skin, result, result, 255);
		cvtColor( result , result, CV_BGR2GRAY);
		//Para dibujar el rectangulo 
		Rect rect = EncontrarSonrisa( result );
		rectangle( frame, rect , cvScalar(0,0,255));

		// Dibuja las barras inferiores y superiores
		dibujarInterface(frame);

		//Muestra el frame
		imshow("Result", frame);
		cvWaitKey();
	}
	return 0;
}
