#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>

using namespace cv;

// Aplica una mascara al frame utilizando grayFrame como mascara
// Cualquier pixel "negro" en grayFrame sera transformado en un pixel
// negro en frame, el resultado se guardara en result
void applyMask( const Mat & grayFrame, const Mat & frame, Mat & result)
{
	result = frame.clone();
	for( int y  =0; y < grayFrame.rows; y++ )
	for( int x = 0; x < grayFrame.cols; x++ )
	{
		uchar val = grayFrame.at<uchar>( Point(x,y) );
		if( val != 0 ) continue;
		
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

	Mat frame, grayFrame, result;
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
	//Captura de la camara
	while(1 )
	{
	capture >> frame ;
	//Convierte el frame a escala de grises
	cvtColor(frame, grayFrame, CV_BGR2GRAY);
	//Histogram Equalization
	equalizeHist(grayFrame, grayFrame);

	//Median blur para disminuir el ruido (no nos intresa encontrar un punto, si no una zona)
	//medianBlur(frame, frame, 5);
	
	inRange(grayFrame, Scalar(200), Scalar(255), grayFrame);
	applyMask(grayFrame, frame, result);
	cvtColor(result, result, CV_BGR2GRAY);

	//Para dibujar el rectangulo 
	Rect rect = EncontrarSonrisa( result );
	rectangle( frame, rect , cvScalar(0,0,255));

	//Muestra el frame
	imshow( "Proyecto", grayFrame);
	imshow( "Original", frame);
	imshow("Result", result);

	cvWaitKey(1);
	}
	return 0;
}
