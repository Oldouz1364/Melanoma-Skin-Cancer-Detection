package com.example.moletest;

import android.graphics.Bitmap;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import org.opencv.core.Mat;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import java.lang.Math;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        OpenCVLoader.initDebug();
    }

    public void displayToast(View v){

        Mat img = null;

        try {
            img = Utils.loadResource(getApplicationContext(),R.drawable.a);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Mat imggray = new Mat();
        Mat img_thresh = new Mat();
        Imgproc.cvtColor(img, imggray, Imgproc.COLOR_BGR2GRAY); //cvtColor(Mat src, Mat dst, int code, int dstCn)  Converts an image from one color space to another.

        //Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGRA);
        //Mat img_result = img.clone();
        //Imgproc.Canny(img, img_result, 80, 90);

        Imgproc.threshold(imggray,img_thresh,0,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);
        // threshold​(Mat src, Mat dst, double thresh, double maxval, int type) Applies a fixed-level threshold to each array element.

        Mat dilate_element = Imgproc.getStructuringElement( Imgproc.MORPH_RECT,
                new Size(3,3));
        Mat img_dilate = new Mat();
        Imgproc.dilate(img_thresh,img_dilate, dilate_element);
        // dilate​(Mat src, Mat dst, Mat kernel, Point anchor, int iterations, int borderType, Scalar borderValue)

        // Calculation of centroid
        // calculate moments of binary image
        //Moments M =Imgproc.moments(img_dilate);
        //Point centroid = new Point(M.m10 / (M.m00 + 1e-5), M.m01 / (M.m00 + 1e-5));//add 1e-5 to avoid division by zero

        //Mat img_fill = new Mat();
        //Imgproc.floodFill(img_dilate, img_fill, centroid, new Scalar(0));
        //floodFill​(Mat image, Mat mask, Point seedPoint, Scalar newVal)

        Mat img_erode = new Mat();
        Mat img_erode2 = new Mat();
        Mat erode_element = Imgproc.getStructuringElement( Imgproc.MORPH_ELLIPSE,
                new Size(3,3));
        Imgproc.erode(img_dilate,img_erode, erode_element);
        //erode​(Mat src, Mat dst, Mat kernel)
        Imgproc.erode(img_erode,img_erode2, erode_element);

        Mat img_filter = new Mat();
        double content_kernel = 1/Math.pow(21, 2);
        Mat kernel = new Mat(21,21, (int) content_kernel);
        //Mat	(	Size 	size,        int 	type,const Scalar & 	s);


        // This filtering step is not giving better results
        //Imgproc.filter2D(img_erode2, img_filter, -1, kernel);
        // filter2D(Mat src, Mat dst, int ddepth, Mat kernel)        Convolves an image with the kernel.

        /*

        //Finds the contours which in this case means the edge of the blobs
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img_filter,contours,Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        // Find contour with max contour area
        double maxContour = 0;
        List<MatOfPoint> blob = new ArrayList<MatOfPoint>();
        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint contour = each.next();
            double contourSize = Imgproc.contourArea(contour);
            if (contourSize > maxContour) {
                maxContour = contourSize;
                blob.add(contour);
            }
        }
        //Create a mask from the largest contour
        int row_size = img_filter.rows();
        int column_size = img_filter.cols();



        Mat mask = new Mat(row_size,column_size,0);
        Imgproc.fillPoly(mask,blob.get(0),1);


        // Find contour with max contour area
        double maxContour = 0;
        List<MatOfPoint> blob = new ArrayList<MatOfPoint>();
        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint contour = each.next();
            double contourSize = Imgproc.contourArea(contour);
            if (contourSize > maxContour) {
                maxContour = contourSize;
                blob.add(contour);
            }
 */


        Mat hierarchy = new Mat();
        java.util.List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img_erode2, contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);
        Scalar color = new Scalar(0,0,0);
        Imgproc.drawContours(img, contours,-1, color);


        Bitmap img_bitmap = Bitmap.createBitmap(img.cols(), img.rows(),Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(img, img_bitmap);

        ImageView imageView = findViewById(R.id.img);
        imageView.setImageBitmap(img_bitmap);




    }
}