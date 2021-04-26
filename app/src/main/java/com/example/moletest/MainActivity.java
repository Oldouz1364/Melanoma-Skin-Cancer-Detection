
package com.example.moletest;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.google.firebase.auth.FirebaseAuth;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import java.lang.Math;

import java.io.IOException;
import java.util.AbstractSequentialList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Vector;


public class MainActivity extends AppCompatActivity {
    private static final String LOG_TAG = "Main Activity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        OpenCVLoader.initDebug();
    }

    public void logout (View view) {
        FirebaseAuth.getInstance().signOut();//logout
        startActivity(new Intent(getApplicationContext(),Login.class));
        finish();
    }


    public void ImageProcessing (View v){

        Mat img = null;


        try {
            img = Utils.loadResource(getApplicationContext(),R.drawable.a);
            Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2BGRA);
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
        //Point centroid = new Point(Math.floor(M.m10 / (M.m00 + 1e-5)), Math.floor(M.m01 / (M.m00 + 1e-5)));//add 1e-5 to avoid division by zero

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



        /*
        // This filtering step is not giving better results
        Mat img_filter = new Mat();
        double content_kernel = 1/Math.pow(21, 2);
        Mat kernel = new Mat(21,21, (int) content_kernel);
        //Mat	(	Size 	size,        int 	type,const Scalar & 	s);

        Imgproc.filter2D(img_erode2, img_filter, -1, kernel);
        // filter2D(Mat src, Mat dst, int ddepth, Mat kernel)        Convolves an image with the kernel.
        */


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
/*
        int largest_area=0;
        int largest_contour_index=0;

        java.util.List<MatOfPoint> contours1 = new ArrayList<MatOfPoint>(); // Vector for storing contour
        MatOfInt4 hierarchy = new MatOfInt4();

        Imgproc.findContours( img_erode2, contours1, hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE); // Find the contours in the image

        for( int i = 0; i< contours1.size(); i++ ) {// iterate through each contour.
            double contourA=Imgproc.contourArea(contours1.get(i),false);  //  Find the area of contour
            if(contourA>largest_area){
                largest_area= (int) contourA;
                largest_contour_index=i;                //Store the index of largest contour
            }
        }
        java.util.List<MatOfPoint> mainContour = (List<MatOfPoint>) contours1.get(largest_contour_index);
        //Scalar color = new Scalar(0,255,0);
        //Imgproc.drawContours( img, contours1, largest_contour_index, color ); // Draw the largest contour using previously stored index.


        //Create a mask from the largest contour
        int row_size = img_erode2.rows();
        int column_size = img_erode2.cols();
        Mat mask = new Mat(row_size,column_size, CvType.CV_8UC1, new Scalar(0));
        Scalar color_white = new Scalar (255,255,255);
        Imgproc.fillPoly(mask, mainContour,color_white);
        */

        /*
        // Get the contour of the region or interest - Not necessary because already done above
        Mat hierarchy = new Mat();
        Scalar color = new Scalar(0,255,0);
        java.util.List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(img_erode2, contours,hierarchy,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);
        Imgproc.drawContours(img, contours,-1, color);
        */

        Mat masked_img = new Mat(img.size(), CvType.CV_8UC3, new Scalar(255, 255, 255)); // Bcs color image
        img.copyTo(masked_img, img_erode2);

        /*
        // Display the image
        Bitmap img_bitmap = Bitmap.createBitmap(masked_img.cols(), masked_img.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(masked_img,img_bitmap);


        ImageView imageView = findViewById(R.id.img);
        imageView.setImageBitmap(img_bitmap);

         */
 
        // Calculation of features
        if (img_erode2 != null) {

            // Lesion border features


            // Compactness
            // First finding the contour
            Mat hierarchy = new Mat();
            java.util.List<MatOfPoint> contour = new ArrayList<MatOfPoint>();
            Imgproc.findContours(img_erode2, contour, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
            // Second calculating the area of the contour
            double Area = Imgproc.contourArea(contour.get(0));
            // Third calculating the perimeter of the contour
            MatOfPoint2f new_contour = new MatOfPoint2f(contour.get(0).toArray());// New variable
            double Perimeter = Imgproc.arcLength(new_contour, true);
            double compactness = 1 - (4 * Math.PI * Area) / Math.pow(Perimeter, 2);
            //String str_compactness = compactness + "";

            // Solidity
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour.get(0), hull); // hull is of type MatOf Int
            // From MatOfInt to MatOfPoint
            List<Point> hullPointList = new ArrayList<Point>();
            MatOfPoint hullPointMat = new MatOfPoint();
            List<MatOfPoint> hullPoints = new ArrayList<MatOfPoint>();
            for (int j = 0; j < hull.toList().size(); j++) {
                hullPointList.add(contour.get(0).toList().get(hull.toList().get(j)));
            }

            hullPointMat.fromList(hullPointList);
            hullPoints.add(hullPointMat);

            // Draw contours + hull results
            int row_size = img_erode2.rows();
            int column_size = img_erode2.cols();
            Mat drawing = new Mat(row_size, column_size, CvType.CV_8UC1, new Scalar(0));
            Scalar color = new Scalar(0, 255, 0);
            Imgproc.drawContours(drawing, hullPoints, -1, color);
            double hull_area = Imgproc.contourArea(hullPointMat);
            double solidity = Area / hull_area;
            //String str_solidity = solidity + "";



            //Variance of distances from border points to centroid of lesion
            // Calculation of centroid

            Moments M = Imgproc.moments(img_erode2); // calculate moments of binary image
            Point centroid = new Point(M.m10 / (M.m00 + 1e-5), M.m01 / (M.m00 + 1e-5));//add 1e-5 to avoid division by zero
            List<Double> distances = new ArrayList<Double>();
            for (int i = 0; i < contour.get(0).rows(); i++) {
                distances.add(Math.sqrt(Math.pow((contour.get(0).get(i, 0)[0] - centroid.getX()),2) + Math.pow((contour.get(0).get(i, 0)[1] - centroid.getY()),2 )));
            }
            // Calculation of the variance of distances
            int size_distances = distances.size();

            double sum = 0.0;
            for(double d : distances){
                    sum += d;
            }
            double mean_distances = sum/size_distances;

            double temp = 0;
            for (double d : distances){
                    temp += (d - mean_distances) * (d - mean_distances);
            }
            double var_distances = temp / (size_distances - 1);
            String str_var_distances = var_distances + "";


            //Assymetry

            // Finding the orientation of the blob
            //https://stackoverflow.com/questions/14720722/binary-image-orientation
            //https://en.wikipedia.org/wiki/Image_moment (Explanation on calculation of orientation with central moments)
            double orientation;
            if (M.mu20-M.mu02 != 0) {
                orientation = 0.5 * Math.atan(2 * M.mu11 / (M.mu20 - M.mu02));
                orientation = (orientation/Math.PI)*180; // Angle in degrees
            }
            else {
                orientation = 0;
            }
            //Creating destination matrix
            Mat rotated_mat = new Mat(row_size, column_size, img_erode2.type());
            //Creating the transformation matrix
            Mat rotationMatrix = Imgproc.getRotationMatrix2D(centroid, orientation, 1);
            // Creating the object of the class Size
            Size size = new Size(row_size, column_size);
            // Rotating the given image https://www.tutorialspoint.com/how-to-rotate-an-image-with-opencv-using-java
            Imgproc.warpAffine(img_erode2, rotated_mat, rotationMatrix, size);

            // Dividing the image in top/bottom and right/left
            Mat rotated_mat_top;
            int centroid_pos_Y = (int) centroid.getY();
            int centroid_pos_X = (int) centroid.getX();
            Mat black_bottom = new Mat(row_size, column_size, CvType.CV_8UC1, new Scalar(0));
            rotated_mat.copyTo(black_bottom);
            Mat black_top = new Mat(row_size, column_size, CvType.CV_8UC1, new Scalar(0));
            rotated_mat.copyTo(black_top);
            Mat black_right = new Mat(row_size, column_size, CvType.CV_8UC1, new Scalar(0));
            rotated_mat.copyTo(black_right);
            Mat black_left = new Mat(row_size, column_size, CvType.CV_8UC1, new Scalar(0));
            rotated_mat.copyTo(black_left);

            // Manipulation of black_bottom
            for (int i=centroid_pos_Y+1;i<=row_size;i++){
                for (int j=0;j<= column_size;j++) {
                    black_bottom.put(i, j, 0);
                }
            }
            // Manipulation of black_top
            for (int i=0;i<=centroid_pos_Y-1;i++){
                for (int j=0;j<= column_size;j++) {
                    black_top.put(i, j, 0);
                }
            }
            // Manipulation of black_right
            for (int i=0;i<=row_size;i++){
                for (int j=centroid_pos_X+1;j<= column_size;j++) {
                    black_right.put(i, j, 0);
                }
            }
            // Manipulation of black_left
            for (int i=0;i<=row_size;i++){
                for (int j=0;j<= centroid_pos_X-1;j++) {
                    black_left.put(i, j, 0);
                }
            }
            // Calculating area of black_bottom
            java.util.List<MatOfPoint> contour_black_bottom = new ArrayList<MatOfPoint>();
            Imgproc.findContours(black_bottom, contour_black_bottom, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
            // Second calculating the area of the contour
            double Area_black_bottom = Imgproc.contourArea(contour_black_bottom.get(0));

            // Calculating area of black_top
            java.util.List<MatOfPoint> contour_black_top = new ArrayList<MatOfPoint>();
            Imgproc.findContours(black_top, contour_black_top, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
            // Second calculating the area of the contour
            double Area_black_top = Imgproc.contourArea(contour_black_top.get(0));

            // Calculating area of black_right
            java.util.List<MatOfPoint> contour_black_right = new ArrayList<MatOfPoint>();
            Imgproc.findContours(black_right, contour_black_right, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
            // Second calculating the area of the contour
            double Area_black_right = Imgproc.contourArea(contour_black_right.get(0));

            // Calculating area of black_left
            java.util.List<MatOfPoint> contour_black_left = new ArrayList<MatOfPoint>();
            Imgproc.findContours(black_left, contour_black_left, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
            // Second calculating the area of the contour
            double Area_black_left = Imgproc.contourArea(contour_black_left.get(0));

            double Ax = Math.abs(Area_black_top - Area_black_bottom);
            double Ay = Math.abs(Area_black_left - Area_black_right);
            double Assymetry = (Ax + Ay)/Area;
            String str_test = Assymetry + "";


            //Lesion Color Feature (LCF)

            // RGB - Mean and standard deviation

            // Splitting the masked bgr image in color channels
            //List<Mat> bgr = new ArrayList<>();
            //Core.split(masked_img, bgr);
            //Mat blue = bgr.get(0);
            //Mat green = bgr.get(1);
            //Mat red = bgr.get(2);

            MatOfDouble mean_masked_image= new MatOfDouble();
            MatOfDouble std_masked_image= new MatOfDouble();

            Core.meanStdDev(masked_img, mean_masked_image, std_masked_image);

            Log.d("meanval1", String.valueOf(mean_masked_image.get(0,0)[0]));
            Log.d("meanval2", String.valueOf(mean_masked_image.get(1,0)[0]));
            Log.d("meanval3", String.valueOf(mean_masked_image.get(2,0)[0]));
            Log.d("stdval1", String.valueOf(std_masked_image.get(0,0)[0]));
            Log.d("stdval2", String.valueOf(std_masked_image.get(1,0)[0]));
            Log.d("stdval3", String.valueOf(std_masked_image.get(2,0)[0]));

            double blue_mean = mean_masked_image.get(0,0)[0];
            double green_mean = mean_masked_image.get(1,0)[0];
            double red_mean = mean_masked_image.get(2,0)[0];

            double blue_std = std_masked_image.get(0,0)[0];
            double green_std = std_masked_image.get(1,0)[0];
            double red_std = std_masked_image.get(2,0)[0];

            // Grayscale - Mean and standard deviation
            Mat masked_gray_img = new Mat(imggray.size(), CvType.CV_8UC1, new Scalar(255));
            imggray.copyTo(masked_gray_img, img_erode2);

            MatOfDouble mean_masked_gray_image= new MatOfDouble();
            MatOfDouble std_masked_gray_image= new MatOfDouble();

            Core.meanStdDev(masked_gray_img, mean_masked_gray_image, std_masked_gray_image);

            Log.d("meanval_gray", String.valueOf(mean_masked_gray_image.get(0,0)[0]));
            Log.d("stdval_gray", String.valueOf(std_masked_gray_image.get(0,0)[0]));

            double gray_mean = mean_masked_gray_image.get(0,0)[0];
            double gray_std = std_masked_gray_image.get(0,0)[0];

            // HSV - Mean and standard deviation
            Mat masked_HSV_img = new Mat();
            Imgproc.cvtColor(masked_img, masked_HSV_img, Imgproc.COLOR_BGR2HSV);

            MatOfDouble mean_masked_HSV_image= new MatOfDouble();
            MatOfDouble std_masked_HSV_image= new MatOfDouble();

            Core.meanStdDev(masked_HSV_img, mean_masked_HSV_image, std_masked_HSV_image);

            Log.d("meanval1_HSV", String.valueOf(mean_masked_HSV_image.get(0,0)[0]));
            Log.d("meanval2_HSV", String.valueOf(mean_masked_HSV_image.get(1,0)[0]));
            Log.d("meanval3_HSV", String.valueOf(mean_masked_HSV_image.get(2,0)[0]));
            Log.d("stdval1_HSV", String.valueOf(std_masked_HSV_image.get(0,0)[0]));
            Log.d("stdval2_HSV", String.valueOf(std_masked_HSV_image.get(1,0)[0]));
            Log.d("stdval3_HSV", String.valueOf(std_masked_HSV_image.get(2,0)[0]));

            double blue_mean_HSV = mean_masked_HSV_image.get(0,0)[0];
            double green_mean_HSV = mean_masked_HSV_image.get(1,0)[0];
            double red_mean_HSV = mean_masked_HSV_image.get(2,0)[0];

            double blue_std_HSV = std_masked_HSV_image.get(0,0)[0];
            double green_std_HSV = std_masked_HSV_image.get(1,0)[0];
            double red_std_HSV = std_masked_HSV_image.get(2,0)[0];


            // Display the image
            Bitmap img_bitmap = Bitmap.createBitmap(masked_gray_img.cols(), masked_gray_img.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(masked_gray_img,img_bitmap);


            ImageView imageView = findViewById(R.id.img);
            imageView.setImageBitmap(img_bitmap);

            TextView textView = findViewById(R.id.Feature);
            textView.setText(str_test);


        }
    }
}