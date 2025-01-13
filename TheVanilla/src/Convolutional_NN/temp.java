package Convolutional_NN;
//import Simple_NN.*;

import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;

public class temp {

    public static void main(String[] args) {
        
        
        // Example input matrix (channel)
        RealMatrix channel = MatrixUtils.createRealMatrix(new double[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        // Example kernel
        RealMatrix kernel = MatrixUtils.createRealMatrix(new double[][]{
            {1, 0},
            {0, -1}
        });

        int stride = 2;
        int padding = 1; // Example: 1-pixel padding

        // Perform convolution
        RealMatrix result = convo2d(channel, kernel, stride, padding);

        // Print the result
        for (int i = 0; i < result.getRowDimension(); i++) {
            for (int j = 0; j < result.getColumnDimension(); j++) {
                System.out.print(result.getEntry(i, j) + " ");
            }
            System.out.println();
        }

    }

    //channel -> NN
    //channel -> c1 -> channel' -> c2 -> channel'' .... cn -> cahnneln -> NN

    //2d convolution
    public static RealMatrix padMatrix(RealMatrix matrix, int padHeight, int padWidth) {
        int originalRows = matrix.getRowDimension();
        int originalCols = matrix.getColumnDimension();

        // Create a padded matrix with zeros
        int newRows = originalRows + 2 * padHeight;
        int newCols = originalCols + 2 * padWidth;
        RealMatrix paddedMatrix = MatrixUtils.createRealMatrix(newRows, newCols);

        // Copy the original matrix into the center of the padded matrix
        for (int i = 0; i < originalRows; i++) {
            for (int j = 0; j < originalCols; j++) {
                paddedMatrix.setEntry(i + padHeight, j + padWidth, matrix.getEntry(i, j));
            }
        }

        return paddedMatrix;
    }

    public static RealMatrix convo2d(RealMatrix channel, RealMatrix kernel, int stride, int padding) {
        
        // Apply padding to the input matrix
        channel = padMatrix(channel, padding, padding);

        int r = kernel.getRowDimension();
        int c = kernel.getColumnDimension();

        int outputRows = (channel.getRowDimension() - r) / stride + 1;
        int outputCols = (channel.getColumnDimension() - c) / stride + 1;

        // Create the output matrix
        RealMatrix output = MatrixUtils.createRealMatrix(outputRows, outputCols);

        for (int outRow = 0; outRow < outputRows; outRow++) {
            for (int outCol = 0; outCol < outputCols; outCol++) {
                double entry = 0;
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        entry += channel.getEntry(i + (outRow * stride), j + (outCol * stride)) * kernel.getEntry(i, j);
                    }
                }
                // Store the computed entry in the output matrix
                output.setEntry(outRow, outCol, entry);
            }
        }

        return output;
    }

}