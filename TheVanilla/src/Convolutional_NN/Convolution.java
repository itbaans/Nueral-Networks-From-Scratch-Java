package Convolutional_NN;

import java.util.Random;

import org.apache.commons.math4.legacy.linear.MatrixUtils;
import org.apache.commons.math4.legacy.linear.RealMatrix;

import Simple_NN.Activations;
import Simple_NN.Layer;

public class Convolution extends Layer {

    int sizeX;
    int sizeY;
    int sizeZ;
    int nKernels;
    int stride;

    int poolSizeX;
    int poolSizeY;
    int poolSizeZ;
    int poolStride;

    int VALID_PADDING = 0;
    int SAME_PADDING = 1;
    int FULL_PADDING = 2;

    int padType;

    RealMatrix[][] kernels;
    RealMatrix[] outChannels;

    Layer next;
    Layer prev;

    Random rand = new Random();

    //TO DO CONSTRUCTER
    public Convolution(int kernelSizeX, int kernelSizeY, int numberOfKernels, int strideSize, 
                  int poolingSizeX, int poolingSizeY, int poolingStride, int paddingType) {
        this.sizeX = kernelSizeX;
        this.sizeY = kernelSizeY;
        this.nKernels = numberOfKernels;
        this.stride = strideSize;
        
        this.poolSizeX = poolingSizeX;
        this.poolSizeY = poolingSizeY;
        this.poolStride = poolingStride;
        
        this.padType = paddingType;
    }

    @Override
    public void connect_prev(Layer l) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'connect_prev'");
    }

    @Override
    public void connect_next(Layer l) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'connect_next'");
    }
    
    //TO DO STUFF
    public RealMatrix pooling(RealMatrix chnl) {
        // Calculate output dimensions correctly
        int outputHeight = (int)Math.floor((chnl.getRowDimension() - poolSizeY)/poolStride + 1);
        int outputWidth = (int)Math.floor((chnl.getColumnDimension() - poolSizeX)/poolStride + 1);
        
        RealMatrix res = MatrixUtils.createRealMatrix(outputHeight, outputWidth);
        
        // For each output position
        for (int outRow = 0; outRow < outputHeight; outRow++) {
            for (int outCol = 0; outCol < outputWidth; outCol++) {
                // Find the max in the pooling window
                double max = Double.NEGATIVE_INFINITY;
                
                // Starting position of current pooling window
                int startRow = outRow * poolStride;
                int startCol = outCol * poolStride;
                
                // Check each element in pooling window
                for (int i = 0; i < poolSizeY; i++) {
                    for (int j = 0; j < poolSizeX; j++) {
                        if (startRow + i < chnl.getRowDimension() && startCol + j < chnl.getColumnDimension()) {
                            double val = chnl.getEntry(startRow + i, startCol + j);
                            if (val > max) {
                                max = val;
                            }
                        }
                    }
                }
                
                res.setEntry(outRow, outCol, max);
            }
        }
        
        return res;
    }

    public void initKernels(RealMatrix[] chnls) {

        sizeZ = chnls.length;

        double[][] ks = new double[sizeX][sizeY];
        kernels = new RealMatrix[nKernels][sizeZ];

        for (int k = 0; k < nKernels; k++) {
            for (int i = 0; i < sizeZ; i++) {
                ks = new double[sizeX][sizeY];  // Reset for each kernel
                for (int j = 0; j < sizeX; j++) {
                    for (int j2 = 0; j2 < sizeY; j2++) {
                        ks[j][j2] = (rand.nextDouble() * 4 - 2);
                    }
                }
                kernels[k][i] = MatrixUtils.createRealMatrix(ks);
            }
        }

        initOutChannels(chnls, padType);

    }

    private void initOutChannels(RealMatrix[] inChnls, int padType) {

        outChannels = new RealMatrix[kernels.length];

        int ph = 0;
        int pw = 0;

        int H = inChnls[0].getRowDimension();
        int W = inChnls[0].getColumnDimension();

        if (padType == SAME_PADDING) {
            ph = (int) Math.ceil((double) ((stride - 1) * H - H + sizeY) / 2);
            pw = (int) Math.ceil((double) ((stride - 1) * W - W + sizeX) / 2);
        } else if (padType == FULL_PADDING) {
            ph = sizeY - 1;
            pw = sizeX - 1;
        }

        for (int k = 0; k < kernels.length; k++) {
            outChannels[k] = convo3d(inChnls, kernels[k], stride, ph, pw);
            outChannels[k] = activation(outChannels[k]);
            outChannels[k] = pooling(outChannels[k]);
        }  

    }

    private RealMatrix activation(RealMatrix chnl) {

        RealMatrix res = chnl.copy();

        int r = chnl.getRowDimension();
        int c = chnl.getColumnDimension();

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                double val = Activations.reLu(chnl.getEntry(i, j));
                res.setEntry(i, j, val);
            } 
        }

        return res;
    }

    private RealMatrix convo3d(RealMatrix[] channel, RealMatrix[] kernel, int stride, int padHeight, int padWidth) {
        if (channel.length != kernel.length) return null; // Ensure the number of channels matches the number of kernels
    
        int z = channel.length;
    
        // Apply padding to all channels before starting the convolution
        RealMatrix[] paddedChannels = new RealMatrix[z];
        for (int sz = 0; sz < z; sz++) {
            paddedChannels[sz] = padMatrix(channel[sz], padHeight, padWidth);
        }
    
        int r = kernel[0].getRowDimension();
        int c = kernel[0].getColumnDimension();
    
        // Calculate output dimensions based on the padded matrix
        int outputRows = (paddedChannels[0].getRowDimension() - r) / stride + 1;
        int outputCols = (paddedChannels[0].getColumnDimension() - c) / stride + 1;
    
        // Create the output matrix
        RealMatrix output = MatrixUtils.createRealMatrix(outputRows, outputCols);
    
        for (int outRow = 0; outRow < outputRows; outRow++) {
            for (int outCol = 0; outCol < outputCols; outCol++) {
                double entry = 0;
    
                // Perform convolution across all channels
                for (int sz = 0; sz < z; sz++) {
                    for (int i = 0; i < r; i++) {
                        for (int j = 0; j < c; j++) {
                            entry += paddedChannels[sz].getEntry(i + (outRow * stride), j + (outCol * stride)) 
                                   * kernel[sz].getEntry(i, j);
                        }
                    }
                }
    
                // Store the computed entry in the output matrix
                output.setEntry(outRow, outCol, entry);
            }
        }
    
        return output;
    }

    private RealMatrix padMatrix(RealMatrix matrix, int padHeight, int padWidth) {
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
    
}
